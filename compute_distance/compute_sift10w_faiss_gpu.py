# faiss-gpu maybe need pythjon3.9
# conda install: conda install -c pytorch faiss-gpu

import faiss
import numpy as np
import pandas as pd
from utils.utils_log import log
from utils.utils_data import ivecs_write, fvecs_write, ivecs_read, fvecs_read


def get_recall_value(true_ids, result_ids):
    """
    Use the intersection length
    """
    sum_radio = 0.0
    topk_check = True
    log.debug(f"result_ids len: {len(result_ids)}")
    log.debug(f"result topk len: {len(result_ids[0])}")
    for index, item in enumerate(result_ids):
        # log.debug("[get_recall_value] true_ids: {}".format(true_ids[index]))
        # log.debug("[get_recall_value] result_ids: {}".format(item))
        # log.debug(f"index: {index}")

        # _tmp = set(item).difference(set(true_ids[index]))
        # if len(_tmp) != 0:
        #     log.debug("[get_recall_value] ids in result_ids but not in true_ids: {}".format(_tmp))

        tmp = set(true_ids[index]).intersection(set(item))
        if len(item) != 0:
            sum_radio += len(tmp) / len(item)
            # log.debug(f"sum_radio: {sum_radio}")
        else:
            topk_check = False
            log.error("[get_recall_value] Length of returned topk is 0, please check.")
    if topk_check is False:
        raise ValueError("[get_recall_value] The result of topk is wrong, please check: {}".format(result_ids))
    return round(sum_radio / len(result_ids), 3)


def get_gt(metric_type, xb, xq, k):
    flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 1
    res = faiss.StandardGpuResources()
    index = None
    log.info("start faiss index")
    if metric_type == "L2":
        index = faiss.GpuIndexFlatL2(res, xb.shape[1], flat_config)

    index.add(xb)
    log.info("start faiss search")
    return index.search(xq, k)


def gen_gnd(train_data_path, query_data_path, top_k=10, expr=None):
    # var
    # read query
    xq = np.load(query_data_path)
    xq = xq.astype(np.float32)
    log.debug(f"xq shape: {xq.shape}")
    log.debug(f"xq type: {type(xq)}, xq[0] type: {type(xq[0])}")

    # read train
    # read npy
    train_npy = np.load(train_data_path)
    train_npy = train_npy.astype(dtype=np.float32)
    log.info(train_npy.shape)
    log.info(train_npy.dtype)
    # gen train df
    int_values = pd.Series(data=[i for i in range(0, train_npy.shape[0])])
    df = pd.DataFrame({"id": int_values})

    df["float32_vector"] = train_npy.tolist()
    log.debug(f"train dataframe shape before expr: {df.shape}")
    if expr is not None:
        df.query(expr=expr, inplace=True)
    log.debug(f"train dataframe shape after expr: {df.shape}")
    log.debug(df)
    idx_df = df["id"]
    log.info(idx_df)
    log.debug(idx_df.shape)
    xb = np.array(df['float32_vector'].tolist(), dtype=np.float32)
    log.debug(f"xb shape: {xb.shape}")

    # concatenate and not need normalize
    # faiss index and search
    distance, ids = get_gt("L2", xb, xq, top_k)
    log.info(f"distance shape: {distance.shape}")
    log.info(f"ids shape: {ids.shape}")

    """
    # gnd_ids: [ [(pk, distance), ...], ...]
    """
    gnd_ids = np.empty(distance.shape, dtype=[('idx', "i4"), ('distance', "f4")])
    for index, value in np.ndenumerate(ids):
        gnd_ids[index] = (idx_df.iloc[value], distance[index])
    log.info(f"gnd_ids shape: {gnd_ids.shape} {gnd_ids[0].shape}")
    log.info(gnd_ids)
    idx = np.empty(gnd_ids.shape, dtype="i4")
    dis = np.empty(gnd_ids.shape, dtype="f4")
    for index, value in np.ndenumerate(gnd_ids):
        idx[index] = value[0]
        dis[index] = value[1]

    idx_path = '~/gnd/idx_0M.ivecs'
    dis_path = '~/gnd/dis_0M.fvecs'
    ivecs_write(idx_path, idx)
    fvecs_write(dis_path, dis)

    # read
    log.info(ivecs_read(idx_path))
    log.info(fvecs_read(dis_path))
    # true_ids = ivecs_read(idx_path)
    # recall = get_recall_value(true_ids[:10000, :top_k], ids)
    # log.info(f"recall: {recall}")


def parser_data_size(data_size):
    return eval(str(data_size)
                .replace("k", "*1000")
                .replace("w", "*10000")
                .replace("m", "*1000000")
                .replace("b", "*1000000000")
                )


if __name__ == '__main__':
    query_path = "/test/milvus/raw_data/sift10w/query.npy"
    train_path = "/test/milvus/raw_data/sift10w/binary_128d_00000.npy"
    gen_gnd(train_path, query_path, top_k=1000, expr='id >= 200')

    # # recall
    # result_ids = milvus.search
    # idx_path = '/home/zong/Downloads/gnd/idx_0M.ivecs'
    # true_ids = ivecs_read(idx_path)
    # recall = get_recall_value(true_ids[:10000, :1000], result_ids)
    # log.info(f"recall: {recall}")
