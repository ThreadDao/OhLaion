# faiss-gpu maybe need pythjon3.9
# conda install: conda install -c pytorch faiss-gpu
import faiss
import numpy as np
from sklearn import preprocessing
import pandas as pd
from utils.utils_log import log


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
        log.debug(f"index: {index}")

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
    flat_config.device = 1
    res = faiss.StandardGpuResources()
    index = None
    log.info("start faiss index")
    if metric_type == "L2":
        index = faiss.GpuIndexFlatL2(res, xb.shape[1], flat_config)

    index.add(xb)
    log.info("start faiss search")
    return index.search(xq, k)


def get_ground_truth_ids():
    gnd_file_name = "/test/raw_data/laion5b_parquet/gnd/faiss_idx_2M.ivecs"
    a = np.fromfile(gnd_file_name, dtype='int32')
    d = a[0]
    true_ids = a.reshape(-1, d + 1)[:, 1:].copy()
    return true_ids


def ivecs_write(fname, m):
    n, d = m.shape
    print(f"gnd shape: {n}, {d}")
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    print(m1)
    m1.tofile(fname)


def gen_file_link(epoch):
    url = "/test/raw_data/laion5b_parquet"
    return f"{url}/binary_768d_{epoch:05d}.parquet"


if __name__ == '__main__':
    # var
    nq = 10000
    top_k = 1000

    # read query
    xq = np.load("/test/raw_data/laion5b_parquet/query.npy")
    xq = xq.astype(np.float32)
    log.debug(f"xq shape: {xq.shape}")
    log.debug(f"xq type: {type(xq)}, xq[0] type: {type(xq[0])}")

    # read train
    for i in range(0, 1000):
        train_data_path = gen_file_link(i)
        df = pd.read_parquet(train_data_path, columns=["float32_vector", "pk"])
        idx_df = df["pk"]
        log.debug(f"{i} train dataframe shape: {df.shape}")
        xb = np.array(df['float32_vector'].tolist(), dtype=np.float32)

        # n2 = np.array(p2['float32_vector'][:66454].tolist(), dtype=np.float32)

        # concatenate and normalize
        log.debug(f"xb shape: {xb.shape}")
        xb = preprocessing.normalize(xb, axis=1, norm="l2")
        log.debug(f"xb shape after normalize: {xb.shape}")

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
        dis_path = f'/test/raw_data/laion5b_parquet/distance_top1000_faiss/distance_{i:05d}.npy'
        np.save(dis_path, gnd_ids)
    # np.save("/data/gnds/2m_top1000_dis.npy", distance)
    # ivecs_write("/test/raw_data/laion5b_parquet/gnd/faiss_gpu_idx_2M.ivecs", ids)

    # true_ids = get_ground_truth_ids()
    # recall = get_recall_value(true_ids[:10000, :top_k], ids)
    # log.info(f"recall: {recall}")
