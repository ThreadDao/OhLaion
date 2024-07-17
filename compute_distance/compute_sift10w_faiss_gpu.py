# faiss-gpu maybe need pythjon3.9
# conda install: conda install -c pytorch faiss-gpu
import faiss
import numpy as np
from sklearn import preprocessing
import pandas as pd
from utils.utils_log import log


def get_gt(metric_type, xb, xq, k, g_device):
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = g_device
    res = faiss.StandardGpuResources()
    index = None
    log.info("start faiss index")
    if metric_type == "L2":
        # index = faiss.IndexFlatL2(xb.shape[1])
        index = faiss.GpuIndexFlatL2(res, xb.shape[1], flat_config)

    index.add(xb)
    log.info("start faiss search")
    return index.search(xq, k)


def gen_file_link(epoch):
    url = "/home/zong/data/wiki"
    return f"{url}/wikipedia_{epoch:05d}.parquet"


def gen_gnd(train_data_range, query_data_path, top_k=1000, expr=None, gpu_device=0):
    total = 0
    # read query
    _xq = pd.read_parquet(query_data_path, columns=["emb"])
    log.debug(f"xq shape: {_xq.shape}")
    log.debug(f"xq type: {type(_xq)}, xq[0] type: {type(_xq['emb'][0][0])}")
    xq = np.array(_xq['emb'].tolist(), dtype=np.float32)
    del _xq

    # read train
    for i in range(train_data_range[0], train_data_range[1]):
        train_data_path = gen_file_link(i)
        log.info(f"train {i}")
        if expr is not None:
            category_df = pd.read_parquet(train_data_path, columns=["id", "category", "tags"])
            category_df.query(expr=expr, inplace=True)
            if category_df.shape[0] == 0:
                continue
        df = pd.read_parquet(train_data_path, columns=["emb", "id", "category", "tags"])
        # filter train data with expr
        if expr is not None:
            df.query(expr=expr, inplace=True)
        idx_df = df["id"]
        log.debug(f"{i} train dataframe shape: {df.shape}")
        total += df.shape[0]
        xb = np.array(df['emb'].tolist(), dtype=np.float32)

        # concatenate and normalize
        log.debug(f"xb shape: {xb.shape}")
        if xb.shape[0] == 0:
            continue
        xb = preprocessing.normalize(xb, axis=1, norm="l2")
        log.debug(f"xb shape after normalize: {xb.shape}")

        # faiss index and search
        distance, ids = get_gt("L2", xb, xq, top_k, gpu_device)
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
        dis_path = f'/home/zong/data/wiki/distance/distance_{i:05d}.npy'
        np.save(dis_path, gnd_ids)
        log.info(f"until train_{i}, total hits {total}")


if __name__ == '__main__':
    train_file_range = [0, 100]
    query_path = "/home/zong/data/wiki/wikipedia_query_set.parquet"
    _top_k = 1000
    _expr = "id > 100"
    _gpu_device = 0
    gen_gnd(train_file_range, query_path, _top_k, _expr, _gpu_device)
