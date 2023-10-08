import time

import torch
import numpy as np
import pandas as pd
from torchmetrics.functional import pairwise_cosine_similarity
from utils.utils_log import log


def gen_path(ptype, index):
    if ptype == "train":
        return f'/test/raw_data/laion5b_parquet/binary_768d_{index:05d}.parquet'
    elif ptype == "dis":
        return f'/test/raw_data/laion5b_parquet/distance_top1000/dis_{index:05d}.npy'
    else:
        raise Exception(f"Now only support ptype: [train, dis]")


def compute_dis(test_npy, train_data_gpu, top_k=10):
    torch.set_printoptions(8)
    # read train parquet file and test npy

    test_data_gpu = torch.tensor(test_npy, device='cuda', dtype=torch.float32)

    start = time.time()
    # distance = torch.cdist(train_data_gpu.norm(), test_data_gpu.norm(), p=2.0)
    distance = pairwise_cosine_similarity(train_data_gpu, test_data_gpu)
    log.info(f"calculate cost is {time.time() - start}")
    log.info(f"distance shape: {distance.shape}")
    # log.info(f"distance[0]: {distance[0]}")

    # transpose distance matrix to [test_npy.shape, len(train_df)]
    transpose_distance = torch.transpose(distance, 0, 1)
    log.info(f"transpose distance shape: {transpose_distance.shape}")
    # log.info(f"transpose distance[0]: {transpose_distance[0]}")
    log.info(f"distance shape: {transpose_distance.shape}")

    # sort
    sorted_data_descending, indices_descending = torch.sort(transpose_distance, dim=1, descending=True)
    log.info(f"sort distance shape: {sorted_data_descending.shape}")
    # log.info(f"sort distance[0]: {sorted_data_descending[0]}")
    log.info(f"sort indices shape: {indices_descending.shape}")
    # log.info(f"sort indices[0]: {indices_descending[0]}")

    # distance = distance_matrix.cpu().numpy()
    top_k_distance_index = indices_descending[:, :top_k]
    top_k_distance_value = sorted_data_descending[:, :top_k]
    log.info(f"top_k distance shape: {top_k_distance_index.shape}")
    # log.info(f"top_k distance[0]: {top_k_distance[0]}")
    # idx = train_df["pk"].tolist()

    top_k_distance_index = top_k_distance_index.cpu().numpy()
    top_k_distance_value = top_k_distance_value.cpu().numpy()
    return top_k_distance_index, top_k_distance_value


def compute_train_file(test_npy, train_path, dis_path, top_k):
    group_size = 100

    # read train parquet file and test npy
    origin_train_df = pd.read_parquet(train_path, columns=["float32_vector", "pk"])
    log.info(f"train data {origin_train_df.shape}")
    grouped_dfs = [test_npy[i:i + group_size] for i in range(0, len(test_npy), group_size)]
    all_distance = []
    all_transpose_distance = []
    train_array = np.array(origin_train_df['float32_vector'].tolist(), dtype=np.float32)
    train_data_gpu_0 = torch.tensor(train_array, device='cuda', dtype=torch.float32)

    for group_df in grouped_dfs:
        group_top_k_distance, group_transpose_distance = compute_dis(group_df, train_data_gpu_0, top_k)
        all_distance.append(group_top_k_distance)
        all_transpose_distance.append(group_transpose_distance)

    # find the top_k ids
    top_k_distance = np.concatenate(all_distance, axis=0)
    # log.info(top_k_distance)
    transpose_distance = np.concatenate(all_transpose_distance, axis=0)
    # log.info(transpose_distance)

    idx = origin_train_df["pk"].tolist()
    # log.info("init idx")
    gnd_ids = np.empty(top_k_distance.shape, dtype=[('idx', "i8"), ('distance', "f8")])
    for index, value in np.ndenumerate(top_k_distance):
        gnd_ids[index] = (idx[value], transpose_distance[index])
    log.info(f"gnd_ids shape: {gnd_ids.shape} {gnd_ids[0].shape}")
    log.info(gnd_ids)
    np.save(dis_path, gnd_ids)
    log.info(f"save file {dis_path.split('/')[-1]} succ")


if __name__ == '__main__':
    test_path = '/test/raw_data/laion5b_parquet/query.npy'
    test_npy = np.load(test_path, allow_pickle=True)
    log.info(f"test data {test_npy.shape}")

    for epoch in range(1906, 2000):
        log.info(f"start process {epoch} file")
        train_path = gen_path("train", epoch)
        dis_path = gen_path("dis", epoch)
        compute_train_file(test_npy, train_path, dis_path, top_k=1000)
        log.info(f"finish process {epoch} file")
