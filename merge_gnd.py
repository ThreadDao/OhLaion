import os
from datetime import datetime
from multiprocessing import Pool
import numpy as np


def ivecs_write(fname, m):
    n, d = m.shape
    print(f"gnd shape: {n}, {d}")
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    print(m1)
    m1.tofile(fname)


def read_topk_from_npy(npy_path):
    if os.path.exists(npy_path):
        print(f"read file {npy_path}")
        return np.load(npy_path, allow_pickle=True)
    else:
        return


def merge_top_k(wait_merge_top_k_list, top_k):
    gnds = np.concatenate(wait_merge_top_k_list, axis=1)
    print(f"{datetime.now()} merge top k concat done")
    sorted_result = np.sort(gnds, axis=1, kind='mergesort', order=["distance"])
    print(f"{datetime.now()} merge top k sort done")
    return sorted_result[:, :top_k]


def auto_merge_gnd(url, start_epoch, end_epoch, top_k):
    step = 500
    init_top_sorted_result = None
    for index in range(start_epoch, end_epoch, step):
        init_top_sorted_result = merge_gnd(url, index, min(index + step, end_epoch), top_k, init_top_sorted_result)

    np.save(f"{url}/dis_idx_{start_epoch}_{end_epoch}.npy", init_top_sorted_result)


def merge_gnd(url, start_epoch, end_epoch, top_k, init_top_sorted_result):
    with Pool() as pool:
        print(f"{datetime.now()} load pool = {pool}")
        all_npy_top_k = pool.starmap(read_topk_from_npy,
                                     [(f"{url}/distance_{index:04d}.npy", top_k) for index in
                                      range(start_epoch, end_epoch) if
                                      os.path.exists(f"{url}/distance_{index:04d}.npy")])
    print(f"{datetime.now()} all npy top k load done")
    print(f'all_npy_top_k len {len(all_npy_top_k)}')

    if init_top_sorted_result is not None:
        all_npy_top_k.append(init_top_sorted_result)

    once_merge_item_num = 5
    while len(all_npy_top_k) != 1:
        wait_merge_top_k_lists = np.array_split(all_npy_top_k, 1 + (len(all_npy_top_k) - 1) / once_merge_item_num)
        print(f"{datetime.now()} merge top k, size = {len(all_npy_top_k)}, split = {len(wait_merge_top_k_lists)}")
        with Pool() as pool:
            print(f"{datetime.now()} merge pool = {pool}")
            all_npy_top_k = pool.starmap(merge_top_k,
                                         [(wait_merge_top_k_list, top_k) for wait_merge_top_k_list in
                                          wait_merge_top_k_lists])

    top_sorted_result = all_npy_top_k[0]
    print(f"{datetime.now()} merge top k done")
    print(f"topk sorted shape: {top_sorted_result.shape}, topk[0] shape: {top_sorted_result[0].shape}")
    print(top_sorted_result)

    # result = np.empty(top_sorted_result.shape, dtype=[('idx', "i8"), ('distance', "f8")])
    # only need id�~Lso�~Hid, distance�~I--> id
    # for index, value in np.ndenumerate(top_sorted_result):
    #     result[index] = value
    # print(result.shape)
    # print(result)
    # ivecs_write('/test/raw_data/laion5b_parquet/gnd/idx_2M.ivecs', result)
    return top_sorted_result


if __name__ == '__main__':
    url = "/test/xxx/raw_data/laion5b_parquet/laion1B-nolang/distance_top1000"
    auto_merge_gnd(url, start_epoch=0, end_epoch=1271, top_k=1000)
