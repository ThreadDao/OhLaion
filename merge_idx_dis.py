import numpy as np
from utils.utils_data import ivecs_write, fvecs_write


def merge_gnd(url, top_k):
    gnds = np.load(f"{url}/5b_distance_05271.npy", allow_pickle=True)
    gnds_1032 = np.load(f"{url}/dis_idx_0_1032.npy", allow_pickle=True)
    gnds_3567 = np.load(f"{url}/dis_idx_1032_3567.npy", allow_pickle=True)
    gnds_5271 = np.load(f"{url}/dis_idx_3567_5271.npy", allow_pickle=True)
    gnds = np.concatenate((gnds, gnds_1032, gnds_3567, gnds_5271), axis=1)

    print(f"all gnd shape: {gnds.shape}")
    sorted_result = np.sort(gnds, axis=1, order=["distance"])
    print(f"sorted shape: {sorted_result.shape}, sorted[0] shape: {sorted_result[0].shape}")
    print(sorted_result[0])

    top_sorted_result = sorted_result[:, :top_k]
    print(f"topk sorted shape: {top_sorted_result.shape}, topk[0] shape: {top_sorted_result[0].shape}")
    print(top_sorted_result[0])

    idx = np.empty(top_sorted_result.shape, dtype="i4")
    dis = np.empty(top_sorted_result.shape, dtype="f4")
    for index, value in np.ndenumerate(top_sorted_result):
        idx[index] = value[0]
        dis[index] = value[1]
    print(idx.shape)
    print(idx)
    print(dis.shape)
    print(dis)
    ivecs_write('/test/raw_data/laion5b_parquet/gnd/idx_5000M.ivecs', idx)
    fvecs_write('/test/raw_data/laion5b_parquet/gnd/dis_5000M.fvecs', dis)


if __name__ == '__main__':
    url = "/test/raw_data/laion5b_parquet/distance_top1000_faiss"
    merge_gnd(url, 1000)
