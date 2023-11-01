from utils.utils_log import log
import random
import pandas as pd
import numpy as np

if __name__ == '__main__':
    base_url = '/test/raw_data/laion5b_parquet'
    nq = 10000
    sample_file = random.sample(range(100), 5)
    log.info(f"sample file {sample_file}")
    rows_per = int(nq / len(sample_file))
    all_vec_dfs = []
    # merge df
    for i in sample_file:
        path = f'{base_url}/binary_768d_{i:05d}.parquet'
        log.info(f"start read file {path}")
        df = pd.read_parquet(path, engine='pyarrow')
        log.info(f"filter out vector column")
        vec_df = df.filter(['float32_vector'], axis=1)
        print(vec_df.shape)
        all_vec_dfs.append(vec_df.sample(rows_per))

    log.info(f"start merge all df")
    last_df = pd.concat(all_vec_dfs, ignore_index=True, sort=False)
    print(last_df.shape)
    print(last_df[:5])

    #
    log.info(f"start save query.npy")
    np.save('/tmp/query.npy', np.vstack(last_df['float32_vector'].to_numpy()))
