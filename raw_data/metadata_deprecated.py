from utils.utils_log import log
import pandas as pd


def merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_parquet_path, pk_index):
    file_name = metadata_path.split("/")[-1]
    log.info(f"start process file  {file_name}")
    # read metadata
    reserve_columns = ['caption', 'NSFW', 'similarity', 'width', 'height', 'original_width', 'original_height', 'md5']
    log.info("read metadata parquet file")
    metadata_df = pd.read_parquet(metadata_path, engine='pyarrow', columns=reserve_columns)
    df_len = len(metadata_df)
    log.info(f"metadata num: {df_len}")
    log.info("metadata columns:")
    log.info([column for column in metadata_df])

    # add pk column
    log.info("start add pk column")
    int_values = pd.Series(data=[i for i in range(pk_index, pk_index + df_len)])
    metadata_df["pk"] = int_values
    log.info("finish add pk column, and columns is:")
    log.info([column for column in metadata_df])

    # add vector column
    log.info("start add vector column")
    vec = np.load(vector_path)
    vec = vec.astype(dtype=np.float32)
    log.info(vec.shape)
    log.info(vec.dtype)
    metadata_df["float32_vector"] = vec.tolist()
    del vec
    log.info("finish add vector column, and columns is:")
    log.info([column for column in metadata_df])

    # save dataframe to parquet file
    log.info(f"start save parquet {file_name}")
    metadata_df.to_parquet(dest_parquet_path)
    log.info(f"finish process file  {file_name}")
    del metadata_df
    return df_len


def gen_url(data_type: str, source_index=0):
    base_url = '/data/laion2B-multi/'
    dest_url = '/test/milvus/raw_data/laion5b_parquet'

    if data_type == "metadata":
        url = f'{base_url}laion2B-multi-metadata/metadata_{source_index:04d}.parquet'
    elif data_type == "vector":
        url = f'{base_url}img_emb/img_emb_{source_index:04d}.npy'
    elif data_type == "dest":
        url = f'{dest_url}/binary_768d_{source_index:05d}.parquet'
    return url


if __name__ == '__main__':
    # 1270 and before are from laion1b-nolang
    pk_index = 3952540762  # next pk index: last_index + 1
    exist_file_index = 3567
    for i in range(600, 700):
        metadata_path = gen_url(data_type="metadata", source_index=i)
        vector_path = gen_url(data_type="vector", source_index=i)
        dest_path = gen_url(data_type="dest", source_index=exist_file_index + i)
        single_df_len = merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_path, pk_index)
        pk_index += single_df_len
        log.info(f"processed entities: {pk_index}")
    log.info(f"total processed entities: {pk_index}")
