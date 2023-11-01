import os
import logging
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

next_pk_5b = 1229565156


class TestLog:
    def __init__(self, logger, log_file):
        self.logger = logger
        self.log_file = log_file

        self.log = logging.getLogger(self.logger)
        self.log.setLevel(logging.DEBUG)

        try:
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s]: "
                                          "%(message)s (%(filename)s:%(lineno)s)")

            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.log.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

        except Exception as e:
            print("Can not use %s to log. error : %s" % (log_file, str(e)))


log = TestLog('laion', log_file='/tmp/laion.log').log


def check_file_exist(file_dir):
    if not os.path.isfile(file_dir):
        msg = "[check_file_exist] File not exist:{}".format(file_dir)
        log.warning(msg)
        return False
    return True


def load_process_progress(_dest_data_dir):
    """
    load the next pk value and next file index from /${_dest_data_dir}/next_file.index
    :param next_pk_5b_index: (str) exist pk_5b_index
    :param _dest_data_dir: (str)
    :return next_pk_index, next_file_index
    """
    next_file_index_path = f"{_dest_data_dir}/next_file.index"
    # load next progress
    if not check_file_exist(next_file_index_path):
        next_pk_index = 0
        next_pk_5b_index = 1229565156
        next_file_index = 3567
    else:
        with open(next_file_index_path, 'r') as f:
            progress_str_array = f.readline().split('\t')
            next_pk_index = int(progress_str_array[0])
            next_file_index = int(progress_str_array[1])
            next_pk_5b_index = int(progress_str_array[2])
    return next_pk_index, next_file_index, next_pk_5b_index


def record_process_progress(_dest_data_dir, next_pk_index, next_file_index, next_pk_5b_index):
    """
    save next pk value and next file index to the dest_dir file: next_file.index.
    So that next time we can continue processing from the place where the previous processing was successful.
    :param _dest_data_dir: (str) where to save the processed file index
    :param next_pk_index: (int) The number of the next pk
    :param next_file_index: (int) The index of the next file to be processed
    :param next_pk_5b_index:  (str) The number of the next pk_5b
    :return: None
    """
    next_file_index_path = f"{_dest_data_dir}/next_file.index"
    with open(next_file_index_path, 'w') as f:
        f.write(f"{next_pk_index}\t{next_file_index}\t{next_pk_5b_index}")


def process_data(source_path, dest_path, pk_start, pk_5b_start):
    log.info(f"start read file {source_path}")
    df = pd.read_parquet(source_path)
    log.info(f"finish read file {source_path}")
    vector_count = df.shape[0]

    # df pk columns rename
    new_pk = pd.Series(data=[i for i in range(pk_start, pk_start + vector_count)])
    df['pk'] = new_pk
    log.info(f"pk from {new_pk[new_pk.first_valid_index()]} to {new_pk[new_pk.last_valid_index()]}")
    pk_5b = pd.Series(data=[i for i in range(pk_5b_start, pk_5b_start + vector_count)])
    df['pk_5b'] = pk_5b
    log.info(f"pk_5b from {pk_5b[pk_5b.first_valid_index()]} to {pk_5b[pk_5b.last_valid_index()]}")

    schema = pa.schema([
        pa.field('pk', pa.int64()),
        pa.field('pk_5b', pa.int64()),
        pa.field('caption', pa.string()),
        pa.field('NSFW', pa.string()),
        pa.field('similarity', pa.float64()),
        pa.field('width', pa.int64()),
        pa.field('height', pa.int64()),
        pa.field('original_width', pa.int64()),
        pa.field('original_height', pa.int64()),
        pa.field('md5', pa.string()),
        pa.field('float32_vector', pa.list_(pa.float32()))])

    log.info(f"start write file to /data/project/test.parquet")
    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, dest_path)
    log.info(f"write parquet done")
    return pk_start + vector_count, pk_5b_start + vector_count


def loop_process_data(source_dir, dest_dir):
    next_pk_index, next_file_index, next_pk_5b_index = load_process_progress(dest_dir)
    log.info(f"load process progress. next_pk_index = {next_pk_index}, next_file_index = {next_file_index}, "
             f"next_pk_5b_index = {next_pk_5b_index}")
    for i in range(3567, 5818):
        if i < next_file_index:
            log.info(
                f"skip processed file. cur_emb_file_name = {i:05d}.parquet")
            continue
        vector_file_path = f"{source_dir}/binary_768d_{i:05d}.parquet"
        dest_data_path = f"{dest_dir}/binary_768d_{i - 3567:05d}.parquet"
        log.info(
            f"process data. cur_emb_file_name_index = {i}, cur_emb_file_name = {i:05d}.parquet, "
            f"vector_file_path = {vector_file_path}, dest_data_path = {dest_data_path}")

        if check_file_exist(vector_file_path):
            next_pk_index, next_pk_5b_index = process_data(vector_file_path, dest_data_path, next_pk_index,
                                                           next_pk_5b_index)
            record_process_progress(dest_dir, next_pk_index, i + 1, next_pk_5b_index)
            log.info(
                f"record process progress. next_pk_index = {next_pk_index}, next_file_index = {i + 1}, next_pk_5b_index = {next_pk_5b_index}")
        else:
            raise Exception(f"{vector_file_path} file not exist, please check!")


if __name__ == '__main__':
    source_dir = "/test/xxx/raw_data/laion5b_parquet"
    dest_dir = "/test/xxx/raw_data/laion5b_parquet/laion2B_multi"
    loop_process_data(source_dir, dest_dir)
