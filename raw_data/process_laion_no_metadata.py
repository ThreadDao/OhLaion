import argparse
import sys
import logging
import os
import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq
from pyquery import PyQuery
import pyarrow as pa


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


log = TestLog('data', log_file='/data/logs/process.log').log


def get_the_eye_vector_file_names(data_url):
    """
    Get all the file names we should download from the-eye.eu
    :return: all file anmes
    """
    log.info("get all the-eye file names")
    resp = requests.get(url=data_url)
    img_emb_a_tags = PyQuery(resp.content).find('a[href^=img_emb]')
    all_file_names = []
    for img_emb_a in img_emb_a_tags:
        all_file_names.append(PyQuery(img_emb_a).attr('href'))
    log.info(f"len all files: {len(all_file_names)}")
    return all_file_names


def check_file_exist(file_dir):
    if not os.path.isfile(file_dir):
        msg = "[check_file_exist] File not exist:{}".format(file_dir)
        log.warning(msg)
        return False
    return True


def process_data(vector_file_path, pk_start, dest_data_path):
    """
    process one file data
    1. read downloaded vector npy file and astype to float32
    2. Auto-increment to generate generate pk values from the pk_start
    3. generate dataframe includes pk and vector
    4. save data to parquet file
    4. update next pk
    :param vector_file_path: vector file path (str)
    :param pk_start: pk start (int)
    :param dest_data_path: dest data path to save parquet (str)
    :return: next pk value
    """
    try:
        # 1. read npy vector data
        vector = np.load(vector_file_path)
        vector = vector.astype(dtype=np.float32)
        vector_count = vector.shape[0]
        log.info(f"vector shape: {vector.shape}, vector dtype is {vector.dtype}")

        # 2. generate pks
        pks = pd.Series(data=[i for i in range(pk_start, pk_start + vector_count)])
        log.info(f"pk values generate from {pk_start} to {pk_start + vector_count - 1}")

        # 3. generate dataframe
        df = pd.DataFrame({
            "pk": pks,
            "float32_vector": vector.tolist()
        })
        log.info(f"generate dataframe done")
        log.info(f"All columns are: {[c for c in df]}")

        # 4. save to parquet file
        schema = pa.schema([
            pa.field('pk', pa.int64()),
            pa.field('float32_vector', pa.list_(pa.float32()))])
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, dest_data_path)
        log.info(f"finish process file {dest_data_path}")

        # return next pk
        return pk_start + vector_count
    except Exception as e:
        log.error(str(e))
        raise Exception(e)


def load_process_progress(_dest_data_dir):
    """
    load the next pk value and next file index from /${_dest_data_dir}/next_file.index
    :param _dest_data_dir: (str)
    :return next_pk_index, next_file_index
    """
    next_file_index_path = f"{_dest_data_dir}/next_file.index"
    if not check_file_exist(next_file_index_path):
        next_pk_index = 0
        next_file_index = 0
    else:
        with open(next_file_index_path, 'r') as f:
            progress_str_array = f.readline().split('\t')
            next_pk_index = int(progress_str_array[0])
            next_file_index = int(progress_str_array[1])
    return next_pk_index, next_file_index


def record_process_progress(_dest_data_dir, next_pk_index, next_file_index):
    """
    save next pk value and next file index to the dest_dir file: next_file.index.
    So that next time we can continue processing from the place where the previous processing was successful.
    :param _dest_data_dir: (str) where to save the processed file index
    :param next_pk_index: (int) The number of the next pk
    :param next_file_index: (int) The index of the next file to be processed
    :return: None
    """
    next_file_index_path = f"{_dest_data_dir}/next_file.index"
    with open(next_file_index_path, 'w') as f:
        f.write(f"{next_pk_index}\t{next_file_index}")


def loop_process_data(data_url, _source_data_dir, _dest_data_dir):
    """
    Process all file data in a loop. If it is not downloaded, it will fail and throw an exception.
    If it is successfully processed, it will be skipped.
    :param _source_data_dir: (str) where your image vector dir
    :param _dest_data_dir: (str) where to save the processed data *.parquet
    :return None
    """
    # read next process file_index and pk_index
    next_pk_index, next_file_index = load_process_progress(_dest_data_dir)
    log.info(f"load process progress. next_pk_index = {next_pk_index}, next_file_index = {next_file_index}")

    # all the-eye file list
    all_file_names = get_the_eye_vector_file_names(data_url)
    log.info(f"get the eye vector file names. len = {len(all_file_names)},\nall_file_names = {all_file_names}")
    for cur_emb_file_name_index, cur_emb_file_name in enumerate(all_file_names):
        if cur_emb_file_name_index < next_file_index:
            log.info(
                f"skip processed file. cur_emb_file_name_index = {cur_emb_file_name_index}, cur_emb_file_name = {cur_emb_file_name}")
            continue

        vector_file_path = f"{_source_data_dir}/{cur_emb_file_name}"
        dest_data_path = f"{_dest_data_dir}/binary_768d_{cur_emb_file_name_index:04d}.parquet"
        log.info(
            f"process data. cur_emb_file_name_index = {cur_emb_file_name_index}, cur_emb_file_name = {cur_emb_file_name}, "
            f"vector_file_path = {vector_file_path}, dest_data_path = {dest_data_path}")

        if check_file_exist(vector_file_path):
            next_pk_index = process_data(vector_file_path, next_pk_index, dest_data_path)
            record_process_progress(_dest_data_dir, next_pk_index, cur_emb_file_name_index + 1)
            log.info(
                f"record process progress. next_pk_index = {next_pk_index}, next_file_index = {cur_emb_file_name_index + 1}")
        else:
            raise Exception(f"{vector_file_path} file not exist, please check!")


if __name__ == '__main__':
    """
    example: python3.8 main.py --source_dir=/your/vector/data/dir --dest_dir=/your/dest/dir
    or 
    python3.8 main.py -s /your/vector/data/dir -d /your/dest/dir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir', type=str, help='The dir of source vector data')
    parser.add_argument('-d', '--dest_dir', type=str, help='The dir to save the data after processing')

    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir

    if not source_dir or not dest_dir:
        log.error("Please pass source dir and dest dir, please see python3 $your_script -h")
        sys.exit(-1)

    data_url = "https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/"
    loop_process_data(data_url, source_dir, dest_dir)
