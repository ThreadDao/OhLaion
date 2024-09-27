import multiprocessing
import logging
import pyarrow.parquet as pq
import pandas as pd
import json
import os
import gc
import sys
import boto3

chunk_size = 40000  # 每个 JSON 文件的行数
pk = "pk"
float32_vector = "float32_vector"
input_dir = '/your/input/parquet/path/'
output_dir = '/your/output/json/path/'


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


log = TestLog('laion', log_file='/tmp/apple.log').log


def loop_gen_file_name(start_i, end_i, index_j):
    for i in range(start_i, end_i):
        for j in range(index_j):
            yield f"{input_dir}/msmarco_passage_{i:02d}-{j}.parquet"


def check_file_exist(file_dir):
    if not os.path.isfile(file_dir):
        msg = "[check_file_exist] File not exist:{}".format(file_dir)
        log.info(msg)
        return False
    return True


s3 = boto3.client('s3')


def upload_to_s3(local_file, s3_bucket, s3_path):
    try:
        s3.upload_file(local_file, s3_bucket, s3_path)
        log.info(f"Uploaded {local_file} to s3://{s3_bucket}/{s3_path}")
        return True
    except Exception as e:
        log.error(f"Failed to upload {local_file} to S3: {str(e)}")
        return False


def read_parquet_chunk(file_name, batch_size):
    parquet_file = pq.ParquetFile(file_name)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()
        yield batch_df


class PreDf:
    def __init__(self, start_i, end_i, index_j):
        self.vectors = []  # 存储所有 vector
        self.pks = []  # 存储所有 pk
        self._loop_file = loop_gen_file_name(start_i, end_i, index_j)
        self._current_chunk_gen = None  # 当前的 chunk 生成器
        self._filename = None  # 当前文件名

    def loop_data(self, ni=500):
        # 当 pks 为空时，从当前或下一个文件中读取数据
        while len(self.pks) < ni:
            # 如果当前没有 chunk 生成器，或者生成器数据已耗尽，则获取下一个文件
            if self._current_chunk_gen is None:
                self._filename = next(self._loop_file, None)
                if self._filename is None:  # 文件名耗尽
                    break
                # 初始化新的文件读取生成器
                self._current_chunk_gen = read_parquet_chunk(self._filename, ni)

            # 尝试从当前 chunk 生成器读取数据
            try:
                _df = next(self._current_chunk_gen)
                self.pks.extend(_df['pk'].to_list())  # 假设 'pk' 是 DataFrame 中的列名
                self.vectors.extend(_df['float32_vector'].to_list())  # 假设 'float32_vector' 是 DataFrame 中的列名
                del _df
                gc.collect()
            except StopIteration:
                # 当前生成器耗尽，重置生成器为 None，继续下一个文件
                self._current_chunk_gen = None

        # 如果仍然没有数据，则表示所有文件读取完毕
        if len(self.pks) == 0:
            return None, None

        # 取出所需数量的数据
        _pk = self.pks[:ni]
        _v = self.vectors[:ni]

        # 更新剩余的数据
        self.pks = self.pks[ni:]
        self.vectors = self.vectors[ni:]

        return _pk, _v


def parquet_to_json(start_i, end_i, index_j):
    dfObj = PreDf(start_i, end_i, index_j)
    _index = 0
    while True:
        chunk_pks, chunk_vectors = dfObj.loop_data(chunk_size)
        if chunk_pks is None:
            break
        log.info(f"Read file {start_i:02d} to {end_i:02d}, chunk={_index}")
        # 按指定格式转换为 JSON
        json_rows = []
        for i in range(len(chunk_pks)):
            json_rows.append({
                pk: chunk_pks[i],
                float32_vector: [round(float(x), 8) for x in chunk_vectors[i].tolist()],
            })
        del chunk_pks, chunk_vectors
        gc.collect()
        json_output = '{"rows":[' + ',\n'.join([json.dumps(row, separators=(',', ':')) for row in json_rows]) + ']}\n'

        # # 生成 JSON 文件名
        output_file = f"{output_dir}/msmarco_passage_{start_i:02d}_{end_i:02d}_chunk_{_index}.json"

        # # 重新写入格式化后的内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_output)

        log.info(f"Written {start_i:02d}-{end_i:02d}, chunk={_index} to {output_file}")

        # cp
        upload_to_s3(output_file, 'file-transfering-bucket',
                     f'milvus/raw_data/msmarco_v2_138M_json/{os.path.basename(output_file)}')
        os.remove(output_file)
        log.info(f"cp s3 done and remove {output_file}")
        _index += 1


index_i = [
    [0, 2],  # [start, end]
    [2, 4],  # [start, end]
    [4, 6],  # [start, end]
    [6, 8],  # [start, end]
    [8, 10]  # [start, end]
]
pool = multiprocessing.Pool(len(index_i))
normal_index_j = 4
pool.starmap(parquet_to_json, [(index_para[0], index_para[1], normal_index_j) for index_para in index_i])
