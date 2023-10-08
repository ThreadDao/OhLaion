from pymilvus import connections, Collection
import time
import numpy as np
from utils.utils_log import log


host = 'envoy.qa-milvus'
port = 8443
limit = 10000000
batch_size = 10000

# host = '10.104.20.131'
# port = 19530
# limit = 100
# batch_size = 10

# connect and init collection
connections.connect(host=host, port=port)
cname = "fouram_aTOPOXCm"
c = Collection(name=cname)

# search iterator params
test_npy = np.load("/test/milvus/raw_data/laion5b_parquet/query.npy")
query_vec = test_npy[:1]
del test_npy

search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 20000},
}

# query iterator
# query_iterator = c.query_iterator(batch_size=batch_size, limit=limit, expr=expr)
search_iterator = c.search_iterator(data=query_vec,
                                    anns_field="float_vector",
                                    param=search_params, batch_size=batch_size, limit=limit)

all_cost = []
entities = 0
while True:
    # turn to the next page
    _start = time.time()
    res = search_iterator.next()
    cost = time.time() - _start
    cost_ms = round(cost*1000, 3)
    if len(res) == 0:
        log.info("query iteration finished, close")
        search_iterator.close()
        break
    log.info(f"query iteration {len(res)} results")
    assert len(res) == batch_size
    log.info(f"ids len: {len(res.ids())}")
    log.info(f"distance len: {len(res.distances())}")
    entities += len(res)

    log.info(f'iterator cost: {cost_ms}ms')
    all_cost.append(cost_ms)

log.info(f"entities: {entities}")
assert entities == limit
log.info(f"all cost len: {len(all_cost)}")
arr_cost = np.array(all_cost)
log.info(arr_cost)
np.savetxt('/data/search_iterator_cost_ms.txt', arr_cost)
