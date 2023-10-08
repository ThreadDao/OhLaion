from pymilvus import connections, Collection
import time
import numpy as np
from utils.utils_log import log

host = 'envoy.qa-milvus'
port = 8443
expr = "id >= 0"
limit = 100000000
batch_size = 10000

# connect and init collection
connections.connect(host=host, port=port)
cname = "fouram_aTOPOXCm"
c = Collection(name=cname)

# query iterator
query_iterator = c.query_iterator(batch_size=batch_size, limit=limit, expr=expr)

all_cost = []
entities = 0
while True:
    # turn to the next page
    _start = time.time()
    res = query_iterator.next()
    cost = time.time() - _start
    cost_ms = round(cost*1000, 3)
    log.info(f'iterator cost: {cost_ms}ms')
    all_cost.append(cost_ms)
    if len(res) == 0:
        log.info("query iteration finished, close")
        query_iterator.close()
        break
    log.info(f"query iteration {len(res)} results")
    entities += len(res)

log.info(f"entities: {entities}")
assert entities == limit
log.info(f"all cost len: {len(all_cost)}")
arr_cost = np.array(all_cost)
np.savetxt('/data/query_iterator_cost_ms_2.txt', arr_cost)