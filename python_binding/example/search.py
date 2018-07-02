from __future__ import print_function
import yannsa
import numpy as np

index = yannsa.EuclideanIndex()
# in fvec format, integer starting from 0 is used as the key for vectors
index.init_data_fvecs('sift_base.fvecs')
index.load_index("python_index")

query = np.array(range(128), dtype=np.float32)
# return values are the keys of 10 nearest neighbors
ret = index.search_knn(query, 10, 100)
print(ret)

# parameters are key, point, search_K
# search_K: the bigger, the more precise, the more cost
index.add_point("new", query, 100)

ret = index.search_knn(query, 10, 100)
print(ret)

