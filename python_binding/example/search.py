from __future__ import print_function
import yannsa
import numpy as np

index = yannsa.EuclideanIndex()
index.init_data('sift_base.fvecs')
index.load_index("python_index")

query = np.array(range(128), dtype=np.float32)
ret = index.search_knn(query, 10, 100)
print(ret)

index.add_point("new", query, 100)

ret = index.search_knn(query, 10, 100)
print(ret)

