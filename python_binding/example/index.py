import yannsa

# EuclideanIndex or CosineIndex 
index = yannsa.EuclideanIndex()

# Init data could use init_data_fvecs or init_data_embedding
# The format of fvec is the same with http://corpus-texmex.irisa.fr 
# The format of embedding is the same with word2vec 

# Data stored are key-vector pairs, and fvecs does not provided the key, so that int-id starting from 0 is used as the key.
index.init_data_fvecs("sift_base.fvecs")

index.build(k=20, join_k=40, lambda_value=0.2, refine_iter_num=20)
index.save_index("python_index")

