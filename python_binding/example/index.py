import yannsa

index = yannsa.EuclideanIndex()
index.init_data("sift_base.fvecs")
index.build(k=20, join_k=40, lambda_value=0.2, refine_iter_num=20)
index.save_index("python_index")

