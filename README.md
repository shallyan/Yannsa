# Yannsa
Yet another nearest neighbor search algorithms, now including k-diverse nearest neighbor (k-DNN) graph.

## k-DNN Graph
Approximate nearest neighbor search is a fundamental problem and has been studied for a few decades. Recently graph-based indexing methods have demonstrated their great efficiency, whose main idea is to construct neighborhood graph offline and perform a greedy search starting from some sampled points of the graph online. 

Most existing graph-based methods focus on building either the precise k-NN graph or the direction-diverse graph. The precise k-NN graph has good exploitation ability, i.e., neighbors are exactly those nearest points, but the searching on the graph might be easily trapped in local optimums due to the lack of the exploration ability. On the contrary, the direction-diverse graph has good exploration ability, i.e., each point connects multidirectional neighbors so that different directions can be explored when traversing on the graph, but it might not be able to exploit the neighbors very well by focusing on exploration too much.

Each point of k-DNN graph is connected to a set of neighbors that are close in distance while diverse in direction. In this way, we can balance the precision and diversity of the neighborhood graph to keep good exploitation and exploration abilities simultaneously. We take a novel view of the graph construction process as search result diversification in IR, which considers each point as the query and the neighbor candidates as documents, and re-ranks the neighbors based on an adaption of the maximal marginal relevance criterion. 

## Features
- Fast and fully parallel index construction
- Fast approximate nearest neighbor search

## C++ Example
- Download ANN_SIFT1M dataset from http://corpus-texmex.irisa.fr
- Compile
  - make index
  - make search
- Indexing
  - binary -data_path -index_save_path -k -join_k -refine_iter_num -lambda
  - ./index sift_base.fvecs sift_index 20 40 15 0.2
- Search
  - binary -data_path -index_path -query_path -ground_truth_path
  - ./search sift_base.fvecs sift_index sift_query.fvecs sift_groundtruth.ivecs 

## Python Package 
### Install
- cd python_binding
- python setup.py install

### Manually on Mac
- g++ -O3 -march=native -fopenmp -shared -std=c++11 -undefined dynamic_lookup `python -m pybind11 --includes` binding.cpp -I ../src/include/ -I ../third_party/ -o yannsa.so
- cp yannsa.so example

### Example 
- cd example
- python index.py
- python search.py
