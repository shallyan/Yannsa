# Yannsa
Yet another nearest neighbor search algorithms, now including [k-diverse nearest neighbor (k-DNN) graph](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16132/16525).

## k-DNN Graph
Approximate nearest neighbor search is a fundamental problem and has been studied for a few decades. Recently graph-based indexing methods have demonstrated their great efficiency, whose main idea is to construct neighborhood graph offline and perform a greedy search starting from some sampled points of the graph online. 

Most existing graph-based methods focus on building either the precise k-NN graph or the direction-diverse graph. The precise k-NN graph has good exploitation ability, i.e., neighbors are exactly those nearest points, but the searching on the graph might be easily trapped in local optimums due to the lack of the exploration ability. On the contrary, the direction-diverse graph has good exploration ability, i.e., each point connects multidirectional neighbors so that different directions can be explored when traversing on the graph, but it might not be able to exploit the neighbors very well by focusing on exploration too much.

Each point of k-DNN graph is connected to a set of neighbors that are close in distance while diverse in direction. In this way, we can balance the precision and diversity of the neighborhood graph to keep good exploitation and exploration abilities simultaneously. We take a novel view of the graph construction process as search result diversification in IR, which considers each point as the query and the neighbor candidates as documents, and re-ranks the neighbors based on an adaption of the maximal marginal relevance criterion. 

## Features
- Fast and fully parallel index construction.
- Fast approximate nearest neighbor search.
- New data insertion .

## Future Work
Now k-DNN Graph search from randomly sampled points, the efficiency can be significantly improved via providing a better start point with other data structure.

## Parameters
### Index
- k: neighbor number of each point, usually 20 is good enough.
- join_k: neighbor selection range, usually 4 times of k, i.e., 80.
- refine_iter_num: iteration number, usually 20.
- lambda: weight between precision and diversity, usually 0.15 ~ 0.20.

### Search
- K: return approximate K nearest neighbors.
- search_K: search range. The bigger, the more precise, the more cost. 
## C++ Example
- Download ANN_SIFT1M dataset from http://corpus-texmex.irisa.fr
- Compile (The compiler must support C++11)
  - make index
  - make search
- Indexing
  - binary -data_path -index_save_path -k -join_k -refine_iter_num -lambda
  - ./index sift_base.fvecs sift_index 20 40 15 0.2
- Search
  - binary -data_path -index_path -query_path -ground_truth_path
  - ./search sift_base.fvecs sift_index sift_query.fvecs sift_groundtruth.ivecs 

## Python Package 
### Install (python2.7+, python3)
- cd python_binding
- python setup.py install

### Example 
- cd example
- python index.py
- python search.py

