#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

int main(int argc, char** argv) {
  if (argc != 7) {
    cout << "binary -data_path -index_path -point_neighbor_num "
         << "-join_point_neighbor_num -max_point_neighbor_num "
         << "-refine_iter_num"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];

  util::GraphIndexParameter param;
  param.point_neighbor_num = atoi(argv[3]);
  param.join_point_neighbor_num = atoi(argv[4]);
  param.max_point_neighbor_num = atoi(argv[5]);
  param.refine_iter_num = atoi(argv[6]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->Build(param);
  graph_index_ptr->SaveKnnGraph(index_path + "_knn_graph");
  graph_index_ptr->SaveIndex(index_path);

  return 0;
}
