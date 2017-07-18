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
  if (argc != 6) {
    cout << "binary -data_path -index_path -extend_index_path "
         << "-lambda -point_neighbor_num"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string extend_index_path = argv[3];
  double lambda = atof(argv[4]);
  int point_neighbor_num = atoi(argv[5]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->LoadIndex(index_path);
  graph_index_ptr->Prune(lambda, point_neighbor_num);
  graph_index_ptr->Reverse();
  graph_index_ptr->SaveIndex(extend_index_path);

  return 0;
}
