#include "yannsa/util/parameter.h"
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
    cout << "binary -data_path -index_save_path -k "
         << "-join_k -refine_iter_num -lambda "
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_save_path = argv[2];

  util::GraphIndexParameter index_param;
  index_param.k = atoi(argv[3]);
  index_param.join_k = atoi(argv[4]);
  index_param.refine_iter_num = atoi(argv[5]);
  index_param.lambda = atof(argv[6]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadBinaryData<float>(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->Build(index_param);
  graph_index_ptr->SaveIndex(index_save_path);

  return 0;
}
