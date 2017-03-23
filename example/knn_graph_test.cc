#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/test_helper.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::test;

int main(int argc, char** argv) {
  string data_path = argv[1];
  string graph_path = argv[2];

  util::GraphIndexParameter param;
  param.point_neighbor_num = atoi(argv[3]);
  param.max_point_neighbor_num = atoi(argv[4]);
  param.bucket_neighbor_num = atoi(argv[5]);
  param.min_bucket_size = atoi(argv[6]);
  param.max_bucket_size = atoi(argv[7]);
  param.refine_iter_num = atoi(argv[8]);
  param.search_point_neighbor_num = atoi(argv[9]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadBinaryData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));
  BaseEncoderPtr<PointVector<float> > 
      binary_encoder_ptr(new BinaryEncoder<PointVector<float>, float>(point_dim, param.bucket_neighbor_num));

  graph_index_ptr->Build(param, binary_encoder_ptr);

  return 0;
}
