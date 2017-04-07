#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "test_helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <cassert>
#include <unordered_map>
#include <algorithm>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::test;

void Normalize(PointVector<float>& vec) {
  float squ_sum = 0.0;
  for (size_t i = 0; i < vec.size(); ++i) {
    squ_sum += vec[i] * vec[i];
  }

  float mode = sqrt(squ_sum);
  if (mode > 1e-10) {
    for (size_t i = 0; i < vec.size(); ++i) {
      vec[i] = vec[i] / mode;
    }
  }
}

int LoadBinaryData(string file_path, DatasetPtr<float>& dataset_ptr) { 
  ifstream in(file_path.c_str(), ios::binary);

  if (!in.is_open()) {
    cout << "open file error" << endl;
    exit(-1);
  }
  int vec_dim=0, vec_num=0;
  in.read((char*)&vec_dim,4);
  in.seekg(0, ios::end);
  vec_num = (size_t)in.tellg() / (vec_dim+1) / 4;
  cout << "vec num: " << vec_num << "\t" 
       << "vec dim: " << vec_dim << endl;

  in.seekg(0, ios::beg);

  size_t ten_percent_num = vec_num / 10;

  PointVector<float> center_point(vec_dim, 0.0);
  for(size_t i = 0; i < vec_num; i++){
    in.seekg(4, ios::cur);
    PointVector<float> point(vec_dim);
    for (int j = 0; j < vec_dim; j++) {
      float value;
      in.read((char*)(&value), 4);
      point[j] = value;
    }

    for (size_t i = 0; i < point.size(); ++i) {
      center_point[i] += point[i];
    }

    stringstream key_str;
    key_str << i;
    string key;
    key_str >> key;
    dataset_ptr->insert(key, point);

    if (i % ten_percent_num == 0) {
      cout << "read " << i << " points" << endl;
    }
  }
  in.close();

  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;

  for (size_t i = 0; i < center_point.size(); ++i) {
    center_point[i] /= dataset_ptr->size();
  }

  for (size_t i = 0; i < dataset_ptr->size(); ++i) {
    PointVector<float>& point = (*dataset_ptr)[i];
    for (size_t j = 0; j < center_point.size(); ++j) {
      point[j] -= center_point[j];
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 12) {
    cout << "binary -data_path -graph_path -hash_length -point_neighbor_num "
         << "-max_point_neighbor_num -bucket_neighbor_num -min_bucket_size "
         << "-max_bucket_size -local_refine_iter_num -global_refine_iter_num -search_point_neighbor_num "
         << "-search_start_point_num"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string graph_path = argv[2];

  int hash_length = atoi(argv[3]);

  util::GraphIndexParameter param;
  param.point_neighbor_num = atoi(argv[4]);
  param.max_point_neighbor_num = atoi(argv[5]);
  param.bucket_neighbor_num = atoi(argv[6]);
  param.min_bucket_size = atoi(argv[7]);
  param.max_bucket_size = atoi(argv[8]);
  param.local_refine_iter_num = atoi(argv[9]);
  param.global_refine_iter_num = atoi(argv[10]);
  param.search_point_neighbor_num = atoi(argv[11]);
  param.search_start_point_num = atoi(argv[12]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadBinaryData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));
  BaseEncoderPtr<PointVector<float> > 
      binary_encoder_ptr(new BinaryEncoder<PointVector<float>, float>(point_dim, hash_length, dataset_ptr));

  graph_index_ptr->Build(param, binary_encoder_ptr);
  graph_index_ptr->Save(graph_path);

  return 0;
}
