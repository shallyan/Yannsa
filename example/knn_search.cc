#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/binary_encoder_imp.h"
#include "yannsa/wrapper/index_helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

int LoadEmbeddingData(const string& file_path,
                      DatasetPtr<float>& dataset_ptr) { 

  ifstream in_file(file_path.c_str());
  if (!in_file.is_open()) {
    cout << "open file error" << endl;
    exit(-1);
  }

  string buff;

  //read word and dim number
  int vec_num = 0, vec_dim = 0;
  getline(in_file, buff);
  stringstream count_stream(buff);
  count_stream >> vec_num >> vec_dim; 
  cout << "vec num: " << vec_num << "\t" 
       << "vec dim: " << vec_dim << endl;

  while (getline(in_file, buff)) {
    stringstream one_word_vec_stream(buff);

    //read word firstly
    string word;
    one_word_vec_stream >> word;

    //then read value
    PointVector<float> point(vec_dim);
    double value = 0.0;
    int dim_count = 0;
    while (one_word_vec_stream >> value) { 
      point[dim_count++] = value;
    }
    point.normalize();

    //check dim num
    assert(dim_count == vec_dim);

    dataset_ptr->insert(word, point);
  }

  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;

  return vec_dim;
}
int main(int argc, char** argv) {
  if (argc < 17) {
    cout << "binary -data_path -graph_path -hash_length -point_neighbor_num "
         << "-max_point_neighbor_num -min_bucket_size -max_bucket_size "
         << "-local_refine_iter_num -global_refine_iter_num -search_point_neighbor_num "
         << "-search_start_point_num -query_path -k -search_k -search_start_point_num -search_result_path"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string graph_path = argv[2];
  int hash_length = atoi(argv[3]);

  util::GraphIndexParameter param;
  param.point_neighbor_num = atoi(argv[4]);
  param.max_point_neighbor_num = atoi(argv[5]);
  param.min_bucket_size = atoi(argv[6]);
  param.max_bucket_size = atoi(argv[7]);
  param.local_refine_iter_num = atoi(argv[8]);
  param.global_refine_iter_num = atoi(argv[9]);
  param.search_point_neighbor_num = atoi(argv[10]);
  param.search_start_point_num = atoi(argv[11]);

  string query_path = argv[12];

  util::GraphSearchParameter search_param;
  search_param.k = atoi(argv[13]);
  search_param.search_k = atoi(argv[14]);
  search_param.search_start_point_num = atoi(argv[15]);

  string search_result_path = argv[16];
   
  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadEmbeddingData(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);
  
  DotGraphIndexPtr<float> graph_index_ptr(new DotGraphIndex<float>(dataset_ptr));
  BinaryEncoderPtr<PointVector<float> > 
      binary_encoder_ptr(new RandomBinaryEncoder<PointVector<float>, float>(point_dim, hash_length));

  graph_index_ptr->Build(param, binary_encoder_ptr);
  graph_index_ptr->Save(graph_path);

  vector<vector<string> > search_result(query_ptr->size());
  util::Log("before search");
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < query_ptr->size(); i++) {
    graph_index_ptr->SearchKnn((*query_ptr)[i], search_param, search_result[i]);
  }
  util::Log("end search");

  ofstream resulte_file(search_result_path);

  for (int i = 0; i < query_ptr->size(); i++) {
    resulte_file << query_ptr->GetKeyById(i) << " ";
    for (auto nn : search_result[i]) {
      resulte_file << nn << " ";
    }
    resulte_file << endl;
  }
  resulte_file.close();

  return 0;
}
