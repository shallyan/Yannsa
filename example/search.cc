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
    //point.normalize();

    //check dim num
    assert(dim_count == vec_dim);

    dataset_ptr->insert(word, point);
  }

  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;

  return vec_dim;
}
int main(int argc, char** argv) {
  if (argc != 8) {
    cout << "binary -data_path -index_path "
         << "-query_path -search_result_path "
         << "-k -search_k -extend_search"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string query_path = argv[3];
  string search_result_path = argv[4];

  util::GraphSearchParameter search_param;
  search_param.k = atoi(argv[5]);
  search_param.search_k = atoi(argv[6]);
  search_param.start_neighbor_num = 10;

  bool extend_search = atoi(argv[7]);
  cout << "extend_search: " << extend_search << endl; 

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadEmbeddingData(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->LoadIndex(index_path);

  vector<vector<string> > search_result(query_ptr->size());

  util::Log("before search");
  int num_cnt = 0;
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < query_ptr->size(); i++) {
    num_cnt += graph_index_ptr->SearchKnn((*query_ptr)[i], search_param, search_result[i], extend_search);
  }
  util::Log("end search");

  cout << "calculate data num: " << num_cnt << endl;
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
