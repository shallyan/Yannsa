#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
#include <iostream>
#include <omp.h>
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
using namespace yannsa::wrapper;

void LoadEmbeddingData(const string& file_path,
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

  int has_read_num = 0;
  size_t ten_percent_num = vec_num / 10 + 10;

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
    
    has_read_num++;
    if (has_read_num % ten_percent_num == 0) {
      cout << "read " << has_read_num << " points" << endl;
    }
  }

  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "binary -data_path -query_path -search_result_path -k"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string query_path = argv[2];
  string search_result_path = argv[3];
  int k = atoi(argv[4]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);

  DotBruteForceIndexPtr<float> brute_index_ptr(new DotBruteForceIndex<float>(dataset_ptr));

  vector<vector<string> > real_knn(query_ptr->size());
  int count = 0;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < query_ptr->size(); i++) {
    brute_index_ptr->SearchKnn((*query_ptr)[i], k, real_knn[i]);
    #pragma omp atomic
    count += 1;
    if (count % 10000 == 0) {
      cout << "finish " << count << endl;
    }
  }

  ofstream save_file(search_result_path);
  for (int i = 0; i < real_knn.size(); i ++) {
    string key = query_ptr->GetKeyById(i);
    save_file << key << " ";
    /*
    // for graph knn
    if (real_knn[i][0] != key) {
      cout << "error" << endl;
      exit(1);
    }
    for (int j = 1; j < real_knn[i].size(); j++) {
    */
    for (int j = 0; j < real_knn[i].size(); j++) {
      save_file << real_knn[i][j] << " ";
    }
    save_file << endl;
  }
  save_file.close();

  return 0;
}
