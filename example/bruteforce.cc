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

  return vec_dim;
}
int main(int argc, char** argv) {
  if (argc != 4) {
    cout << "binary -data_path -graph_path -point_neighbor_num "
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string graph_path = argv[2];
  int point_neighbor_num = atoi(argv[3]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadEmbeddingData(data_path, dataset_ptr);
  
  DotBruteForceIndexPtr<float> brute_index_ptr(new DotBruteForceIndex<float>(dataset_ptr));

  int k = 11;
  vector<vector<string> > real_knn(dataset_ptr->size());
  int count = 0;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < dataset_ptr->size(); i++) {
    brute_index_ptr->SearchKnn((*dataset_ptr)[i], k, real_knn[i]);
    #pragma omp atomic
    count += 1;
    if (count % 10000 == 0) {
      cout << "finish " << count << endl;
    }
  }

  ofstream save_file(graph_path);
  for (int i = 0; i < real_knn.size(); i ++) {
    string key = dataset_ptr->GetKeyById(i);
    save_file << key << " ";
    if (real_knn[i][0] != key) {
      cout << "error" << endl;
      exit(1);
    }
    for (int j = 1; j < real_knn[i].size(); j++) {
      save_file << real_knn[i][j] << " ";
    }
    save_file << endl;
  }
  save_file.close();

  return 0;
}
