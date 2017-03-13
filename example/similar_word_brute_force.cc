#include "yannsa/util/parameter.h"
#include "yannsa/wrapper/distance.h"
#include "yannsa/wrapper/index_helper.h"
#include "yannsa/wrapper/binary_encoder.h"
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <omp.h>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

void LogTime(const std::string& prompt) {
  time_t now = time(0);
  char* dt = ctime(&now);
  cout << prompt << ": " << dt;
}

int CreateDataset(const string& file_path,
                  DatasetPtr<float>& dataset_ptr) { 

  ifstream in_file(file_path.c_str());
  cout << "data file " << file_path << endl;

  string buff;

  //read word and dim number
  int vec_num = 0, vec_dim = 0;
  getline(in_file, buff);
  stringstream count_stream(buff);
  count_stream >> vec_num >> vec_dim; 
  cout << "vec num: " << vec_num << "\t" 
       << "vec dim: " << vec_dim << endl;

  int has_read_num = 0;

  while (getline(in_file, buff)) {
    stringstream one_word_vec_stream(buff);

    //read word firstly
    string word;
    one_word_vec_stream >> word;

    //then read value
    PointVector<float> point(vec_dim);
    double value = 0.0;
    int dim_count = 0;
    while (one_word_vec_stream >> value){ 
      point[dim_count++] = value;
    }

    //check dim num
    assert(dim_count == vec_dim);

    dataset_ptr->insert(word, point);
    
    has_read_num++;
    if (has_read_num % 10000 == 0) {
      cout << "read " << has_read_num << " points" << endl;
    }
  }

  cout << "create data and query set done, data num: " 
       << dataset_ptr->size() << endl;

  return vec_dim;
}

int main() {
  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  DatasetPtr<float> querys_ptr(new Dataset<float>());

  LogTime("start read dataset");
  int point_dim = CreateDataset("data/glove_10w", dataset_ptr);
  LogTime("end read dataset");
  
  CosineBruteForceIndexPtr<float> truth_index_ptr(new CosineBruteForceIndex<float>(dataset_ptr));

  int k = 11;
  vector<string> actual_result;
  ofstream true_file("data/glove_10w_knn");

  LogTime("start query search");
  #pragma omp parallel for schedule(static)
  for(int query_id = 0; query_id < dataset_ptr->size(); query_id++) {
    truth_index_ptr->SearchKnn(dataset_ptr->data_at(query_id), k, actual_result);
    true_file << dataset_ptr->key_at(query_id) << " ";
    for (int i = 1; i < actual_result.size(); i++) {
      true_file << actual_result[i] << " ";
    }
    true_file << endl;
  }

  return 0;
}
