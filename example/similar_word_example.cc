#include "yannsa/util/parameter.h"
#include "yannsa/wrapper/distance.h"
#include "yannsa/wrapper/index_helper.h"
#include "yannsa/wrapper/binary_encoder.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <ctime>

#define LITTLE_DATA_TEST

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
                  DatasetPtr<float>& dataset_ptr, 
                  DatasetPtr<float>& querys_ptr,
                  float query_ratio) {

  ifstream in_file(file_path.c_str());
  string buff;

  //read word and dim number
  int vec_num = 0, vec_dim = 0;
  getline(in_file, buff);
  stringstream count_stream(buff);
  count_stream >> vec_num >> vec_dim; 
  cout << "vec num: " << vec_num << "\t" 
       << "vec dim: " << vec_dim << endl;

  int querys_num = static_cast<int>(vec_num * query_ratio);
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

    if (has_read_num < querys_num) {
      querys_ptr->Insert(word, point);
    }
    else {
      dataset_ptr->Insert(word, point);
    }
    
    has_read_num++;
    if (has_read_num % 10000 == 0) {
      cout << "read " << has_read_num << " points" << endl;
    }
  }

  cout << "create data and query set done, data num: " 
       << dataset_ptr->Size() << " query num: " << querys_ptr->Size() << endl;

  return vec_dim;
}

int main() {
  float query_ratio = 0.0001;
  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  DatasetPtr<float> querys_ptr(new Dataset<float>());
  LogTime("start read dataset");
#if defined(LITTLE_DATA_TEST)
  int point_dim = CreateDataset("data/word_rep", dataset_ptr, querys_ptr, query_ratio);
#elif defined(LARGE_DATA_TEST)
  int point_dim = CreateDataset("data/glove.twitter.27B.100d.txt", dataset_ptr, querys_ptr, query_ratio);
#endif
  LogTime("end read dataset");
  
  CosineBruteForceIndexPtr<float> truth_index_ptr(new CosineBruteForceIndex<float>(dataset_ptr));
  CosineGraphIndexPtr<float> graph_index_ptr(new CosineGraphIndex<float>(dataset_ptr));
  util::GraphIndexParameter param;
  param.point_neighbor_num = 10;
  param.bucket_key_point_num = 10;
  param.bucket_neighbor_num = 10;
#if defined(LITTLE_DATA_TEST)
  param.min_bucket_size = 50;
  param.max_bucket_size = 200;
#elif defined(LARGE_DATA_TEST)
  param.min_bucket_size = 500;
  param.max_bucket_size = 2000;
#endif
  BaseEncoderPtr<PointVector<float> > 
      binary_encoder_ptr(new BinaryEncoder<PointVector<float>, float>(point_dim, 10));

  LogTime("start build index");
  graph_index_ptr->Build(param, binary_encoder_ptr);
  LogTime("end build index");

  vector<string> actual_result;
  vector<string> graph_result;
  vector<string> result_intersection;
  int k = 1;
  int hit_count = 0;
  auto iter = querys_ptr->Begin();
  for(int query_id = 0; iter != querys_ptr->End(); iter++, query_id++) {
    truth_index_ptr->SearchKnn(iter->second, k, actual_result);
    graph_index_ptr->SearchKnn(iter->second, k, graph_result);

    cout << "[" << query_id << "]" << iter->first << endl;
    cout << "actual result: ";
    for (auto& one_nn : actual_result) {
      cout << one_nn << " ";
    }
    cout << endl;
    cout << "graph result: ";
    for (auto& one_nn : graph_result) {
      cout << one_nn << " ";
    }
    cout << endl;
    
    sort(actual_result.begin(), actual_result.end());
    sort(graph_result.begin(), graph_result.end());

    result_intersection.clear();
    set_intersection(actual_result.begin(), actual_result.end(), 
                     graph_result.begin(), graph_result.end(), 
                     back_inserter(result_intersection));
    int cur_hit_count = result_intersection.size();
    hit_count += cur_hit_count;
    cout << "precision: " << cur_hit_count * 1.0 / k << endl << endl;
  }
  cout << "average precision: " << hit_count * 1.0 / k *querys_ptr->Size() << endl;

  return 0;
}
