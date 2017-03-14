#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/test_helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::test;

void ReadGroundTruth(const string& file_path,
                     unordered_map<string, vector<string> >& ground_truth) {
  ifstream in_file(file_path.c_str());
  cout << "result file " << file_path << endl;

  string buff;
  while (getline(in_file, buff)) {
    stringstream one_word_vec_stream(buff);

    //read word firstly
    string word;
    one_word_vec_stream >> word;

    //then read value
    vector<string> one_result;
    string one_neighbor;
    while (one_word_vec_stream >> one_neighbor){ 
      one_result.push_back(one_neighbor);
    }

    ground_truth[word] = one_result;
  }

  cout << "read result done, data num: " 
       << ground_truth.size() << endl;

}

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
    Normalize(point);

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
  int point_dim = CreateDataset("data/glove_10w", dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));
  util::GraphIndexParameter param;
  param.point_neighbor_num = 10;
  param.bucket_key_point_num = 10;
  param.bucket_neighbor_num = 10;
  param.min_bucket_size = 50;
  param.max_bucket_size = 200;
  BaseEncoderPtr<PointVector<float> > 
      binary_encoder_ptr(new BinaryEncoder<PointVector<float>, float>(point_dim, 10));

  graph_index_ptr->Build(param, binary_encoder_ptr);

  unordered_map<string, vector<string> > ground_truth; 
  ReadGroundTruth("data/glove_10w_knn", ground_truth);
  vector<string> actual_result;
  vector<string> graph_result;
  vector<string> result_intersection;
  int k = 10;
  int hit_count = 0;
  auto iter = dataset_ptr->begin();
  for(int query_id = 0; iter != dataset_ptr->end(); iter++, query_id++) {
    if (query_id % 10000 == 0) {
      cout << query_id << endl;
    }
    
    graph_index_ptr->GraphKnn(iter->first, k, graph_result);
    
    actual_result.clear();
    actual_result.insert(actual_result.end(), ground_truth[iter->first].begin(), 
                                              ground_truth[iter->first].begin()+k);
    sort(actual_result.begin(), actual_result.end());
    sort(graph_result.begin(), graph_result.end());

    result_intersection.clear();
    set_intersection(actual_result.begin(), actual_result.end(), 
                     graph_result.begin(), graph_result.end(), 
                     back_inserter(result_intersection));
    int cur_hit_count = result_intersection.size();
    hit_count += cur_hit_count;
    //cout << "precision: " << cur_hit_count * 1.0 / k << endl << endl;
  }
  cout << "average precision: " << hit_count * 1.0 / (k *dataset_ptr->size()) << endl;

  return 0;
}
