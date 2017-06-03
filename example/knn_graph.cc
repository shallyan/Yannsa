#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
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
  if (argc != 6) {
    cout << "binary -data_path -graph_path -point_neighbor_num "
         << "-max_point_neighbor_num -refine_iter_num"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string graph_path = argv[2];

  util::GraphIndexParameter param;
  param.point_neighbor_num = atoi(argv[3]);
  param.max_point_neighbor_num = atoi(argv[4]);
  param.refine_iter_num = atoi(argv[5]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadEmbeddingData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->Build(param);
  graph_index_ptr->Save(graph_path);

  return 0;
}
