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
  if (argc != 5) {
    cout << "binary -data_path -index_path -extend_index_path "
         << "-exploit_num"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string extend_index_path = argv[3];

  int exploit_num = atoi(argv[4]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  int point_dim = LoadEmbeddingData(data_path, dataset_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->LoadIndex(index_path);
  graph_index_ptr->Prune(exploit_num);
  graph_index_ptr->Reverse();
  graph_index_ptr->SaveIndex(extend_index_path);

  return 0;
}
