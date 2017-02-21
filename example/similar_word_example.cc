#include "yannsa/wrapper/distance.h"
#include "yannsa/wrapper/index_helper.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace yannsa;
using namespace yannsa::wrapper;

DatasetPtr<float> CreateDataset(const string& file_path) {
  DatasetPtr<float> dataset_ptr(new Dataset<float>());

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
    while (one_word_vec_stream >> value){ 
      point[dim_count++] = value;
    }

    //check dim num
    assert(dim_count == vec_dim);

    dataset_ptr->AddPoint(word, point);
  }

  return dataset_ptr;
}

int main() {
  DatasetPtr<float> dataset_ptr = CreateDataset("data/word_rep");

  return 0;
}
