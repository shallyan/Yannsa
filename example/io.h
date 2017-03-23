#include "yannsa/wrapper/test_helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <cassert>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::test;

int LoadBinaryData(string file_path, DatasetPtr<float>& dataset_ptr) { 
  ifstream in(file_path.c_str(), ios::binary);

  if (!in.is_open()) {
    cout << "open file error" << endl;
    exit(-1);
  }
  int vec_dim=0, vec_num=0;
  in.read((char*)&vec_dim,4);
  in.seekg(0, ios::end);
  vec_num = (size_t)in.tellg() / (vec_dim+1) / 4;
  cout << "vec num: " << vec_num << "\t" 
       << "vec dim: " << vec_dim << endl;

  in.seekg(0, ios::beg);

  size_t ten_percent_num = vec_num / 10;

  for(size_t i = 0; i < vec_num; i++){
    in.seekg(4, ios::cur);
    PointVector<float> point(vec_dim);
    for (int j = 0; j < vec_dim; j++) {
      float value;
      in.read((char*)(&value), 4);
      point[j] = value;
    }
    stringstream key_str;
    key_str << i;
    string key;
    key_str >> key;
    dataset_ptr->insert(key, point);

    if (i % ten_percent_num == 0) {
      cout << "read " << i << " points" << endl;
    }
  }
  in.close();

  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;
}

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
  size_t ten_percent_num = vec_num / 10;

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
