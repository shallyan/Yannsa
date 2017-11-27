#ifndef YANNSA_IO_H
#define YANNSA_IO_H 

#include "yannsa/wrapper/index_helper.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>

using namespace std;
using namespace yannsa;
using namespace yannsa::wrapper;

template<typename DataType> 
void LoadBinaryData(const string& file_path,
                    DatasetPtr<DataType>& dataset_ptr) { 

  ifstream in_file(file_path, ios::binary);

  int point_dim = 0;
  in_file.read(reinterpret_cast<char*>(&point_dim), sizeof(int));

  in_file.seekg(0, ios::end);
  IntIndex point_num = in_file.tellg() / (4 + point_dim * 4);

  cout << file_path << " has " << point_num << " points and " << point_dim << " dims" << endl;
  in_file.seekg(0, ios::beg);
  for (IntIndex point_id = 0; point_id < point_num; point_id++) {
    in_file.seekg(4, ios::cur);
    PointVector<DataType> point(point_dim);
    DataType value = 0;
    for (int d = 0; d < point_dim; d++) {
      in_file.read(reinterpret_cast<char*>(&value), sizeof(DataType));
      point[d] = value;
    }

    stringstream key_str;
    key_str << point_id;
    string key;
    key_str >> key;
    dataset_ptr->insert(key, point);
  }

  in_file.close();
  cout << "create dataset done, data num: " 
       << dataset_ptr->size() << endl;
}

#endif
