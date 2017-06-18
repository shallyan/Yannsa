#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "binary -data_path -index_path "
         << "-query_path -search_result_path "
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string query_path = argv[3];
  string search_result_path = argv[4];

  util::Log("Load data");

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);

  util::Log("Load data done");
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  util::Log("Load index");
  graph_index_ptr->LoadIndex(index_path);
  util::Log("Load index done");

  int k, search_k;
  while (true) {
    cout << "Input k and search_k: ";
    cin >> k >> search_k;

    GraphSearchParameter search_param;
    search_param.k = k;
    search_param.search_k = search_k;
    search_param.start_neighbor_num = 10;

    vector<vector<string> > search_result(query_ptr->size());

    string prompt = "Before search: ";
    prompt += "k = " + to_string(k) + " search_k = " + to_string(search_k);
    util::Log(prompt);
    int num_cnt = 0;
    for (size_t i = 0; i < query_ptr->size(); i++) {
      num_cnt += graph_index_ptr->SearchKnn((*query_ptr)[i], search_param, search_result[i]);
    }
    util::Log("End search");

    cout << "Calculate data num: " << num_cnt << endl;
    ofstream result_file(search_result_path);

    for (size_t i = 0; i < query_ptr->size(); i++) {
      result_file << query_ptr->GetKeyById(i) << " ";
      for (auto nn : search_result[i]) {
        result_file << nn << " ";
      }
      result_file << endl;
    }
    result_file.close();
  }

  return 0;
}
