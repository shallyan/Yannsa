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
  if (argc != 7) {
    cout << "binary -data_path -index_path "
         << "-query_path -search_result_path "
         << "-k -search_k"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string query_path = argv[3];
  string search_result_path = argv[4];

  util::GraphSearchParameter search_param;
  search_param.k = atoi(argv[5]);
  search_param.search_k = atoi(argv[6]);
  search_param.start_neighbor_num = 10;

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  graph_index_ptr->LoadIndex(index_path);

  vector<vector<string> > search_result(query_ptr->size());

  util::Log("before search");
  int num_cnt = 0;
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < query_ptr->size(); i++) {
    num_cnt += graph_index_ptr->SearchKnn((*query_ptr)[i], search_param, search_result[i]);
  }
  util::Log("end search");

  cout << "calculate data num: " << num_cnt << endl;
  ofstream result_file(search_result_path);

  for (int i = 0; i < query_ptr->size(); i++) {
    result_file << query_ptr->GetKeyById(i) << " ";
    for (auto nn : search_result[i]) {
      result_file << nn << " ";
    }
    result_file << endl;
  }
  result_file.close();

  return 0;
}
