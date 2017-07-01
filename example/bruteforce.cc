#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
#include "io.h"
#include <iostream>
#include <omp.h>
#include <fstream>
#include <utility>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "binary -data_path -query_path -search_result_path -k"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string query_path = argv[2];
  string search_result_path = argv[3];
  int k = atoi(argv[4]);

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadEmbeddingData(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadEmbeddingData(query_path, query_ptr);

  EuclideanBruteForceIndexPtr<float> brute_index_ptr(new EuclideanBruteForceIndex<float>(dataset_ptr));

  vector<IdList> real_knn(query_ptr->size());
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < query_ptr->size(); i++) {
    brute_index_ptr->SearchKnn((*query_ptr)[i], k, real_knn[i]);
  }

  ofstream save_file(search_result_path);
  for (size_t i = 0; i < real_knn.size(); i++) {
    for (size_t j = 0; j < real_knn[i].size(); j++) {
      save_file << real_knn[i][j] << " ";
    }
    save_file << endl;
  }
  save_file.close();

  return 0;
}
