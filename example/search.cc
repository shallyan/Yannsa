#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/wrapper/index_helper.h"
#include "io.h"
#include <iostream>
#include <set>
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
    cout << "binary -data_path -index_path -query_path -ground_truth_path"
         << endl;
    return 0;
  }
  string data_path = argv[1];
  string index_path = argv[2];
  string query_path = argv[3];
  string ground_truth_path = argv[4];

  util::Log("Load data");

  DatasetPtr<float> dataset_ptr(new Dataset<float>());
  LoadBinaryData<float>(data_path, dataset_ptr);
  DatasetPtr<float> query_ptr(new Dataset<float>());
  LoadBinaryData<float>(query_path, query_ptr);
  DatasetPtr<IntIndex> ground_truth_ptr(new Dataset<IntIndex>());
  LoadBinaryData<IntIndex>(ground_truth_path, ground_truth_ptr);

  util::Log("Load data done");
  
  EuclideanGraphIndexPtr<float> graph_index_ptr(new EuclideanGraphIndex<float>(dataset_ptr));

  util::Log("Load index");
  graph_index_ptr->LoadIndex(index_path);
  util::Log("Load index done");

  int K, search_K;
  while (true) {
    cout << "Input K and search_K: ";
    cin >> K >> search_K;

    GraphSearchParameter search_param;
    search_param.K = K;
    search_param.search_K = search_K;

    vector<vector<string> > search_results(query_ptr->size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < query_ptr->size(); i++) {
      graph_index_ptr->SearchKnn((*query_ptr)[i], search_param, search_results[i]);
    }

    // evaluate
    int hit_cnt = 0;
    for (size_t i = 0; i < query_ptr->size(); i++) {
      set<IntIndex> rtn_results, true_results;
      for (size_t j = 0; j < K; j++) {
        rtn_results.insert(stoi(search_results[i][j]));
        true_results.insert((*ground_truth_ptr)[i][j]);
      }

      vector<IntIndex> hit_results;
      set_intersection(rtn_results.begin(), rtn_results.end(),
                       true_results.begin(), true_results.end(), back_inserter(hit_results));
      hit_cnt += hit_results.size();
    }
    cout << K << "-NN Precision: " << hit_cnt * 1.0 / (query_ptr->size() * K) << endl;
  }

  return 0;
}
