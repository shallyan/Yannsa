#include "yannsa/core/base_index.h"
#include "yannsa/core/graph_index.h"
#include "yannsa/core/brute_force_index.h"
#include "yannsa/wrapper/distance.h"
#include "yannsa/wrapper/index_helper.h"
#include "yannsa/util/parameter.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;
using namespace yannsa;
using namespace yannsa::core;
using namespace yannsa::util;
using namespace yannsa::wrapper;

TEST(IndexTest, PointDistancePair) {
  PointDistancePair<int, float> point_dist_a(1, 0.3);
  PointDistancePair<int, float> point_dist_b(1, 0.4);
  ASSERT_TRUE(point_dist_a < point_dist_b);
}

TEST(IndexTest, CreateGraphIndex) {
  DotGraphIndex<float>::DatasetPtr dataset_ptr(new DotGraphIndex<float>::Dataset()); 

  DotGraphIndex<float>::PointVector point_a(3);
  point_a << 0.1, 0.2, 0.3;

  dataset_ptr->insert("a", point_a);
  ASSERT_EQ(dataset_ptr->size(), 1);
}

TEST(IndexTest, BruteForceIndexSearch) {
  DotBruteForceIndex<float>::DatasetPtr dataset_ptr(new DotBruteForceIndex<float>::Dataset()); 

  DotBruteForceIndex<float>::PointVector point_a(3);
  point_a << 0.1, 0.2, 0.3;
  DotBruteForceIndex<float>::PointVector point_b(3);
  point_b << 0.0, 0.1, 0.2;
  DotBruteForceIndex<float>::PointVector point_c(3);
  point_c << 0.11, 0.2, 0.3;
  DotBruteForceIndex<float>::PointVector point_d(3);
  point_d << 0.9, 0.1, 0.2;

  dataset_ptr->insert("a", point_a);
  dataset_ptr->insert("b", point_b);
  dataset_ptr->insert("c", point_c);
  dataset_ptr->insert("d", point_d);
  dataset_ptr->insert("e", point_d);
  dataset_ptr->insert("f", point_d);
  dataset_ptr->insert("g", point_d);
  ASSERT_EQ(dataset_ptr->size(), 7);
}

TEST(IndexTest, BruteForceIndexSearchWithWrapperRep) {
  DatasetPtr<float> dataset_ptr(new Dataset<float>()); 

  PointVector<float> point_a(3);
  point_a << 0.1, 0.2, 0.3;
  PointVector<float> point_b(3);
  point_b << 0.0, 0.1, 0.2;
  PointVector<float> point_c(3);
  point_c << 0.11, 0.2, 0.3;
  PointVector<float> point_d(3);
  point_d << 0.9, 0.1, 0.2;

  dataset_ptr->insert("a", point_a);
  dataset_ptr->insert("b", point_b);
  dataset_ptr->insert("c", point_c);
  dataset_ptr->insert("d", point_d);
  dataset_ptr->insert("e", point_d);
  dataset_ptr->insert("f", point_d);
  dataset_ptr->insert("g", point_d);
  ASSERT_EQ(dataset_ptr->size(), 7);

  DotBruteForceIndex<float> index(dataset_ptr);
  /*
  vector<string> result;
  index.SearchKnn(point_a, 3, result);
  ASSERT_EQ(result[0], "a");
  ASSERT_EQ(result[1], "c");
  ASSERT_EQ(result[2], "b");

  index.SearchKnn(point_d, 3, result);
  ASSERT_GE(result[0], "d");
  ASSERT_GE(result[1], "d");
  ASSERT_GE(result[2], "d");
  */
}
