#include "yannsa/core/base_index.h"
#include "yannsa/core/graph_index.h"
#include "yannsa/core/brute_force_index.h"
#include "yannsa/wrapper/distance_helper.h"
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
  CosineGraphIndex<float>::IndexDatasetPtr dataset_ptr(new CosineGraphIndex<float>::IndexDataset()); 

  CosineGraphIndex<float>::IndexPoint point_a(3);
  point_a << 0.1, 0.2, 0.3;

  dataset_ptr->AddPoint("a", point_a);
  ASSERT_EQ(dataset_ptr->Size(), 1);

  CosineGraphIndex<float> index(dataset_ptr);
  ASSERT_FALSE(index.HaveBuilt());
}

TEST(IndexTest, BruteForceIndexSearch) {
  CosineBruteForceIndex<float>::IndexDatasetPtr dataset_ptr(new CosineBruteForceIndex<float>::IndexDataset()); 

  CosineBruteForceIndex<float>::IndexPoint point_a(3);
  point_a << 0.1, 0.2, 0.3;
  CosineBruteForceIndex<float>::IndexPoint point_b(3);
  point_b << 0.0, 0.1, 0.2;
  CosineBruteForceIndex<float>::IndexPoint point_c(3);
  point_c << 0.11, 0.2, 0.3;
  CosineBruteForceIndex<float>::IndexPoint point_d(3);
  point_d << 0.9, 0.1, 0.2;

  dataset_ptr->AddPoint("a", point_a);
  dataset_ptr->AddPoint("b", point_b);
  dataset_ptr->AddPoint("c", point_c);
  dataset_ptr->AddPoint("d", point_d);
  dataset_ptr->AddPoint("e", point_d);
  dataset_ptr->AddPoint("f", point_d);
  dataset_ptr->AddPoint("g", point_d);
  ASSERT_EQ(dataset_ptr->Size(), 7);

  CosineBruteForceIndex<float> index(dataset_ptr);
  vector<string> result;
  index.Search(point_a, 3, result);
  ASSERT_EQ(result[0], "a");
  ASSERT_EQ(result[1], "c");
  ASSERT_EQ(result[2], "b");

  index.Search(point_d, 3, result);
  ASSERT_GE(result[0], "d");
  ASSERT_GE(result[1], "d");
  ASSERT_GE(result[2], "d");
}
