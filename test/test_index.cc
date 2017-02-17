#include "yannsa/core/index.h"
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
  PointDistancePair<float> point_dist_a(1, 0.3);
  PointDistancePair<float> point_dist_b(1, 0.4);
  ASSERT_TRUE(point_dist_a < point_dist_b);
}

TEST(IndexTest, CreateIndex) {
  IndexParameter param;
  param.neighbor_num = 10;
  CosineIndex<float>::IndexDatasetPtr dataset_ptr(new CosineIndex<float>::IndexDataset()); 

  CosineIndex<float>::IndexPoint point_a(3);
  point_a << 0.1, 0.2, 0.3;

  dataset_ptr->AddPoint("a", point_a);
  ASSERT_EQ(dataset_ptr->Size(), 1);

  CosineIndex<float> index(dataset_ptr, param);
  ASSERT_FALSE(index.HaveBuilt());
}
