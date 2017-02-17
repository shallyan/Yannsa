#include "yannsa/wrapper/distance_helper.h"
#include "yannsa/util/point_vector.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/core/dataset.h"
#include <gtest/gtest.h>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;
using namespace yannsa::core;

TEST(DatasetTest, DatasetAddPoint) {
  PointVector<float> point_a(3);
  point_a << 0.1, 0.2, 0.3;

  Dataset<string, PointVector<float> > dataset;
  dataset.AddPoint("point_a", point_a); 

  ASSERT_EQ(dataset.Size(), 1);

  ASSERT_THROW(dataset.AddPoint("point_a", point_a), DataKeyExistError); 
}

TEST(DatasetTest, DatasetIndexOf) {
  PointVector<float> point_a(3);
  point_a << 0.1, 0.2, 0.3;

  Dataset<string, PointVector<float> > dataset;
  dataset.AddPoint("point_a", point_a); 
  dataset.AddPoint("point_b", point_a); 
  dataset.AddPoint("point_c", point_a); 

  ASSERT_EQ(dataset.Size(), 3);
  ASSERT_EQ(dataset.IndexOf("point_a"), 0);
  ASSERT_EQ(dataset.IndexOf("point_b"), 1);
  ASSERT_EQ(dataset.IndexOf("point_c"), 2);

  ASSERT_THROW(dataset.IndexOf("point_x"), DataKeyNotExistError); 
}

TEST(DatasetTest, DatasetIterator) {
  PointVector<int> point_a(3);
  point_a << 1, 2, 3;

  PointVector<int> point_b(3);
  point_b << 2, 3, 4;

  typedef Dataset<string, PointVector<int> > DatasetType;
  DatasetType dataset;
  dataset.AddPoint("point_a", point_a); 
  dataset.AddPoint("point_b", point_b); 

  DatasetType::DataIterator iter = dataset.Begin();
  ASSERT_EQ(iter->key, "point_a");
  ASSERT_EQ(iter->point[0], 1);

  iter++;
  ASSERT_EQ(iter->key, "point_b");
  ASSERT_EQ(iter->point[0], 2);

  iter++;
  ASSERT_EQ(iter, dataset.End());
}
