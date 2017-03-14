#include "yannsa/wrapper/distance.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/util/container.h"
#include <gtest/gtest.h>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

TEST(ContainerTest, AddPoint) {
  PointVector<float> point_a(3);
  point_a << 0.1, 0.2, 0.3;

  Container<PointVector<float> > dataset;
  dataset.insert("point_a", point_a); 

  ASSERT_EQ(dataset.size(), 1);

  ASSERT_THROW(dataset.insert("point_a", point_a), KeyExistError); 
}

TEST(ContainerTest, Visit) {
  PointVector<int> point_a(3);
  point_a << 1, 2, 3;

  PointVector<int> point_b(3);
  point_b << 2, 3, 4;

  typedef Container<PointVector<int> > Dataset;
  Dataset dataset;
  dataset.insert("point_a", point_a); 
  dataset.insert("point_b", point_b); 

  ASSERT_EQ(dataset.GetKeyById(0), "point_a");
  ASSERT_EQ(dataset[0][0], 1);
}
