#include "yannsa/util/point_vector.h"
#include <gtest/gtest.h>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

const float precision = 0.000001;

TEST(PointVectorTest, CreatePointVector) {
  PointVector<float> point_a(3);
  point_a[0] = 1.0;
  point_a[1] = 2.0;
  point_a[2] = 0.5;

  PointVector<float> point_b(3);
  point_b << 1.0, 0.3, 2.0;

  ASSERT_NEAR(point_a.dot(point_b), 2.6, precision);
}
