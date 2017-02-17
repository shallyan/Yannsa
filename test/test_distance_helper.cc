#include "yannsa/wrapper/distance_helper.h"
#include "yannsa/util/point_vector.h"
#include <gtest/gtest.h>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;
using namespace yannsa::wrapper;

const float precision = 0.000001;

TEST(DistanceHelperTest, DotDistance) {
  PointVector<float> point_a(3);
  point_a[0] = 1.0;
  point_a[1] = 2.0;
  point_a[2] = 0.5;

  PointVector<float> point_b(3);
  point_b << 1.0, 0.3, 2.0;

  DotDistance<float> dot_distance_func;
  ASSERT_NEAR(dot_distance_func(point_a, point_b), -2.6, precision);
}

TEST(DistanceHelperTest, CosineDistance) {
  PointVector<float> point_a(3);
  point_a << 1.0, 1.3, 0.7;

  PointVector<float> point_b(3);
  point_b << 1.0, -2.0, 2.3;

  CosineDistance<float> cosine_distance_func ;
  ASSERT_NEAR(cosine_distance_func(point_a, point_b), -0.001748, precision);
}

TEST(DistanceHelperTest, EuclideanDistance) {
  PointVector<float> point_a(3);
  point_a << 1.0, 1.3, 0.7;

  PointVector<float> point_b(3);
  point_b << 1.0, -2.0, 2.3;

  EuclideanDistance<float> euclidean_distance_func ;
  ASSERT_NEAR(euclidean_distance_func(point_a, point_b), 3.667424, precision);
}
