#include "yannsa/util/point_pair_distance_table.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

TEST(PointPairDistanceTableTest, Create) {
  PointPairDistanceTable<int, float> t;
}

