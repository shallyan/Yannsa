#include "yannsa/wrapper/binary_encoder.h"
#include <gtest/gtest.h>
#include <string>

using namespace std;
using namespace yannsa;
using namespace yannsa::wrapper;

TEST(BinaryEncoderTest, RandomGenerator) {
  RealRandomGenerator<float> rg(-1.0, 1.0);
  int test_num = 100;
  for (int i = 0; i < test_num; i++) {
    float rand_num = rg.Random();
    ASSERT_TRUE(rand_num > -1.00001);
    ASSERT_TRUE(rand_num < 1.00001);
  }
}

TEST(BinaryEncoderTest, BinaryEncoder) {
  BinaryEncoder<PointVector<float>, float> bin_encoder(2, 2);
  PointVector<float> point(2);
  point << 1.0, 1.0;
  IntCode code = bin_encoder.Encode(point);
  ASSERT_TRUE(code >= 0);
  ASSERT_TRUE(code <= 3);
}
