#include "yannsa/util/random_generator.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

TEST(RandomGeneratorTest, IntRandomGenerator) {
  IntRandomGenerator rg(1, 3);
  int test_num = 10;
  for (int i = 0; i < test_num; i++) {
    int rand_num = rg.Random();
    ASSERT_TRUE(rand_num >= 1);
    ASSERT_TRUE(rand_num <= 3);
  }
}
