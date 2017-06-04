#include "yannsa/util/sorted_array.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

TEST(SortedArrayTest, Create) {
  SortedArray<int> h(5);
  ASSERT_EQ(h.size(), 0);
}

TEST(SortedArrayTest, SortedArray) {
  SortedArray<int> h(3);
  ASSERT_EQ(h.size(), 0);
  h.insert_array(3);
  ASSERT_EQ(h.size(), 1);
  h.insert_array(3);
  ASSERT_EQ(h.size(), 1);
  h.insert_array(2);
  ASSERT_EQ(h.size(), 2);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  h.insert_array(3);
  ASSERT_EQ(h.size(), 2);
  h.insert_array(2);
  ASSERT_EQ(h.size(), 2);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  h.insert_array(3);
  ASSERT_EQ(h.size(), 2);
  h.insert_array(5);
  ASSERT_EQ(h.size(), 3);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 5);
  h.insert_array(5);
  ASSERT_EQ(h.size(), 3);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 5);
  h.insert_array(6);
  ASSERT_EQ(h.size(), 3);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 5);
  h.insert_array(4);
  ASSERT_EQ(h.size(), 3);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 4);

  h.remax_size(4);
  ASSERT_EQ(h.size(), 3);

  h.insert_array(5);
  ASSERT_EQ(h.size(), 4);
  ASSERT_EQ(h[0], 2);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 4);
  ASSERT_EQ(h[3], 5);

  h.remax_size(5);
  h.insert_array(1);
  ASSERT_EQ(h.size(), 5);
  ASSERT_EQ(h[0], 1);
  ASSERT_EQ(h[1], 2);
  ASSERT_EQ(h[2], 3);
  ASSERT_EQ(h[3], 4);
  ASSERT_EQ(h[4], 5);
}
