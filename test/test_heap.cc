#include <gtest/gtest.h>
#include <util/heap.h>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

TEST(HeapTest, Create) {
  Heap<int> h(5);
  ASSERT_EQ(h.Size(), 0);
}

TEST(HeapTest, Insert) {
  Heap<int> h(3);

  h.Insert(10);
  ASSERT_EQ(h.Size(), 1);

  int items[] = {3, 2, 1, 5, 2, 6, 3, 6};
  for (int& item : items) {
    h.Insert(item);
  }
  ASSERT_EQ(h.Size(), 3);

  vector<int>& content = h.GetContent();
  ASSERT_EQ(content[0], 2);
}
