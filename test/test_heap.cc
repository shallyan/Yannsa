#include <gtest/gtest.h>
#include <util/heap.h>

using namespace yannsa;
using namespace yannsa::util;

TEST(HeapTest, Create) {
  Heap<int> h(5);
}
