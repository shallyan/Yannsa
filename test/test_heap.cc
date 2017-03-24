#include "yannsa/util/heap.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;
using namespace yannsa;
using namespace yannsa::util;

TEST(HeapTest, Create) {
  Heap<int> h(5);
  ASSERT_EQ(h.size(), 0);
}

TEST(HeapTest, Insert) {
  Heap<int> h(3);

  h.insert_heap(10);
  ASSERT_EQ(h.size(), 1);

  int items[] = {3, 2, 1, 5, 6};
  for (int& item : items) {
    h.insert_heap(item);
  }
  ASSERT_EQ(h.size(), 3);

  Heap<int>::iterator iter = h.begin();
  ASSERT_EQ(*iter, 3);

  h.sort();
  Heap<int>::iterator iter2 = h.begin();
  ASSERT_EQ(*iter2, 1);
}

TEST(HeapTest, InsertObj) {
  struct Obj {
    int num;
    string str; 

    Obj(int n, string s) : num(n), str(s) {}
    inline bool operator<(const Obj& obj_a) const {
      return str < obj_a.str;
    }
    inline bool operator==(const Obj& obj_a) const {
      return str == obj_a.str;
    }
  };

  Heap<Obj> h(3);
  h.insert_heap(Obj(3, "abc"));
  h.insert_heap(Obj(2, "def"));

  Heap<Obj>::iterator iter = h.begin();
  ASSERT_EQ(iter->num, 2);
  ASSERT_EQ(iter->str, "def");

  h.insert_heap(Obj(5, "tbc"));
  Heap<Obj>::iterator iter2 = h.begin();
  ASSERT_EQ(iter2->str, "tbc");
}

TEST(HeapTest, SortedArray) {
  Heap<int> h(3);
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
