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

  h.insert(10);
  ASSERT_EQ(h.size(), 1);

  int items[] = {3, 2, 1, 5, 6};
  for (int& item : items) {
    h.insert(item);
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
  h.insert(Obj(3, "abc"));
  h.insert(Obj(2, "def"));

  Heap<Obj>::iterator iter = h.begin();
  ASSERT_EQ(iter->num, 2);
  ASSERT_EQ(iter->str, "def");

  h.insert(Obj(5, "tbc"));
  Heap<Obj>::iterator iter2 = h.begin();
  ASSERT_EQ(iter2->str, "tbc");
}

