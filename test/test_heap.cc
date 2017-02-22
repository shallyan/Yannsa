#include "yannsa/util/heap.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

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

  Heap<int>::Iterator iter = h.Begin();
  ASSERT_EQ(*iter, 2);

  h.Sort();
  Heap<int>::Iterator iter2 = h.Begin();
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
  };

  Heap<Obj> h(3);
  h.Insert(Obj(3, "abc"));
  h.Insert(Obj(2, "def"));

  Heap<Obj>::Iterator iter = h.Begin();
  ASSERT_EQ(iter->num, 2);
  ASSERT_EQ(iter->str, "def");

  h.Insert(Obj(5, "tbc"));
  Heap<Obj>::Iterator iter2 = h.Begin();
  ASSERT_EQ(iter2->str, "tbc");
}

