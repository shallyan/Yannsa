#ifndef YANNSA_POINT_PAIR_H
#define YANNSA_POINT_PAIR_H

#include <map>
#include <set>
#include <utility>
#include <queue>

namespace yannsa {
namespace util {

template <typename IndexType>
class PointPairSet {
  public:
    typedef std::pair<IndexType, IndexType> Key;
    typedef typename std::set<Key>::iterator iterator;

  public:
    bool exist(IndexType point_a, IndexType point_b) {
      Key key = ConstructKey(point_a, point_b);
      return table.find(key) != table.end();
    }

    void insert(IndexType point_a, IndexType point_b) {
      Key key = ConstructKey(point_a, point_b);
      table.insert(key);
    }

    void erase(IndexType point_a, IndexType point_b) {
      Key key = ConstructKey(point_a, point_b);
      table.erase(key);
    }

    iterator begin() {
      return table.begin();
    }

    iterator end() {
      return table.end();
    }

    inline int size() {
      return table.size();
    }
  private:
    Key ConstructKey(IndexType point_a, IndexType point_b) {
      return (point_a < point_b ? std::make_pair(point_a, point_b) : std::make_pair(point_b, point_a));
    }

  private:
    std::set<Key> table;
};

template <typename IndexType>
using PointPair = std::pair<IndexType, IndexType>; 

template <typename IndexType>
using PointPairList = std::vector<PointPair<IndexType> >;

} // namespace util
} // namespace yannsa

#endif
