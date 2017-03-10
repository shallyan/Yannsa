#ifndef YANNSA_POINT_PAIR_DISTANCE_TABLE_H
#define YANNSA_POINT_PAIR_DISTANCE_TABLE_H

#include <map>
#include <set>
#include <utility>
#include <queue>

namespace yannsa {
namespace util {

template <typename IndexType>
class PointPairTable {
  public:
    typedef std::pair<IndexType, IndexType> TableKey;
    typedef typename std::set<TableKey>::iterator Iterator;

  public:
    bool Exist(IndexType point_a, IndexType point_b) {
      TableKey key = ConstructKey(point_a, point_b);
      return table.find(key) != table.end();
    }

    void Insert(IndexType point_a, IndexType point_b) {
      TableKey key = ConstructKey(point_a, point_b);
      table.insert(key);
    }

    void Erase(IndexType point_a, IndexType point_b) {
      TableKey key = ConstructKey(point_a, point_b);
      table.erase(key);
    }

    Iterator Begin() {
      return table.begin();
    }

    Iterator End() {
      return table.end();
    }

    inline int Size() {
      return table.size();
    }
  private:
    TableKey ConstructKey(IndexType point_a, IndexType point_b) {
      return (point_a < point_b ? std::make_pair(point_a, point_b) : std::make_pair(point_b, point_a));
    }

  private:
    std::set<TableKey> table;
};

template <typename IndexType>
using PointPair = std::pair<IndexType, IndexType>; 

template <typename IndexType>
using PointPairList = std::vector<PointPair<IndexType> >;

} // namespace util
} // namespace yannsa

#endif
