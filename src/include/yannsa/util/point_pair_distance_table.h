#ifndef YANNSA_POINT_PAIR_DISTANCE_TABLE_H
#define YANNSA_POINT_PAIR_DISTANCE_TABLE_H

#include <map>
#include <set>
#include <utility>
#include <queue>

namespace yannsa {
namespace util {

template <typename IndexType, typename DistanceType>
class PointPairDistanceTable {
  public:
    typedef std::pair<IndexType, IndexType> TableKey;

  public:
    bool Exist(IndexType point_a, IndexType point_b) {
      TableKey key = ConstructKey(point_a, point_b);
      return table.find(key) != table.end();
    }

    void Insert(IndexType point_a, IndexType point_b, DistanceType dist) {
      TableKey key = ConstructKey(point_a, point_b);
      table[key] = dist;
    }

    bool Get(IndexType point_a, IndexType point_b, DistanceType& dist) {
      TableKey key = ConstructKey(point_a, point_b);
      auto iter = table.find(key);
      if (iter == table.end()) {
        return false;
      }
      dist = iter->second;
      return true;
    }

  private:
    TableKey ConstructKey(IndexType point_a, IndexType point_b) {
      return (point_a < point_b ? std::make_pair(point_a, point_b) : std::make_pair(point_b, point_a));
    }

  private:
    std::map<TableKey, DistanceType> table;
};

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
using PointPairQueue = std::queue<std::pair<IndexType, IndexType> >;

template <typename IndexType>
using PointPairList = std::vector<std::pair<IndexType, IndexType> >;

} // namespace util
} // namespace yannsa

#endif
