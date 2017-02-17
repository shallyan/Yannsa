#ifndef YANNSA_DATASET_H
#define YANNSA_DATASET_H

#include "yannsa/base/error_definition.h"
#include "yannsa/base/type_definition.h"
#include <unordered_map>
#include <vector>

namespace yannsa {
namespace core {

template <typename KeyType, typename PointType>
struct KeyPointPair {
  KeyType key;
  PointType point;
  KeyPointPair(KeyType k, PointType p) : key(k), point(p) {}
};

template <typename KeyType, typename PointType>
class Dataset {
  public:
    typedef KeyPointPair<KeyType, PointType> DataKeyPointPair;
    typedef typename std::vector<DataKeyPointPair>::iterator DataIterator;

  public:
    void AddPoint(const KeyType& key, const PointType& new_point) {
      if (key2index_.find(key) != key2index_.end()) {
        throw DataKeyExistError("Key already exists in dataset!");
      }

      key2index_[key] = index2point_.size();
      index2point_.push_back(DataKeyPointPair(key, new_point));
    }

    inline size_t Size() const {
      return index2point_.size();
    }

    inline IntIndex IndexOf(const KeyType& key) {
      auto point_iter = key2index_.find(key);
      if (point_iter == key2index_.end()) {
        throw DataKeyNotExistError("Key does not exist in dataset!");
      }
      return point_iter->second; 
    }

    void Clear() {
      key2index_.clear();
      index2point_.clear();
    }

    inline DataIterator Begin() {
      return index2point_.begin();
    }

    inline DataIterator End() {
      return index2point_.end();
    }

  private:
    // key to index, for fast look up
    // index to point, for fast traverse
    // 2-layer storage can be used to rearrange dataset for cache efficiency
    std::unordered_map<KeyType, IntIndex> key2index_;
    std::vector<DataKeyPointPair> index2point_;
};

} // namespace core 
} // namespace yannsa

#endif
