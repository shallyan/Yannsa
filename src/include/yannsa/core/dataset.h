#ifndef YANNSA_DATASET_H
#define YANNSA_DATASET_H

#include "yannsa/base/error_definition.h"
#include "yannsa/base/type_definition.h"
#include <map>
#include <vector>

namespace yannsa {
namespace core {

template <typename KeyType, typename PointType>
class Dataset {
  // rearrange dataset for cache efficiency
  public:
    void AddPoint(const KeyType& key, const PointType& new_point) {
      if (key2index_.find(key) != key2index_.end()) {
        // key exist
        throw DataKeyExistError("Key already exists in dataset!");
      }

      key2index_[key] = index2point_.size();
      index2point_.push_back(new_point);
    }

    inline size_t Size() const {
      return index2point_.size();
    }

    inline IndexType IndexOf(const KeyType& key) {
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
  private:
    // key to index
    std::map<KeyType, IndexType> key2index_;
    // index to point 
    std::vector<PointType> index2point_;
};

} // namespace core 
} // namespace yannsa

#endif
