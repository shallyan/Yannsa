#ifndef YANNSA_DATASET_H
#define YANNSA_DATASET_H

#include "yannsa/util/common.h"
#include <map>
#include <vector>

namespace yannsa {
namespace util {

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

    inline size_t Size() {
      return index2point_.size();
    }

  private:
    // key to index
    std::map<KeyType, int> key2index_;
    // index to point 
    std::vector<PointType> index2point_;
};

} // namespace util
} // namespace yannsa

#endif
