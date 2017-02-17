#ifndef YANNSA_DATASET_H
#define YANNSA_DATASET_H

#include "yannsa/base/error_definition.h"
#include "yannsa/base/type_definition.h"
#include <unordered_map>
#include <vector>
#include <utility>

namespace yannsa {
namespace core {

template <typename PointType>
class Dataset {
  public:
    typedef std::pair<std::string, PointType> DataKeyPointPair;
    typedef typename std::vector<DataKeyPointPair>::iterator DataIterator;

  public:
    void AddPoint(const std::string& key, const PointType& new_point) {
      if (key2index_.find(key) != key2index_.end()) {
        throw DataKeyExistError("Key already exists in dataset!");
      }

      key2index_[key] = index2key_point_pair_.size();
      index2key_point_pair_.push_back(std::make_pair(key, new_point));
    }

    inline size_t Size() const {
      return index2key_point_pair_.size();
    }

    void Clear() {
      key2index_.clear();
      index2key_point_pair_.clear();
    }

    inline DataIterator Begin() {
      return index2key_point_pair_.begin();
    }

    inline DataIterator End() {
      return index2key_point_pair_.end();
    }

  private:
    // key to index, for fast look up
    // index to point, for fast traverse
    // 2-layer storage can be used to rearrange dataset for cache efficiency
    std::unordered_map<std::string, IntIndex> key2index_;
    std::vector<DataKeyPointPair> index2key_point_pair_;
};

} // namespace core 
} // namespace yannsa

#endif
