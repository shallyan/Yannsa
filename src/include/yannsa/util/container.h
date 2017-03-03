#ifndef YANNSA_CONTAINER_H
#define YANNSA_CONTAINER_H 

#include "yannsa/base/error_definition.h"
#include "yannsa/base/type_definition.h"
#include <unordered_map>
#include <vector>
#include <utility>

namespace yannsa {
namespace util {

template <typename PointType>
class Container {
  public:
    typedef std::pair<std::string, PointType> KeyPointPair;
    typedef typename std::vector<KeyPointPair>::iterator Iterator;

  public:
    void Insert(const std::string& key, const PointType& new_point) {
      if (key2index_.find(key) != key2index_.end()) {
        throw KeyExistError("Key already exists!");
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

    inline Iterator Begin() {
      return index2key_point_pair_.begin();
    }

    inline Iterator End() {
      return index2key_point_pair_.end();
    }

    // thread-safe get
    inline const PointType& Get(const std::string& key) {
      // user ensure existence
      std::unordered_map<std::string, IntIndex>::const_iterator iter = key2index_.find(key);
      return index2key_point_pair_[iter->second].second; 
    }

  private:
    // key to index, for fast look up
    // index to point, for fast traverse
    // 2-layer storage can be used to rearrange dataset for cache efficiency
    std::unordered_map<std::string, IntIndex> key2index_;
    std::vector<KeyPointPair> index2key_point_pair_;
};

} // namespace util 
} // namespace yannsa

#endif
