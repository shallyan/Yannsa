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
    typedef typename std::vector<PointType>::iterator iterator;

  public:
    void insert(const std::string& key, const PointType& new_point) {
      if (key2id_.find(key) != key2id_.end()) {
        throw KeyExistError("Key already exists!");
      }

      key2id_[key] = id2point_.size();
      id2point_.push_back(new_point);
      id2key_.push_back(key);
    }

    inline size_t size() const {
      return id2point_.size();
    }

    void clear() {
      id2key_.clear();
      id2point_.clear();
      key2id_.clear();
    }

    inline const PointType& operator[] (IntIndex i) const {
      return id2point_[i];
    }

    inline const std::string& GetKeyById(IntIndex i) const {
      return id2key_[i];
    }

    inline const iterator begin() const {
      return id2point_.begin();
    }

    inline const iterator end() const {
      return id2point_.end();
    }

  private:
    std::unordered_map<std::string, IntIndex> key2id_;
    std::vector<std::string> id2key_;
    std::vector<PointType> id2point_;
};

} // namespace util 
} // namespace yannsa

#endif
