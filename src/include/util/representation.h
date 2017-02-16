#ifndef YANNSA_REPRESENTATION_H
#define YANNSA_REPRESENTATION_H

#include "util/common.h"
#include <eigen/Dense>
#include <map>
#include <vector>

namespace yannsa {
namespace util {

// use Eigen to represent vector temporarily
template <typename VectorCoordinateType>
using PointVector = Eigen::Matrix<VectorCoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename KeyType, typename VectorCoordinateType>
class DataSet {
  // rearrange dataset for cache efficiency
  public:
    void AddPoint(const KeyType& key, const PointVector<VectorCoordinateType>& new_point) {
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
    std::vector<PointVector<VectorCoordinateType> > index2point_;
};

} // namespace util
} // namespace yannsa

#endif
