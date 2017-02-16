#ifndef YANNSA_CORE_H
#define YANNSA_CORE_H

#include "yannsa/base/type_difinition.h"
#include "yannsa/core/dataset.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/distance.h"
#include <vector>

namespace yannsa {
namespace core {

template <typename DistanceType>
struct PointDistancePair {
  IndexType point_index;
  DistanceType distance;
  inline bool operator<(const PointDistancePair& point_distance_pair) const {
    return distance < point_distance_pair.distance;
  }
};

template <typename DistanceType>
struct IndexNode {
  IndexNode(int neighbor_num) : nearest_neighbor(neighbor_num) {}
  Heap<PointDistancePair<DistanceType> > nearest_neighbor;
};

/*
template <typename KeyType, typename PointType,
          typename DistanceType, typename DistanceFuncType>
class Index {
  public:
    // parameters
    void Index(int neighbor_num) : neighbor_num_(neighbor_num) {}

    void Build() {

    }
    void AddPoint(const KeyType& key, const PointType& new_point) {
      // add data to dataset
      dataset.AddPoint(key, new_point);

      // add data to knn index
      IndexType point_index = index2key_.size();
      index2key_.push_back(key);
      index2neighbor_.push_back(IndexNode(neighbor_num_));
      this->UpdateIndex(point_index);
    }

    void UpdateIndex(IndexType new_point_index) {

    }
  private:
    std::vector<KeyType> index2key_;
    std::vector<IndexNode<DistanceType> > index2neighbor_;

    typedef shared_ptr<Dataset<KeyType, PointType> > DatasetPtr; 
    DatasetPtr dataset_ptr_;
    int neighbor_num_;
};
*/

} // namespace core 
} // namespace yannsa

#endif
