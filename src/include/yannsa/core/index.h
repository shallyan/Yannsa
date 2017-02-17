#ifndef YANNSA_CORE_H
#define YANNSA_CORE_H

#include "yannsa/base/type_definition.h"
#include "yannsa/core/dataset.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/distance.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename DistanceType>
struct PointDistancePair {
  IndexType point_index;
  DistanceType distance;

  PointDistancePair(IndexType point, DistanceType dist) : 
                    point_index(point), distance(dist) {}
  inline bool operator<(const PointDistancePair& point_distance_pair) const {
    return distance < point_distance_pair.distance;
  }
};

template <typename DistanceType>
struct IndexNode {
  IndexNode(int neighbor_num) : nearest_neighbor(neighbor_num) {}
  util::Heap<PointDistancePair<DistanceType> > nearest_neighbor;
};

template <typename KeyType, typename PointType,
          typename DistanceFuncType, typename DistanceType = float>
class Index {
  public:
    typedef std::shared_ptr<Dataset<KeyType, PointType> > IndexDatasetPtr; 
    typedef Dataset<KeyType, PointType> IndexDataset; 

  public:
    // parameters
    Index(IndexDatasetPtr& dataset_ptr, const util::IndexParameter& index_param) : 
          dataset_ptr_(dataset_ptr), index_param_(index_param), have_built_(false) {}

    void Build() {

      have_built_ = true;
    }

    void Clear() {
      index2key_.clear();
      index2neighbor_.clear();
    }

    /*
    void AddPoint(const KeyType& key, const PointType& new_point) {
      // add data to dataset
      dataset.AddPoint(key, new_point);

      // add data to knn index
      IndexType point_index = index2key_.size();
      index2key_.push_back(key);
      index2neighbor_.push_back(IndexNode(neighbor_num_));
      this->UpdateIndex(point_index);
    }
    */

    void Search(const PointType& query, std::vector<KeyType>& search_result) {
      // Init some points, search from these points
    }

  private:
    std::vector<KeyType> index2key_;
    std::vector<IndexNode<DistanceType> > index2neighbor_;

    IndexDatasetPtr dataset_ptr_;
    util::IndexParameter index_param_;

    // whether have built index
    bool have_built_;
};

template <typename CoordinateType>
using CosineIndex = Index<int, 
                          util::PointVector<CoordinateType>, 
                          util::CosineDistance<CoordinateType>, 
                          CoordinateType>;

} // namespace core 
} // namespace yannsa

#endif
