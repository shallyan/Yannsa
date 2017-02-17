#ifndef YANNSA_BASE_INDEX_H
#define YANNSA_BASE_INDEX_H

#include "yannsa/base/type_definition.h"
#include "yannsa/core/dataset.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename IndexType, typename DistanceType>
struct PointDistancePair {
  IndexType point_index;
  DistanceType distance;

  PointDistancePair(IndexType point, DistanceType dist) : 
                    point_index(point), distance(dist) {}
  inline bool operator<(const PointDistancePair& point_distance_pair) const {
    return distance < point_distance_pair.distance;
  }
};

template <typename KeyType, typename PointType,
          typename DistanceFuncType, typename DistanceType = float>
class BaseIndex {
  public:
    typedef Dataset<KeyType, PointType> IndexDataset; 
    typedef std::shared_ptr<IndexDataset> IndexDatasetPtr; 

  public:
    // parameters
    BaseIndex(IndexDatasetPtr& dataset_ptr) :
              dataset_ptr_(dataset_ptr), have_built_(false) {}

    inline bool HaveBuilt() {
      return have_built_;
    }

    /*
    void AddPoint(const KeyType& key, const PointType& new_point) {
      // add data to dataset
      dataset.AddPoint(key, new_point);

      // add data to knn index
      IntIndex point_index = index2key_.size();
      index2key_.push_back(key);
      index2neighbor_.push_back(IndexNode(neighbor_num_));
      this->UpdateIndex(point_index);
    }
    */

    virtual void Build() {} 

    virtual void Clear() {} 

    virtual void Search(const PointType& query, int k, std::vector<KeyType>& search_result) = 0; 

  protected:
    IndexDatasetPtr dataset_ptr_;

    // whether have built index
    bool have_built_;
};

} // namespace core 
} // namespace yannsa

#endif
