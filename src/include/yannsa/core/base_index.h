#ifndef YANNSA_BASE_INDEX_H
#define YANNSA_BASE_INDEX_H

#include "yannsa/base/type_definition.h"
#include "yannsa/util/container.h"
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

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class BaseIndex {
  public:
    typedef util::Container<PointType> Dataset; 
    typedef std::shared_ptr<Dataset> DatasetPtr; 
    typedef PointType PointVector;

  public:
    // parameters
    BaseIndex(DatasetPtr& dataset_ptr) :
              dataset_ptr_(dataset_ptr), have_built_(false) {}

    inline bool HaveBuilt() {
      return have_built_;
    }

    virtual void Build() {} 

    virtual void Clear() {} 

    virtual void Search(const PointType& query, int k, std::vector<std::string>& search_result) = 0; 

  protected:
    DatasetPtr dataset_ptr_;

    // whether have built index
    bool have_built_;
};

} // namespace core 
} // namespace yannsa

#endif
