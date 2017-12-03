#ifndef YANNSA_BASE_INDEX_H
#define YANNSA_BASE_INDEX_H

#include "yannsa/base/type_definition.h"
#include "yannsa/util/container.h"
#include "yannsa/util/sorted_array.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename IndexType, typename DistanceType>
struct PointDistancePair {
  IndexType id;
  DistanceType distance;
  bool flag;

  PointDistancePair(IndexType point_id=0, DistanceType dist=0, bool f=true) : 
                    id(point_id), distance(dist), flag(f) {}

  inline bool operator<(const PointDistancePair& point_distance_pair) const {
    return distance < point_distance_pair.distance;
  }

  inline bool operator==(const PointDistancePair& point_distance_pair) const {
    return id == point_distance_pair.id;
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

    inline void CheckIndexIsNotBuilt() {
      if (have_built_) {
        throw IndexBuildError("index has already been built!");
      }
    }

    inline void CheckIndexIsBuilt() {
      if (!have_built_) {
        throw IndexBuildError("index has not been built!");
      }
    }

    inline void SetIndexBuiltFlag() {
      have_built_ = true;
    }

    virtual void Build() {} 

    virtual void Clear() {} 
  protected:
    DatasetPtr dataset_ptr_;

    // whether index has been built
    bool have_built_;
};

} // namespace core 
} // namespace yannsa

#endif
