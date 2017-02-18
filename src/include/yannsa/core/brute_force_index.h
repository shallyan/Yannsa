#ifndef YANNSA_BRUTE_FORCE_H
#define YANNSA_BRUTE_FORCE_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class BruteForceIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::Dataset Dataset;
    typedef typename BaseClass::DatasetPtr DatasetPtr;
    typedef typename BaseClass::PointVector PointVector;

  private:
    typedef PointDistancePair<std::string, DistanceType> PointDistancePairItem;

  public:
    BruteForceIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void SearchKnn(const PointType& query, int k, std::vector<std::string>& search_result) {
      search_result.clear();

      DistanceFuncType distance_func;
      util::Heap<PointDistancePairItem> k_candidates(k); 

      typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
      while (iter != this->dataset_ptr_->End()) {
        DistanceType dist = distance_func(iter->second, query); 
        k_candidates.Insert(PointDistancePairItem(iter->first, dist));
        iter++;
      } 
      k_candidates.Sort();
      auto candidate_content = k_candidates.GetContent();
      for (auto& candidate : candidate_content) {
        search_result.push_back(candidate.point_index);
      }
    }
};

} // namespace core 
} // namespace yannsa

#endif
