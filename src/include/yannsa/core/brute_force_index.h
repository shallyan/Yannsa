#ifndef YANNSA_BRUTE_FORCE_H
#define YANNSA_BRUTE_FORCE_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/sorted_array.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class BruteForceIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;

  private:
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem;

  public:
    BruteForceIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void SearchKnn(const PointType& query, int K, std::vector<std::string>& search_result) {
      search_result.clear();

      DistanceFuncType distance_func;
      util::SortedArray<PointDistancePairItem> K_candidates(K); 

      for (IntIndex i = 0; i < this->dataset_ptr_->size(); i++) { 
        DistanceType dist = distance_func((*this->dataset_ptr_)[i], query); 
        K_candidates.insert(PointDistancePairItem(i, dist));
      } 

      auto candidate_iter = K_candidates.begin();
      for (; candidate_iter != K_candidates.end(); candidate_iter++) {
        search_result.push_back(this->dataset_ptr_->GetKeyById(candidate_iter->id));
      }
    }
};

} // namespace core 
} // namespace yannsa

#endif
