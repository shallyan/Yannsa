#ifndef YANNSA_BRUTE_FORCE_H
#define YANNSA_BRUTE_FORCE_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/dataset.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <memory>

namespace yannsa {
namespace core {

template <typename KeyType, typename PointType,
          typename DistanceFuncType, typename DistanceType = float>
class BruteForceIndex : public BaseIndex<KeyType, PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<KeyType, PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::IndexDataset IndexDataset;
    typedef typename BaseClass::IndexDatasetPtr IndexDatasetPtr;
    typedef typename BaseClass::IndexDataset::DataIterator DataIterator;
    typedef PointType IndexPoint;

  public:
    BruteForceIndex(typename BaseClass::IndexDatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Search(const PointType& query, int k, std::vector<KeyType>& search_result) {
      search_result.clear();

      DistanceFuncType distance_func;
      util::Heap<PointDistancePair<KeyType, DistanceType> > k_candidates(k); 

      DataIterator iter = this->dataset_ptr_->Begin();
      while (iter != this->dataset_ptr_->End()) {
        DistanceType dist = distance_func(iter->point, query); 
        k_candidates.Insert(PointDistancePair<KeyType, DistanceType>(iter->key, dist));
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
