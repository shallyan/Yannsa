#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/code.h"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <iostream>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class GraphIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::Dataset Dataset;
    typedef typename BaseClass::DatasetPtr DatasetPtr;
    typedef typename BaseClass::PointVector PointVector;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param, util::BaseCoder<PointType>& coder) {
      Clear();

      std::map<IntCode, std::vector<IntIndex> > buckets; 
      typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
      while (iter != this->dataset_ptr_->End()) {
        std::string& key = iter->first;
        PointType& point = iter->second;

        // record point key and index
        IntIndex point_index = index2key_.size();
        index2key_.push_back(key);

        // encode point
        IntCode point_code = coder.Code(point);
        buckets[point_code].push_back(point_index);

        iter++;
      }
      
      std::map<IntCode, std::vector<IntIndex> >::iterator it = buckets.begin(); 
      for(; it != buckets.end(); it++) {
        std::cout << it->first << " : " << it->second.size() << std::endl;
      }

      this->have_built_ = true;
    }

    void Clear() {
      index2key_.clear();
      index2neighbor_.clear();
    }

    void SearchKnn(const PointType& query, int k, std::vector<std::string>& search_result) {
      search_result.clear();
      // Init some points, search from these points
    }


  private:
    struct IndexNode {
      IndexNode(int neighbor_num) : nearest_neighbor(neighbor_num) {}
      util::Heap<PointDistancePair<std::string, DistanceType> > nearest_neighbor;
    };

  private:
    std::vector<std::string> index2key_;
    std::vector<IndexNode> index2neighbor_;
};

} // namespace core 
} // namespace yannsa

#endif
