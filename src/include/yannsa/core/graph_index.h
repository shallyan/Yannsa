#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include <vector>
#include <string>
#include <memory>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class GraphIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::Dataset Dataset;
    typedef typename BaseClass::DatasetPtr DatasetPtr;
    typedef typename BaseClass::Dataset::Iterator Iterator;
    typedef PointType PointVector;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param) {
      Clear();

      Iterator iter = this->dataset_ptr_->Begin();
      while (iter != this->dataset_ptr_->End()) {
        iter++;
      }
        
      this->have_built_ = true;
    }

    void Clear() {
      index2key_.clear();
      index2neighbor_.clear();
    }

    void Search(const PointType& query, int k, std::vector<std::string>& search_result) {
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
