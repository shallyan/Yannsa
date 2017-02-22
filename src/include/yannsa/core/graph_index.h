#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <iostream>
#include <climits>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class GraphIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::Dataset Dataset;
    typedef typename BaseClass::DatasetPtr DatasetPtr;
    typedef typename BaseClass::PointVector PointVector;

  private:
    typedef std::unordered_map<IntCode, std::vector<IntIndex> > Bucket2Point; 
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::unordered_map<IntCode, BucketHeap> BucketKnnGraph;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoder<PointType>& encoder); 

    void Clear() {
      index2key_.clear();
      index2neighbor_.clear();
    }

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

  private:
    void Encode2Buckets(util::BaseEncoder<PointType>& encoder, 
                        Bucket2Point& bucket2point); 

    void SplitBucket(Bucket2Point& bucket2point, 
                     IntCode cur_bucket, 
                     int max_bucket_size, 
                     int min_bucket_size);

    void MergeBucket(Bucket2Point& bucket2point,
                     IntCode cur_bucket,
                     int max_bucket_size,
                     int min_bucket_size,
                     BucketHeap& bucket_neighbor_dist,
                     std::unordered_set<IntCode>& merged_buckets); 

    void SplitAndMergeBuckets(const util::GraphIndexParameter& index_param, 
                              Bucket2Point& bucket2point, 
                              BucketKnnGraph& bucket_knn_graph); 

    void ConstructBucketsKnnGraph(const util::GraphIndexParameter& index_param, 
                                  Bucket2Point& bucket2point,
                                  BucketKnnGraph& bucket_knn_graph); 

  private:
    struct IndexNode {
      IndexNode(int neighbor_num) : nearest_neighbor(neighbor_num) {}
      util::Heap<PointDistancePair<std::string, DistanceType> > nearest_neighbor;
    };

  private:
    std::vector<std::string> index2key_;
    std::vector<IndexNode> index2neighbor_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param, 
    util::BaseEncoder<PointType>& encoder) {
  Clear();

  // encode
  Bucket2Point bucket2point; 
  Encode2Buckets(encoder, bucket2point);
  
  // construct bucket knn graph
  BucketKnnGraph bucket_knn_graph;
  ConstructBucketsKnnGraph(index_param, bucket2point, bucket_knn_graph);

  // split and merge buckets
  SplitAndMergeBuckets(index_param, bucket2point, bucket_knn_graph);

  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Encode2Buckets(
    util::BaseEncoder<PointType>& encoder, 
    Bucket2Point& bucket2point) {
  bucket2point.clear();
  typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
  while (iter != this->dataset_ptr_->End()) {
    std::string& key = iter->first;
    PointType& point = iter->second;

    // record point key and index
    IntIndex point_index = index2key_.size();
    index2key_.push_back(key);

    // encode point
    IntCode point_code = encoder.Encode(point);
    bucket2point[point_code].push_back(point_index);

    iter++;
  }
  
  /*
  std::unordered_map<IntCode, std::vector<IntIndex> >::iterator it = bucket2point.begin(); 
  for(; it != bucket2point.end(); it++) {
    std::cout << it->first << " : " << it->second.size() << std::endl;
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBucket(
    Bucket2Point& bucket2point, 
    IntCode cur_bucket, 
    int max_bucket_size, 
    int min_bucket_size) {
  int high_bits_num = sizeof(IntCode) * CHAR_BIT / 2; 
  int new_bucket_count = 0;
  while (bucket2point[cur_bucket].size() > max_bucket_size + min_bucket_size) {
    new_bucket_count++;
    // avoid overflow
    if (new_bucket_count > (1 << (high_bits_num - 1)) - 1) {
        break;
    }

    IntCode new_bucket = cur_bucket + (new_bucket_count << high_bits_num);
    auto bucket_begin_iter = bucket2point[cur_bucket].end() - max_bucket_size;
    auto bucket_end_iter = bucket2point[cur_bucket].end();
    bucket2point[new_bucket] = std::vector<IntIndex>(bucket_begin_iter, bucket_end_iter);
    bucket2point[cur_bucket].erase(bucket_begin_iter, bucket_end_iter); 
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBucket(
    Bucket2Point& bucket2point,
    IntCode cur_bucket,
    int max_bucket_size,
    int min_bucket_size,
    BucketHeap& bucket_neighbor_dist,
    std::unordered_set<IntCode>& merged_buckets) {
  if (bucket2point[cur_bucket].size() < min_bucket_size) {
    bucket_neighbor_dist.Sort();
    for (auto neighbor : bucket_neighbor_dist.GetContent()) {
      // neighbor bucket has been merged
      if (merged_buckets.find(neighbor.id) != merged_buckets.end()) {
        continue;
      }
      
      // if not exceed split threshold
      if (bucket2point[neighbor.id].size() + bucket2point[cur_bucket].size() 
          <= max_bucket_size + min_bucket_size) {
        bucket2point[cur_bucket].insert(bucket2point[cur_bucket].end(),
                                        bucket2point[neighbor.id].begin(),
                                        bucket2point[neighbor.id].end());
        merged_buckets.insert(neighbor.id);
      }

      // check
      if (bucket2point[cur_bucket].size() > min_bucket_size) {
        break;
      }
    }
  }
}


template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitAndMergeBuckets(
    const util::GraphIndexParameter& index_param, 
    Bucket2Point& bucket2point, 
    BucketKnnGraph& bucket_knn_graph) {
  // split and merge
  std::unordered_set<IntCode> merged_buckets;
  int split_threshold = static_cast<int>(index_param.max_bucket_size + 
                                         index_param.min_bucket_size);
  auto iter = bucket_knn_graph.begin(); 
  for (; iter != bucket_knn_graph.end(); iter++) {
    IntCode cur_bucket = iter->first;

    // check whether current bucket has been merged 
    if (merged_buckets.find(cur_bucket) != merged_buckets.end()) {
      continue;
    }

    SplitBucket(bucket2point, cur_bucket, index_param.max_bucket_size, index_param.min_bucket_size);

    BucketHeap& bucket_neighbor_dist = iter->second;
    MergeBucket(bucket2point, cur_bucket, index_param.max_bucket_size, index_param.min_bucket_size, 
                bucket_neighbor_dist, merged_buckets); 
  }

  /*
  std::cout << std::endl << "=====Split===== " << bucket2point.size() << std::endl;
  */

  // remove merged buckets
  for (auto bucket_id : merged_buckets) {
    bucket2point.erase(bucket_id);
  }

  /*
  std::cout << std::endl << "=====Split===== " << bucket2point.size() << std::endl;

  std::unordered_map<IntCode, std::vector<IntIndex> >::iterator it = bucket2point.begin(); 
  for(; it != bucket2point.end(); it++) {
    std::cout << it->first << " : " << it->second.size() << std::endl;
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConstructBucketsKnnGraph(
    const util::GraphIndexParameter& index_param, 
    Bucket2Point& bucket2point,
    BucketKnnGraph& bucket_knn_graph) {
  std::vector<IntCode> buckets;
  for (auto& item : bucket2point) {
    buckets.push_back(item.first);
  }

  for (int i = 0; i < buckets.size(); i++) {
    IntCode cur_bucket = buckets[i];
    // avoid tmp heap obj
    bucket_knn_graph.insert(BucketKnnGraph::value_type(cur_bucket, BucketHeap(index_param.bucket_neighbor_num)));
    for (int j = 0; j < buckets.size(); j++) {
      if (i != j) {
        // calculate hamming distance
        IntCode xor_result = buckets[i] ^ buckets[j];
        IntCode hamming_dist = 0;
        while (xor_result) {
          xor_result &= xor_result-1;
          hamming_dist++;
        }
        bucket_knn_graph[cur_bucket].Insert(BucketDistancePairItem(buckets[j], hamming_dist)); 
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, 
    int k, 
    std::vector<std::string>& search_result) {
  search_result.clear();
  // Init some points, search from these points
}

} // namespace core 
} // namespace yannsa

#endif
