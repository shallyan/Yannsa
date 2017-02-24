#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include "yannsa/util/point_pair_distance_table.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <iostream>
#include <climits>
#include <algorithm>

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
    // bucket
    typedef std::vector<IntCode> BucketList;
    typedef std::unordered_map<IntCode, IntCode> MergedBucketMap; 
    typedef std::unordered_map<IntCode, std::vector<IntIndex> > Bucket2Point; 
    typedef std::shared_ptr<Bucket2Point> Bucket2PointPtr;
    // bucket knn graph : MAP --> discrete and not too many items
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::unordered_map<IntCode, BucketHeap> BucketKnnGraph;
    typedef std::shared_ptr<BucketKnnGraph> BucketKnnGraphPtr;

    // point
    typedef std::vector<IntIndex> PointList;
    // point knn graph : VECTOR --> continues
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointHeap;
    typedef std::vector<PointHeap> ContinuesPointKnnGraph;
    typedef std::unordered_map<IntIndex, PointHeap> DiscretePointKnnGraph;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoderPtr<PointType>& encoder_ptr); 

    void Clear() {
      index2key_.clear();
      all_point_knn_graph_.clear();
      key_point_knn_graph_.clear();
      bucket2key_point_.clear();
      merged_bucket_map_.clear();
    }

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

  private:
    IntCode CalculateHammingDistance(IntCode x, IntCode y) {
      IntCode hamming_dist = 0;
      IntCode xor_result = x ^ y;
      while (xor_result) {
        xor_result &= xor_result-1;
        hamming_dist++;
      }
      return hamming_dist;
    }

    // half for initial bucket code and distance calculation, half for split
    inline int GetHalfBucketCodeLength() {
      return sizeof(IntCode) * CHAR_BIT / 2; 
    }

    void Encode2Buckets(Bucket2Point& bucket2point, 
                        int point_neighbor_num); 

    void SplitOneBucket(Bucket2Point& bucket2point, 
                     IntCode cur_bucket, 
                     int max_bucket_size, 
                     int min_bucket_size);

    void SplitBuckets(Bucket2Point& bucket2point, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketList& to_split_bucket_list); 

    void MergeOneBucket(Bucket2Point& bucket2point,
                        IntCode cur_bucket,
                        int max_bucket_size,
                        int min_bucket_size,
                        BucketHeap& bucket_neighbor_dist);

    void MergeBuckets(Bucket2Point& bucket2point, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& to_merge_bucket_knn_graph); 

    void BuildBucketsKnnGraph(BucketList& bucket_list,
                              BucketList& neighbor_bucket_list,
                              int bucket_neighbor_num,
                              BucketKnnGraph& bucket_knn_graph); 

    void BuildAllBucketsPointsKnnGraph(Bucket2Point& bucket2point);

    template<typename PointKnnGraphType>
    void BuildPointsKnnGraph(PointList& point_list, PointKnnGraphType& point_knn_graph); 

    void FindBucketKeyPoints(Bucket2Point& bucket2point,
                             int key_point_num);

    void GetSplitMergeBucketsList(Bucket2Point& bucket2point, 
                                  BucketList& bucket_list, 
                                  BucketList& to_split_bucket_list, 
                                  BucketList& to_merge_bucket_list,
                                  int max_bucket_size, 
                                  int min_bucket_size); 
    
  private:
    std::vector<std::string> index2key_;
    ContinuesPointKnnGraph all_point_knn_graph_;
    DiscretePointKnnGraph key_point_knn_graph_;
    Bucket2Point bucket2key_point_;
    MergedBucketMap merged_bucket_map_; 
    util::BaseEncoderPtr<PointType> encoder_ptr_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 
  Clear();

  // record encoder for query search
  encoder_ptr_ = encoder_ptr;

  // encode
  Bucket2PointPtr bucket2point_ptr(new Bucket2Point()); 
  Encode2Buckets(*bucket2point_ptr, index_param.point_neighbor_num); 
  
  std::cout << "encode buckets done" << std::endl;

  // get bucket list
  int max_bucket_size = index_param.max_bucket_size;
  int min_bucket_size = index_param.min_bucket_size;
  BucketList bucket_list, to_split_bucket_list, to_merge_bucket_list;
  GetSplitMergeBucketsList(*bucket2point_ptr, bucket_list, to_split_bucket_list, 
                           to_merge_bucket_list, max_bucket_size, min_bucket_size);

  std::cout << "get split merge buckets done" << std::endl;

  {
    // construct bucket knn graph which need be merged
    BucketKnnGraphPtr to_merge_bucket_knn_graph_ptr(new BucketKnnGraph());
    BuildBucketsKnnGraph(to_merge_bucket_list, bucket_list, 
                         index_param.bucket_neighbor_num, 
                         *to_merge_bucket_knn_graph_ptr);

    std::cout << "build merge buckets knn graph done" << std::endl;

    // split firstly for that too small bucketscan be merged
    SplitBuckets(*bucket2point_ptr, max_bucket_size, min_bucket_size, to_split_bucket_list); 

    std::cout << "split done" << std::endl;

    // merge buckets
    MergeBuckets(*bucket2point_ptr, max_bucket_size, min_bucket_size, *to_merge_bucket_knn_graph_ptr);
  }

  std::cout << "merge done" << std::endl;

  // print bucket num
  /*
  for (auto x : *bucket2point_ptr) {
    std::cout << x.second.size() << "\t"; 
  }
  std::cout << std::endl;
  */

  // build point knn graph
  BuildAllBucketsPointsKnnGraph(*bucket2point_ptr);

  std::cout << "build point knn graph done" << std::endl;

  // count point in degree 
  FindBucketKeyPoints(*bucket2point_ptr, index_param.bucket_key_point_num);

  std::cout << "find bucket key points done" << std::endl;

  // build key point knn graph
  PointList key_point_list;
  for (auto bucket_key_point : bucket2key_point_) {
    key_point_list.insert(key_point_list.end(),
                          bucket_key_point.second.begin(),
                          bucket_key_point.second.end());
  }
  for (auto key_point : key_point_list) {
    key_point_knn_graph_.insert(typename DiscretePointKnnGraph::value_type(
          key_point, 
          PointHeap(index_param.point_neighbor_num)));
  }

  std::cout << "get key points list done, size: " << key_point_list.size() << std::endl;

  BuildPointsKnnGraph(key_point_list, key_point_knn_graph_); 

  std::cout << "build key points knn graph done" << std::endl;

  // build
  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Encode2Buckets(
    Bucket2Point& bucket2point, 
    int point_neighbor_num) { 
  bucket2point.clear();
  typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
  while (iter != this->dataset_ptr_->End()) {
    std::string& key = iter->first;
    PointType& point = iter->second;

    // record point key and index
    IntIndex point_index = index2key_.size();
    index2key_.push_back(key);
    all_point_knn_graph_.push_back(PointHeap(point_neighbor_num));

    // encode point
    IntCode point_code = encoder_ptr_->Encode(point);
    bucket2point[point_code].push_back(point_index);

    iter++;
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetSplitMergeBucketsList(
    Bucket2Point& bucket2point, BucketList& bucket_list, 
    BucketList& to_split_bucket_list, BucketList& to_merge_bucket_list,
    int max_bucket_size, int min_bucket_size) {
  for (auto& item : bucket2point) {
    bucket_list.push_back(item.first);
    if (item.second.size() > max_bucket_size + min_bucket_size) {
      to_split_bucket_list.push_back(item.first);
    }
    if (item.second.size() < min_bucket_size) {
      to_merge_bucket_list.push_back(item.first);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBuckets(
    Bucket2Point& bucket2point, 
    int max_bucket_size,
    int min_bucket_size,
    BucketList& to_split_bucket_list) {
  for (IntCode big_bucket : to_split_bucket_list) {
    SplitOneBucket(bucket2point, big_bucket, max_bucket_size, min_bucket_size);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitOneBucket(
    Bucket2Point& bucket2point, 
    IntCode cur_bucket, 
    int max_bucket_size, 
    int min_bucket_size) {
  int high_bits_num = GetHalfBucketCodeLength();
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeOneBucket(
    Bucket2Point& bucket2point,
    IntCode cur_bucket,
    int max_bucket_size,
    int min_bucket_size,
    BucketHeap& bucket_neighbor_dist) {
  if (bucket2point[cur_bucket].size() < min_bucket_size) {
    bucket_neighbor_dist.Sort();
    auto bucket_neighbor_iter = bucket_neighbor_dist.Begin();
    for (; bucket_neighbor_iter != bucket_neighbor_dist.End(); bucket_neighbor_iter++) {
      IntCode neighbor_bucket = bucket_neighbor_iter->id;
      // neighbor bucket has been merged
      if (merged_bucket_map_.find(neighbor_bucket) != merged_bucket_map_.end()) {
        continue;
      }
      
      // if not exceed split threshold
      if (bucket2point[neighbor_bucket].size() + bucket2point[cur_bucket].size() 
          <= max_bucket_size + min_bucket_size) {
        bucket2point[cur_bucket].insert(bucket2point[cur_bucket].end(),
                                        bucket2point[neighbor_bucket].begin(),
                                        bucket2point[neighbor_bucket].end());
        merged_bucket_map_[neighbor_bucket] = cur_bucket;
      }

      // check
      if (bucket2point[cur_bucket].size() > min_bucket_size) {
        break;
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBuckets(
    Bucket2Point& bucket2point, 
    int max_bucket_size,
    int min_bucket_size,
    BucketKnnGraph& to_merge_bucket_knn_graph) {
  // split and merge
  int split_threshold = static_cast<int>(max_bucket_size + min_bucket_size);
  auto iter = to_merge_bucket_knn_graph.begin(); 
  for (; iter != to_merge_bucket_knn_graph.end(); iter++) {
    IntCode cur_bucket = iter->first;
    // check whether current bucket has been merged 
    if (merged_bucket_map_.find(cur_bucket) != merged_bucket_map_.end()) {
      continue;
    }

    BucketHeap& bucket_neighbor_dist = iter->second;
    MergeOneBucket(bucket2point, cur_bucket, max_bucket_size, min_bucket_size, 
                   bucket_neighbor_dist); 
  }

  // remove merged buckets
  for (auto bucket_pair : merged_bucket_map_) {
    bucket2point.erase(bucket_pair.first);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildBucketsKnnGraph(
    BucketList& bucket_list,
    BucketList& neighbor_bucket_list,
    int bucket_neighbor_num,
    BucketKnnGraph& bucket_knn_graph) {

  for (IntCode cur_bucket : bucket_list) {
    // avoid tmp heap obj
    bucket_knn_graph.insert(BucketKnnGraph::value_type(cur_bucket, BucketHeap(bucket_neighbor_num)));
    for (IntCode neighbor_bucket : neighbor_bucket_list) {
      if (cur_bucket != neighbor_bucket) {
        // calculate hamming distance
        int half_bucket_code_length = GetHalfBucketCodeLength(); 
        IntCode hamming_dist = CalculateHammingDistance(cur_bucket << half_bucket_code_length, 
                                                        neighbor_bucket << half_bucket_code_length);
        bucket_knn_graph[cur_bucket].Insert(BucketDistancePairItem(neighbor_bucket, hamming_dist)); 
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildPointsKnnGraph(
    PointList& point_list,
    PointKnnGraphType& point_knn_graph) {
  DistanceFuncType distance_func;
  for (int i = 0; i < point_list.size(); i++) {
    IntIndex cur_point = point_list[i];
    for (int j = i+1; j < point_list.size(); j++) {
      IntIndex neighbor_point = point_list[j];
      // avoid repeated distance calculation
      DistanceType dist = distance_func(this->dataset_ptr_->Get(index2key_[cur_point]),
                                        this->dataset_ptr_->Get(index2key_[neighbor_point])); 
      point_knn_graph[cur_point].Insert(PointDistancePairItem(neighbor_point, dist));
      point_knn_graph[neighbor_point].Insert(PointDistancePairItem(cur_point, dist));
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsPointsKnnGraph(
    Bucket2Point& bucket2point) {
  auto iter = bucket2point.begin(); 
  for (; iter != bucket2point.end(); iter++) {
    BuildPointsKnnGraph(iter->second, all_point_knn_graph_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints(
    Bucket2Point& bucket2point,
    int key_point_num) {
  // count points in degree
  std::unordered_map<IntIndex, int> point_in_degree_count;
  for (auto& nearest_neighbor : all_point_knn_graph_) {
    auto iter = nearest_neighbor.Begin();
    for (; iter != nearest_neighbor.End(); iter++) {
      auto cur_count_iter = point_in_degree_count.find(iter->id);
      if (cur_count_iter == point_in_degree_count.end()) {
        point_in_degree_count[iter->id] = 1;
      }
      else {
        cur_count_iter->second += 1;
      }
    }
  }

  // find key points
  util::Heap<PointDistancePair<int, int> > key_heap(key_point_num);
  auto bucket_point_iter = bucket2point.begin();
  for (; bucket_point_iter != bucket2point.end(); bucket_point_iter++) {
    IntCode cur_bucket = bucket_point_iter->first;
    std::vector<IntIndex>& point_list = bucket_point_iter->second;

    key_heap.Clear();
    for (auto point_id : point_list) {
      key_heap.Insert(PointDistancePair<int, int>(point_id, -point_in_degree_count[point_id]));
    }
    
    auto key_point_iter = key_heap.Begin();
    for (; key_point_iter != key_heap.End(); key_point_iter++) {
      bucket2key_point_[cur_bucket].push_back(key_point_iter->id); 
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, 
    int k, 
    std::vector<std::string>& search_result) {
  if (!this->have_built_) {
    throw IndexNotBuildError("Graph index hasn't been built!"); 
  }

  search_result.clear();

  // Init some points, search from these points
}

} // namespace core 
} // namespace yannsa

#endif
