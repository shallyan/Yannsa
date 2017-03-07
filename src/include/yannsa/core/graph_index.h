#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include "yannsa/util/point_pair_distance_table.h"
#include "yannsa/util/logging.h"
#include "yannsa/util/random_generator.h"
#include <omp.h>
#include <vector>
#include <bitset>
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
    typedef std::vector<char> DynamicBitset;
    typedef std::vector<IntIndex> IdList;

    // bucket
    typedef std::vector<IdList> BucketId2PointList;
    // bucket knn graph : MAP --> discrete and not too many items
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::vector<BucketHeap> BucketKnnGraph;
    typedef std::unordered_map<IntIndex, IntIndex> BucketId2BucketId; 

    // point
    typedef std::unordered_set<IntIndex> PointSet;
    // point knn graph : VECTOR --> continues
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointHeap;
    typedef std::vector<PointHeap> ContinuesPointKnnGraph;
    typedef std::unordered_map<IntIndex, PointHeap> DiscretePointKnnGraph;

    typedef std::unordered_map<IntIndex, std::unordered_set<IntIndex> > Point2PointSet;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoderPtr<PointType>& encoder_ptr); 

    inline const PointType& GetPoint(IntIndex point_id) {
      return this->dataset_ptr_->Get(index2key_[point_id]);
    }
    
    inline IntIndex PointSize() {
      return index2key_.size();
    }

    inline IntIndex BucketSize() {
      return bucket_id2code_.size();
    }

    void Clear() {
      // point
      index2key_.clear();
      all_point_knn_graph_.clear();
      key_point_knn_graph_.clear();

      // bucket
      bucket_code2id_.clear();
      bucket_id2code_.clear();
      bucket2key_point_.clear();
      merged_bucket_map_.clear();
    }

    void InitPointIndex(int point_neighbor_num);

    void InitBucketIndex() {
      // set all bucket flag be not merged
      merged_bucket_flag_ = DynamicBitset(BucketSize(), 0);
      bucket2key_point_ = BucketId2PointList(BucketSize());
    }

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

    // for test
    void GraphKnn(const std::string& query_key, 
                  int k, 
                  std::vector<std::string>& search_result) {
      search_result.clear();

      int index;
      for (index = 0; index < index2key_.size(); index++) {
        if (index2key_[index] == query_key) {
          break;
        }
      }
      auto& neighbor_heap = all_point_knn_graph_[index];
      neighbor_heap.Sort();
      auto iter = neighbor_heap.Begin();
      for (; iter != neighbor_heap.End(); iter++) {
        search_result.push_back(index2key_[iter->id]);
      }
    }

  private:
    IntCode HammingDistance(IntCode x, IntCode y) {
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

    void Encode2Buckets(BucketId2PointList& bucket2point_list); 

    void SplitMergeBuckets(BucketId2PointList& bucket2point_list, 
                           BucketKnnGraph& bucket_knn_graph,
                           int max_bucket_size, 
                           int min_bucket_size);

    void SplitOneBucket(BucketId2PointList& bucket2point_list, 
                     IntCode cur_bucket, 
                     int max_bucket_size, 
                     int min_bucket_size);

    void SplitBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size);

    void MergeOneBucket(BucketId2PointList& bucket2point_list,
                        IntCode bucket_id,
                        int max_bucket_size,
                        int min_bucket_size,
                        BucketHeap& bucket_neighbor_heap);

    void MergeBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& bucket_knn_graph); 

    void BuildBucketsKnnGraph(BucketKnnGraph& bucket_knn_graph); 

    void BuildAllBucketsPointsKnnGraph(BucketId2PointList& bucket2point_list);

    template<typename PointKnnGraphType>
    void BuildPointsKnnGraph(const IdList& point_list, PointKnnGraphType& point_knn_graph); 

    void FindBucketKeyPoints(BucketId2PointList& bucket2point_list,
                             int key_point_num);

    template <typename PointKnnGraphType>
    void FindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                        IntIndex start_point_id, PointHeap& k_candidates_heap, 
                        DynamicBitset& visited_point_flag);

    template <typename PointKnnGraphType>
    void GreedyFindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                              IntIndex start_point_id, PointHeap& k_candidates_heap,
                              DynamicBitset& visited_point_flag);

    void ConnectBucketPoints(BucketId2PointList& bucket2point_list, 
                             BucketKnnGraph& bucket_knn_graph, int point_neighbor_num); 

    void RefineByExpansion(int iter_num/*, float precision*/) {
      IntIndex max_point_id = all_point_knn_graph_.size();
      for (int loop = 0; loop < iter_num; loop++) {
        // parallel
        // 1 step neighbor
        Point2PointSet point2neighbor;
        for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
          PointHeap& neighbor_heap = all_point_knn_graph_[point_id];
          for (auto iter = neighbor_heap.Begin(); iter != neighbor_heap.End(); iter++) {
            point2neighbor[point_id].insert(iter->id);
          }
        }
          // 2 step neighbor
        Point2PointSet point2two_step_neighbor;
        for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
          PointSet& step1_neighbor = point2neighbor[point_id];
          for (IntIndex step1_point : step1_neighbor) {
            for (IntIndex step2_point : point2neighbor[step1_point]) {
              if (step1_neighbor.find(step2_point) == step1_neighbor.end()) {
                point2two_step_neighbor[point_id].insert(step2_point);
              }
            }
          }
        }

        int update_count = 0;
        // parallel
        for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
          PointSet& step2_neighbor = point2two_step_neighbor[point_id];
          for (IntIndex step2_point : step2_neighbor) {
            DistanceType dist = distance_func_(GetPoint(point_id), GetPoint(step2_point));
            update_count += all_point_knn_graph_[point_id].Insert(PointDistancePairItem(step2_point, dist));
            //update_count += all_point_knn_graph_[neighbor_id].Insert(PointDistancePairItem(point_id, dist));
          }
        }
        if (update_count == 0) {
          break;
        }
      }
    }

    void GetPointReverseNeighbors(Point2PointSet& point_reverse_neighbor) {
      for (IntIndex point_id = 0; point_id < all_point_knn_graph_.size(); point_id++) {
        PointHeap& neighbor_heap = all_point_knn_graph_[point_id];
        for (auto iter = neighbor_heap.Begin(); iter != neighbor_heap.End(); iter++) {
          point_reverse_neighbor[iter->id].insert(point_id);
        }
      }
    }

  private:
    std::vector<std::string> index2key_;
    ContinuesPointKnnGraph all_point_knn_graph_;

    BucketId2PointList bucket2key_point_;
    DiscretePointKnnGraph key_point_knn_graph_;

    std::vector<IntCode> bucket_id2code_;
    std::unordered_map<IntCode, int> bucket_code2id_;

    BucketId2BucketId merged_bucket_map_; 
    DynamicBitset merged_bucket_flag_;

    util::BaseEncoderPtr<PointType> encoder_ptr_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::InitPointIndex(int point_neighbor_num) {
  typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
  while (iter != this->dataset_ptr_->End()) {
    std::string& key = iter->first;

    // record point key and index
    IntIndex point_id = index2key_.size();
    index2key_.push_back(key);
    all_point_knn_graph_.push_back(PointHeap(point_neighbor_num));

    iter++;
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 
  encoder_ptr_ = encoder_ptr;

  InitPointIndex(index_param.point_neighbor_num);

  // encode
  util::Log("before encode");
  BucketId2PointList bucket2point_list;
  Encode2Buckets(bucket2point_list);
  util::Log("encode done");

  InitBucketIndex();

  // construct bucket knn graph
  util::Log("before bucket knn graph");
  BucketKnnGraph bucket_knn_graph(BucketSize(), BucketHeap(index_param.bucket_neighbor_num));
  BuildBucketsKnnGraph(bucket_knn_graph);
  util::Log("end bucket knn graph");

  util::Log("before split merge bucket");
  SplitMergeBuckets(bucket2point_list, bucket_knn_graph, 
                    index_param.max_bucket_size, index_param.min_bucket_size); 
  util::Log("end split merge bucket");

  // build point knn graph
  util::Log("before all point knn graph");
  BuildAllBucketsPointsKnnGraph(bucket2point_list);
  util::Log("end all point knn graph");

  // find key points in bucket
  util::Log("before find bucket key points");
  FindBucketKeyPoints(bucket2point_list, index_param.bucket_key_point_num);
  util::Log("end find bucket key points");

  // join bucket knn graph1 and graph
  util::Log("before connect buckets");
  ConnectBucketPoints(bucket2point_list, bucket_knn_graph, index_param.point_neighbor_num); 
  util::Log("end connect buckets");

  util::Log("start refine ");
  //RefineByExpansion(1);
  util::Log("end refine ");

  /*
  // build key point knn graph
  IdList key_point_list;
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
  */

  // build
  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectBucketPoints(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    int point_neighbor_num) {

  // get each bucket connect bucket
  util::PointPairTable<IntCode> connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    auto& neighbor_buckets_heap = bucket_knn_graph[bucket_id];
    for (auto iter = neighbor_buckets_heap.Begin();
              iter != neighbor_buckets_heap.End(); iter++) {
      IntIndex from_bucket = bucket_id;
      IntIndex target_bucket = iter->id;
      // from bucket and target bucket may be merged
      if (merged_bucket_flag_[from_bucket]) {
        from_bucket = merged_bucket_map_[from_bucket];
      }
      if (merged_bucket_flag_[target_bucket]) {
        target_bucket = merged_bucket_map_[target_bucket];
      }
      if (from_bucket != target_bucket) {
        connect_pair_set.Insert(from_bucket, target_bucket);
      }
    }
  }
  
  util::PointPairList<IntCode> connect_pair_list(connect_pair_set.Begin(), connect_pair_set.End());
  DynamicBitset connect_pair_flag(connect_pair_list.size(), 0);
  std::vector<util::PointPairList<IntCode> > batch_connect_pairs;
  while (true) {
    util::PointPairList<IntCode> one_batch_pair_list;
    DynamicBitset selected_bucket_flag(BucketSize(), 0); 
    for (int con_pair_id = 0; con_pair_id < connect_pair_flag.size(); con_pair_id++) {
      if (!connect_pair_flag[con_pair_id]) {
        auto& cur_pair = connect_pair_list[con_pair_id];
        if (!selected_bucket_flag[cur_pair.first] && !selected_bucket_flag[cur_pair.second]) {
          selected_bucket_flag[cur_pair.first] = 1;
          selected_bucket_flag[cur_pair.second] = 1;
          connect_pair_flag[con_pair_id] = 1;
          one_batch_pair_list.push_back(cur_pair);
        }
      }
    }
    if (one_batch_pair_list.size() == 0) {
      break;
    }
    batch_connect_pairs.push_back(one_batch_pair_list);
  }

  // expensive, so only calculate once
  Point2PointSet point_reverse_neighbor;
  GetPointReverseNeighbors(point_reverse_neighbor); 

  util::Log("start lsh search");
  for (util::PointPairList<IntCode>& one_batch_pair_list : batch_connect_pairs) {
    ContinuesPointKnnGraph to_update_candidates(all_point_knn_graph_.size(), PointHeap(point_neighbor_num));

    //#pragma omp parallel for schedule(dynamic, 5)
    for (int pair_id = 0; pair_id < one_batch_pair_list.size(); pair_id++) {
      IntCode bucket_id = one_batch_pair_list[pair_id].first;
      IntCode neighbor_bucket_id = one_batch_pair_list[pair_id].second;

      const IdList& start_point_list = bucket2key_point_[neighbor_bucket_id];
      const IdList& bucket_point_list = bucket2point_list[bucket_id];

      DynamicBitset point_has_searched_flag(PointSize(), 0);
      DynamicBitset bucket_point_flag(PointSize(), 0);
      for (IntIndex point_id : bucket_point_list) {
        bucket_point_flag[point_id] = 1;
      }

      for (IntIndex point_id : bucket_point_list) {
        if (point_has_searched_flag[point_id]) {
          continue;
        }

        // current point search start from key point
        /*
        if (start_point_id == -1) {
          IntIndex near_id;
          DistanceType near_dist = 100000.0;
          for (auto start_point_id : start_point_list) {
            DistanceType ddd = distance_func_(GetPoint(start_point_id), GetPoint(point_id));
            if (ddd < near_dist) {
              near_dist = ddd;
              near_id = start_point_id;
            }
          }
          start_point_id = near_id;
        }
        */
        {
          DynamicBitset visited_point_flag(bucket_point_flag);
          IntIndex start_point_id = start_point_list[0];
          FindKnnInGraph(GetPoint(point_id), all_point_knn_graph_,
                               start_point_id, to_update_candidates[point_id],
                               visited_point_flag); 
        }

        point_has_searched_flag[point_id] = 1;
        to_update_candidates[point_id].Sort();

        PointSet& reverse_neighbor = point_reverse_neighbor[point_id];
        for (auto reverse_point_id : reverse_neighbor) {
          if (point_has_searched_flag[reverse_point_id] || !bucket_point_flag[reverse_point_id]) {
            continue;
          }
          DynamicBitset visited_point_flag(bucket_point_flag);
          IntIndex start_point_id = to_update_candidates[point_id].Front().id;
          GreedyFindKnnInGraph(GetPoint(reverse_point_id), all_point_knn_graph_,
                               start_point_id, to_update_candidates[reverse_point_id],
                               visited_point_flag);
          point_has_searched_flag[reverse_point_id] = 1;
        }
      }
    }

    // update
    //#pragma omp parallel for schedule(dynamic, 1000)
    for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
      PointHeap& candidate_heap = to_update_candidates[point_id];
      for (auto iter = candidate_heap.Begin(); iter != candidate_heap.End(); iter++) {
        all_point_knn_graph_[point_id].Insert(*iter);
        all_point_knn_graph_[iter->id].Insert(PointDistancePairItem(point_id, iter->distance));
      }
    }
  }

  util::Log("end lsh search");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Encode2Buckets(
    BucketId2PointList& bucket2point_list) { 
  // encode
  std::vector<IntCode> point_code_list(index2key_.size());
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < index2key_.size(); point_id++) {
    IntCode point_code = encoder_ptr_->Encode(GetPoint(point_id));
    point_code_list[point_id] = point_code;
  }

  for (IntIndex point_id = 0; point_id < point_code_list.size(); point_id++) {
    IntCode point_code = point_code_list[point_id];
    // create new bucket if not exist
    if (bucket_code2id_.find(point_code) == bucket_code2id_.end()) {
      IntIndex new_bucket_id = bucket_id2code_.size();
      bucket_id2code_.push_back(point_code);
      bucket_code2id_[point_code] = new_bucket_id; 
      bucket2point_list.push_back(IdList());
    }

    IntIndex bucket_id = bucket_code2id_[point_code];
    bucket2point_list[bucket_id].push_back(point_id);
  }
}


template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitMergeBuckets(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    int max_bucket_size, int min_bucket_size) {

  // split firstly for that too small bucketscan be merged
  //SplitBuckets(bucket2point_list, max_bucket_size, min_bucket_size, to_split_bucket_list); 

  int before_size = bucket2point_list.size();
  std::cout << "before merge size " << before_size << std::endl;
  // merge buckets
  MergeBuckets(bucket2point_list, max_bucket_size, min_bucket_size, bucket_knn_graph);

  std::cout << "after merge size " << before_size - merged_bucket_map_.size() << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBuckets(
    BucketId2PointList& bucket2point_list, 
    int max_bucket_size,
    int min_bucket_size) {

  /*
  IdList& to_split_bucket_list;
  for (auto& item : bucket2point_list) {
    if (item.second.size() > max_bucket_size + min_bucket_size) {
      to_split_bucket_list.push_back(item.first);
    }
  }

  for (IntCode big_bucket : to_split_bucket_list) {
    SplitOneBucket(bucket2point_list, big_bucket, max_bucket_size, min_bucket_size);
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitOneBucket(
    BucketId2PointList& bucket2point_list, 
    IntCode cur_bucket, 
    int max_bucket_size, 
    int min_bucket_size) {
  /*
  int high_bits_num = GetHalfBucketCodeLength();
  int new_bucket_count = 0;
  while (bucket2point_list[cur_bucket].size() > max_bucket_size + min_bucket_size) {
    new_bucket_count++;
    // avoid overflow
    if (new_bucket_count > (1 << (high_bits_num - 1)) - 1) {
        break;
    }

    IntCode new_bucket = cur_bucket + (new_bucket_count << high_bits_num);
    auto bucket_begin_iter = bucket2point_list[cur_bucket].end() - max_bucket_size;
    auto bucket_end_iter = bucket2point_list[cur_bucket].end();
    bucket2point_list[new_bucket] = std::vector<IntIndex>(bucket_begin_iter, bucket_end_iter);
    bucket2point_list[cur_bucket].erase(bucket_begin_iter, bucket_end_iter); 
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeOneBucket(
    BucketId2PointList& bucket2point_list, IntCode cur_bucket,
    int max_bucket_size, int min_bucket_size,
    BucketHeap& bucket_neighbor_heap) {

  if (merged_bucket_map_.find(cur_bucket) != merged_bucket_map_.end() || 
      bucket2point_list[cur_bucket].size() >= min_bucket_size) {
    return;
  }

  bucket_neighbor_heap.Sort();
  auto bucket_neighbor_iter = bucket_neighbor_heap.Begin();
  for (; bucket_neighbor_iter != bucket_neighbor_heap.End(); bucket_neighbor_iter++) {
    // merge cur_bucket into neighbor_buckt
    IntCode neighbor_bucket = bucket_neighbor_iter->id;
    // neighbor bucket may be merged and merged bucket may also be merged
    BucketId2BucketId::iterator merged_iter;
    while ((merged_iter = merged_bucket_map_.find(neighbor_bucket)) != merged_bucket_map_.end()) {
      neighbor_bucket = merged_iter->second;
    }
    // for example: 3->8 8->5, then 5->3
    if (cur_bucket == neighbor_bucket) {
      continue;
    }
    
    // if not exceed split threshold
    if (bucket2point_list[neighbor_bucket].size() + bucket2point_list[cur_bucket].size() 
        <= max_bucket_size + min_bucket_size) {
      bucket2point_list[neighbor_bucket].insert(bucket2point_list[neighbor_bucket].end(),
                                                bucket2point_list[cur_bucket].begin(),
                                                bucket2point_list[cur_bucket].end());
      bucket2point_list[cur_bucket].clear();
      merged_bucket_map_[cur_bucket] = neighbor_bucket;
      merged_bucket_flag_[cur_bucket] = 1;
      break;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBuckets(
    BucketId2PointList& bucket2point_list, 
    int max_bucket_size,
    int min_bucket_size,
    BucketKnnGraph& bucket_knn_graph) {

  // merge
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    if (bucket2point_list[bucket_id].size() >= min_bucket_size) {
      continue;
    }
    BucketHeap& bucket_neighbor_heap = bucket_knn_graph[bucket_id];
    MergeOneBucket(bucket2point_list, bucket_id, max_bucket_size, min_bucket_size, 
                   bucket_neighbor_heap); 
  }

  // refine merged bucket map
  BucketId2BucketId final_merged_bucket_map; 
  for (auto bucket_pair : merged_bucket_map_) {
    IntCode merged_bucket = bucket_pair.first;
    IntCode target_bucket = bucket_pair.second;
    BucketId2BucketId::const_iterator merged_iter;
    while ((merged_iter = merged_bucket_map_.find(target_bucket)) != merged_bucket_map_.end()) {
      target_bucket = merged_iter->second;
    }
    final_merged_bucket_map[merged_bucket] = target_bucket;
  }
  merged_bucket_map_.swap(final_merged_bucket_map);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildBucketsKnnGraph(
    BucketKnnGraph& bucket_knn_graph) {

  // repeat calculate bucket pair distance for parallel 
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id_i = 0; bucket_id_i < BucketSize(); bucket_id_i++) {
    for (IntIndex bucket_id_j = 0; bucket_id_j < BucketSize(); bucket_id_j++) {
      if (bucket_id_i == bucket_id_j) {
        continue;
      }

      int half_bucket_code_length = GetHalfBucketCodeLength(); 
      IntCode hamming_dist = HammingDistance(bucket_id_i << half_bucket_code_length, 
                                             bucket_id_j << half_bucket_code_length);
      bucket_knn_graph[bucket_id_i].Insert(BucketDistancePairItem(bucket_id_j, hamming_dist)); 
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildPointsKnnGraph(
    const IdList& point_list,
    PointKnnGraphType& point_knn_graph) {
  for (int i = 0; i < point_list.size(); i++) {
    IntIndex cur_point = point_list[i];
    for (int j = i+1; j < point_list.size(); j++) {
      IntIndex neighbor_point = point_list[j];
      // avoid repeated distance calculation
      DistanceType dist = distance_func_(GetPoint(cur_point), GetPoint(neighbor_point));
      point_knn_graph[cur_point].Insert(PointDistancePairItem(neighbor_point, dist));
      point_knn_graph[neighbor_point].Insert(PointDistancePairItem(cur_point, dist));
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsPointsKnnGraph(
    BucketId2PointList& bucket2point_list) {
  #pragma omp parallel for schedule(dynamic, 5)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    BuildPointsKnnGraph(bucket2point_list[bucket_id], all_point_knn_graph_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints(
    BucketId2PointList& bucket2point_list,
    int key_point_num) {
  // count points in degree
  std::vector<int> point_in_degree_count(all_point_knn_graph_.size(), 0);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < all_point_knn_graph_.size(); point_id++) {
    PointHeap& nearest_neighbor = all_point_knn_graph_[point_id];
    auto iter = nearest_neighbor.Begin();
    for (; iter != nearest_neighbor.End(); iter++) {
      #pragma omp atomic
      point_in_degree_count[iter->id]++;
    }
  }

  // find key points
  typedef util::Heap<PointDistancePair<IntIndex, int> > InDegreeHeap;
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    const IdList& point_list = bucket2point_list[bucket_id];
    if (point_list.empty()) {
      continue;
    }

    InDegreeHeap min_in_degree_heap(0);
    for (auto point_id : point_list) {
      // won't repeat
      min_in_degree_heap.Push(PointDistancePair<IntIndex, int>(point_id, point_in_degree_count[point_id]));
    }
    min_in_degree_heap.Sort();

    DynamicBitset point_pass_flag(PointSize(), 0);
    for (auto point_iter = min_in_degree_heap.Begin(); 
              point_iter != min_in_degree_heap.End(); point_iter++) {
      if (point_pass_flag[point_iter->id]) {
        continue;
      }
      point_pass_flag[point_iter->id] = 1;

      bucket2key_point_[bucket_id].push_back(point_iter->id);
      // find enough num key points
      if (bucket2key_point_[bucket_id].size() > key_point_num) {
        break;
      }

      // pass current key point's neighbor
      PointHeap& nearest_neighbor = all_point_knn_graph_[point_iter->id];
      for (auto neighbor_iter = nearest_neighbor.Begin();
                neighbor_iter != nearest_neighbor.End(); neighbor_iter++) {
        point_pass_flag[neighbor_iter->id] = 1;
      }
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

  /*
  IdList start_points;
  IntCode bucket_code = encoder_ptr_->Encode(query);

  // bucket may be merged and merged bucket may also be merged
  Bucket2Bucket::iterator merged_iter;
  while ((merged_iter = merged_bucket_map_.find(bucket_code)) != merged_bucket_map_.end()) {
    bucket_code = merged_iter->second;
  }

  // if bucket not exist, use near bucket (now bucket must exist)
  for (int step = 0; ; step++) {
    IntCode left_bucket = bucket_code - step;
    IntCode right_bucket = bucket_code + step;
    if (left_bucket >= 0) {
      auto find_iter = bucket2key_point_.find(left_bucket);
      if (find_iter != bucket2key_point_.end()){
        start_points = find_iter->second;
        break;
      }
      else {
        std::cout << "bucket not exist" << std::endl;
      }
    }

    // won't overflow
    if (right_bucket != bucket_code) {
      auto find_iter = bucket2key_point_.find(right_bucket);
      if (find_iter != bucket2key_point_.end()){
        start_points = find_iter->second;
        break;
      }
    }
  }

  // search from start points in key point knn graph
  std::unordered_set<IntIndex> has_visited_point;
  PointHeap key_candidates_heap(0);
  for (const auto& one_start_point : start_points) {
    if (has_visited_point.find(one_start_point) == has_visited_point.end()) {
      DistanceType dist = distance_func_(GetPoint(one_start_point), query);
      PointDistancePairItem point_dist(one_start_point, dist);
      has_visited_point.insert(one_start_point);
    
      key_candidates_heap.Insert(point_dist);
  */
      /*
      PointHeap nearest_heap(100);
      FindKnnInGraph(query, key_point_knn_graph_, point_dist, nearest_heap, has_visited_point); 
      for (auto ii = nearest_heap.Begin(); ii != nearest_heap.End(); ii++) {
        key_candidates_heap.Insert(*ii);
      }
      */
  /*
    }
  }

  // search from key points in all point knn graph
  //has_visited_point.clear();
  PointHeap result_candidates_heap(k);
  auto key_iter = key_candidates_heap.Begin();
  for (; key_iter != key_candidates_heap.End(); key_iter++) {
    PointHeap k_candidates_heap(k);
    //FindKnnInGraph(query, all_point_knn_graph_, *key_iter, k_candidates_heap, has_visited_point, PointSet()); 
    for (auto ii = k_candidates_heap.Begin(); ii != k_candidates_heap.End(); ii++) {
      result_candidates_heap.Insert(*ii);
    }
  }
  
  search_result.clear();
  result_candidates_heap.Sort();
  auto result_iter = result_candidates_heap.Begin();
  for (; result_iter != result_candidates_heap.End(); result_iter++) {
    search_result.push_back(index2key_[result_iter->id]);
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindKnnInGraph(
    const PointType& query, PointKnnGraphType& knn_graph,
    IntIndex start_point_id, PointHeap& k_candidates_heap,
    DynamicBitset& visited_point_flag) {

  if (visited_point_flag[start_point_id]) {
    return;
  }

  DistanceType start_dist = distance_func_(GetPoint(start_point_id), query);
  PointDistancePairItem start_point(start_point_id, start_dist);

  PointHeap traverse_heap(0);
  traverse_heap.Insert(start_point);
  k_candidates_heap.Insert(start_point);
  while (traverse_heap.Size() > 0) {
    // start from current nearest point
    IntIndex cur_nearest_point = traverse_heap.Front().id;
    traverse_heap.Pop();

    // explore current nearest point's neighbor
    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.Begin();
    for (; neighbor_iter != point_neighbor.End(); neighbor_iter++) {
      if (visited_point_flag[neighbor_iter->id]) {
        continue;
      }

      DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
      PointDistancePairItem point_dist(neighbor_iter->id, dist);

      int update_count = k_candidates_heap.Insert(point_dist);
      if (update_count > 0) {
        traverse_heap.Insert(point_dist);
      }
      visited_point_flag[neighbor_iter->id] = 1;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GreedyFindKnnInGraph(
    const PointType& query, PointKnnGraphType& knn_graph,
    IntIndex start_point_id, PointHeap& k_candidates_heap,
    DynamicBitset& visited_point_flag) {

  if (visited_point_flag[start_point_id]) {
    return;
  }

  DistanceType start_dist = distance_func_(GetPoint(start_point_id), query);
  PointDistancePairItem start_point(start_point_id, start_dist);

  PointHeap traverse_heap(1);
  traverse_heap.Insert(start_point);
  k_candidates_heap.Insert(start_point);
  while (traverse_heap.Size() > 0) {
    // start from current nearest point
    IntIndex cur_nearest_point = traverse_heap.Front().id;
    //traverse_heap.Pop();

    // explore current nearest point's neighbor
    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.Begin();
    for (; neighbor_iter != point_neighbor.End(); neighbor_iter++) {
      if (visited_point_flag[neighbor_iter->id]) {
        continue;
      }

      DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
      PointDistancePairItem point_dist(neighbor_iter->id, dist);

      int update_count = k_candidates_heap.Insert(point_dist);
      if (update_count > 0) {
        traverse_heap.Insert(point_dist);
      }
      visited_point_flag[neighbor_iter->id] = 1;
    }
    if (traverse_heap.Front().id == cur_nearest_point) {
      break;
    }
  }
}

} // namespace core 
} // namespace yannsa

#endif
