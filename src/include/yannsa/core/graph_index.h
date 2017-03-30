#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include "yannsa/util/point_pair.h"
#include "yannsa/util/logging.h"
#include "yannsa/util/lock.h"
#include "yannsa/util/random_generator.h"
#include "yannsa/core/bi_neighbor.h"
#include "yannsa/core/base_index.h"
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>

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
    typedef std::vector<IdList> BucketId2PointList;
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketNeighbor;
    typedef std::vector<BucketNeighbor> BucketKnnGraph;
    typedef std::unordered_map<IntIndex, IntIndex> BucketId2BucketIdMap; 
    typedef std::unordered_map<IntIndex, IdList> BucketId2BucketIdListMap; 
    typedef std::vector<IdList> BucketId2BucketIdList;

    // point
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointNeighbor;
    typedef std::vector<PointNeighbor> ContinuesPointKnnGraph;
    typedef ContinuesBiNeighborInfo<ContinuesPointKnnGraph, DistanceType> PointBiNeighborInfo;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Clear(); 

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoderPtr<PointType>& encoder_ptr); 

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

    void Save(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param,
              util::BaseEncoderPtr<PointType>& encoder_ptr); 

    inline const PointType& GetPoint(IntIndex point_id) {
      return (*this->dataset_ptr_)[point_id];
    }
    
    inline IntIndex PointSize() {
      return all_point_knn_graph_.size();
    }

    inline IntIndex OriginBucketSize() {
      return bucket_id2code_.size();
    }

    inline IntIndex AllBucketSize() {
      return bucket2key_point_.size();
    }

    void LocalitySensitiveHashing(BucketId2PointList& bucket2point_list,
                                  IdList& point2bucket); 

    void MergeOneBucket(BucketId2PointList& bucket2point_list,
                        IntCode bucket_id,
                        int max_bucket_size,
                        int min_bucket_size,
                        BucketNeighbor& bucket_neighbor_heap);

    void MergeBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& bucket_knn_graph); 

    void UpdatePoint2Bucket(IdList& point2bucket); 

    void BuildBucketsKnnGraph(BucketKnnGraph& bucket_knn_graph); 

    void BuildAllBucketsApproximateKnnGraph(BucketId2PointList& bucket2point_list,
                                            int max_bucket_size,
                                            DynamicBitset& refine_point_flag);

    template<typename PointKnnGraphType>
    void BuildPointsKnnGraph(const IdList& point_list, PointKnnGraphType& point_knn_graph); 

    void GetNearestKeyPoint(IntIndex bucket_id, int k, const PointType& point_vec, IdList& nearest_list);

    void SortBucketPointsByInDegree(BucketId2PointList& bucket2point_list);

    void FindBucketKeyPoints(BucketId2PointList& bucket2point_list);

    template <typename PointKnnGraphType>
    void GreedyFindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                              IntIndex start_point_id, PointNeighbor& k_candidates_heap,
                              DynamicBitset& visited_point_flag);

    void LocalitySensitiveSearch(BucketId2PointList& bucket2point_list, 
                                 BucketKnnGraph& bucket_knn_graph);

    void GetBucket2ConnectBucketList(BucketId2PointList& bucket2point_list,
                                     BucketKnnGraph& bucket_knn_graph, 
                                     BucketId2BucketIdList& bucket2connect_list);

    void ConnectBucket2BucketList(BucketId2PointList& bucket2point_list,
                                  IntIndex bucket_id, IdList& neighbor_bucket_list,
                                  ContinuesPointKnnGraph& to_update_candidates,
                                  bool is_full_locality=false);

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);
    int UpdatePointKnn(IntIndex point1, IntIndex point2);

    void RefineGraph(PointBiNeighborInfo& bi_neighbor_info, IdList& point2bucket, 
                     bool is_join_same_bucket, int iteration_num);
    void MarkCornerPoints(DynamicBitset& boundary_point_flag, IdList& point2bucket);

    int JoinBiNeighbor(PointBiNeighborInfo& bi_neighbor_info, IdList& point2bucket, 
                       bool is_join_same_bucket);

  private:
    int point_neighbor_num_;
    int max_point_neighbor_num_;
    int search_point_neighbor_num_;
    int search_start_point_num_;

    ContinuesPointKnnGraph all_point_knn_graph_;

    BucketId2PointList bucket2key_point_;

    std::vector<IntCode> bucket_id2code_;
    std::unordered_map<IntCode, int> bucket_code2id_;

    BucketId2BucketIdMap merged_bucket_map_; 

    util::BaseEncoderPtr<PointType> encoder_ptr_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Clear() {
  this->have_built_ = false;
  // point
  all_point_knn_graph_.clear();

  // bucket
  bucket_code2id_.clear();
  bucket_id2code_.clear();
  bucket2key_point_.clear();
  merged_bucket_map_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Init(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 

  encoder_ptr_ = encoder_ptr;
  point_neighbor_num_ = index_param.point_neighbor_num;
  max_point_neighbor_num_ = std::max(index_param.max_point_neighbor_num,
                                     point_neighbor_num_);
  search_point_neighbor_num_ = std::min(index_param.search_point_neighbor_num,
                                        point_neighbor_num_);
  search_start_point_num_ = index_param.search_start_point_num;

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_knn_graph_ = ContinuesPointKnnGraph(max_point_id, PointNeighbor(point_neighbor_num_));
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Save(
    const std::string file_path) {

  std::ofstream save_file(file_path);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    save_file << this->dataset_ptr_->GetKeyById(point_id) << " ";
    PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];
    size_t effect_size = point_neighbor.effect_size(point_neighbor_num_);
    for (size_t i = 0; i < effect_size; i++) {
      save_file << this->dataset_ptr_->GetKeyById(point_neighbor[i].id) << " "; 
    }
    save_file << std::endl;
  }
  save_file.close();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 

  util::Log("before build");

  if (this->have_built_) {
    throw IndexBuildError("Graph index has already been built!");
  }

  Init(index_param, encoder_ptr);

  // encode and init bukcet
  IdList point2bucket;
  BucketId2PointList bucket2point_list;
  LocalitySensitiveHashing(bucket2point_list, point2bucket);

  // construct bucket knn graph
  BucketKnnGraph bucket_knn_graph(OriginBucketSize(), BucketNeighbor(index_param.bucket_neighbor_num));
  BuildBucketsKnnGraph(bucket_knn_graph);

  // merge buckets
  MergeBuckets(bucket2point_list, index_param.max_bucket_size, index_param.min_bucket_size, bucket_knn_graph);
  UpdatePoint2Bucket(point2bucket);
  std::cout << "merged bucket size: " << merged_bucket_map_.size() << std::endl;

  PointBiNeighborInfo bi_neighbor_info(all_point_knn_graph_);

  // build point knn graph
  util::Log("before build buckets knn graph");
  DynamicBitset refine_point_flag(PointSize(), 0);
  BuildAllBucketsApproximateKnnGraph(bucket2point_list, index_param.max_bucket_size, refine_point_flag);
  util::Log("end build buckets knn graph");

  bi_neighbor_info.Init(refine_point_flag);

  util::Log("before local refine");
  RefineGraph(bi_neighbor_info, point2bucket, true, index_param.local_refine_iter_num);
  util::Log("end local refine");

  // search
  SortBucketPointsByInDegree(bucket2point_list);
  FindBucketKeyPoints(bucket2point_list);
  {
  clock_t s, e;
  s = clock();
  util::Log("before search");
  LocalitySensitiveSearch(bucket2point_list, bucket_knn_graph);
  e = clock();
  std::cout << "locality search: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  /*
  util::Log("before refine");
  {
  clock_t s, e;
  s = clock();
  bi_neighbor_info.Init();
  RefineGraph(point2bi_neighbor, point2bucket, false, index_param.global_refine_iter_num);
  e = clock();
  std::cout << "refine: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }
  */

  // build
  this->have_built_ = true;

  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MarkCornerPoints(
    DynamicBitset& boundary_point_flag, IdList& point2bucket) {

  int cc = 0;
  for (auto x : boundary_point_flag) {
    if (x) cc++;
  }
  std::cout << "boundary point num: " << cc << std::endl;

  // find corner points from boundary points
  #pragma omp parallel for schedule(static)
  for (int point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];
    point_neighbor.remax_size(max_point_neighbor_num_);
    if (!boundary_point_flag[point_id]) {
      continue;
    }

    // corner point neighbors come from more than 2 buckets
    size_t effect_size = point_neighbor.effect_size(point_neighbor_num_);
    IdSet neighbor_bucket_set;
    for (size_t i = 0; i < effect_size; i++) {
      IntIndex neighbor_bucket_id = point2bucket[point_neighbor[i].id];
      if (merged_bucket_map_.find(neighbor_bucket_id) != merged_bucket_map_.end()) {
        neighbor_bucket_id = merged_bucket_map_[neighbor_bucket_id];
      }
      neighbor_bucket_set.insert(neighbor_bucket_id);
    }

    // corner points
    if (neighbor_bucket_set.size() <= 2) {
      boundary_point_flag[point_id] = 0;
    }
    else {
      //point_neighbor.remax_size(max_point_neighbor_num_);
    }
  }

  cc = 0;
  for (auto x : boundary_point_flag) {
    if (x) cc++;
  }
  std::cout << "corner point num: " << cc << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::RefineGraph(
    PointBiNeighborInfo& bi_neighbor_info, IdList& point2bucket, 
    bool is_join_same_bucket, int iteration_num) {

  for (int loop = 0; loop < iteration_num; loop++) {
    clock_t s, e, e1;
    s = clock();
    bi_neighbor_info.Update(search_start_point_num_, 100);
    e = clock();
    std::cout << "refine init: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;

    int update_count = JoinBiNeighbor(bi_neighbor_info, point2bucket, is_join_same_bucket);
    e1 = clock();
    std::cout << "refine update: " << (e1-e)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;

    if (update_count == 0) {
      util::Log("update count stop");
      break;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::JoinBiNeighbor(
    PointBiNeighborInfo& bi_neighbor_info, IdList& point2bucket, 
    bool is_refine_same_bucket) {

  int update_count = 0;
  #pragma omp parallel for schedule(dynamic, 5) default(shared) reduction(+:update_count)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    auto& point_bi_neighbor = bi_neighbor_info.point2bi_neighbor[point_id];

    IdList& new_list = point_bi_neighbor.new_list;
    IdList& reverse_new_list = point_bi_neighbor.reverse_new_list;
    if (new_list.size() == 0 && reverse_new_list.size() == 0) {
      continue;
    }
    new_list.insert(new_list.end(), reverse_new_list.begin(), reverse_new_list.end());

    IdList& old_list = point_bi_neighbor.old_list;
    IdList& reverse_old_list = point_bi_neighbor.reverse_old_list;
    old_list.insert(old_list.end(), reverse_old_list.begin(), reverse_old_list.end());

    // update new 
    int cur_update_count = 0;
    for (size_t i = 0; i < new_list.size(); i++) {
      IntIndex u1 = new_list[i];

      for (size_t j = i+1; j < new_list.size(); j++) {
        IntIndex u2 = new_list[j];
        if (!is_refine_same_bucket && point2bucket[u1] == point2bucket[u2]) {
          continue;
        }
        cur_update_count += UpdatePointKnn(u1, u2);
      }

      for (size_t j = 0; j < old_list.size(); j++) {
        IntIndex u2 = old_list[j];
        if (!is_refine_same_bucket && point2bucket[u1] == point2bucket[u2]) {
          continue;
        }
        cur_update_count += UpdatePointKnn(u1, u2);
      }
    }
    update_count += cur_update_count;
    point_bi_neighbor.is_updated = cur_update_count;
  }
  return update_count;
}

/*
   Here we make use of some strategis proposed by KGraph(NN-Descent)
   Many thanks to KGraph
*/

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointKnn(
    IntIndex point1, IntIndex point2) {

  if (point1 == point2) {
    return 0;
  }

  DistanceType dist = distance_func_(GetPoint(point1), GetPoint(point2));
  return UpdatePointKnn(point1, point2, dist);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointKnn(
    IntIndex point1, IntIndex point2, DistanceType dist) {

  if (point1 == point2) {
    return 0;
  }

  int update_count = 0;
  int update_pos = all_point_knn_graph_[point1].parallel_insert_array(PointDistancePairItem(point2, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;
  update_pos = all_point_knn_graph_[point2].parallel_insert_array(PointDistancePairItem(point1, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;

  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectBucket2BucketList(
    BucketId2PointList& bucket2point_list,
    IntIndex bucket_id, IdList& neighbor_bucket_list,
    ContinuesPointKnnGraph& to_update_candidates,
    bool is_full_locality) {

  DynamicBitset visited_point_flag(PointSize(), 0);
  const IdList& bucket_point_list = bucket2point_list[bucket_id];
  // connect bucket_id to neighbor_bucket_list
  for (IntIndex point_id : bucket_point_list) {
    PointNeighbor& point_update_neighbor = to_update_candidates[point_id];

    // point has beed updated
    if (point_update_neighbor.size() > 0) {
      continue;
    }
    
    const PointType& point_vec = GetPoint(point_id);

    IdList start_point_list;
    // check whether neighor has been updated
    PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];
    size_t point_neighbor_num = point_neighbor.effect_size(point_neighbor_num_);
    if (is_full_locality) {
      for (size_t i = 0; i < point_neighbor_num; i++) {
        IntIndex neighbor_point_id = point_neighbor[i].id;
        PointNeighbor& neighbor_point_update_neighbor = to_update_candidates[neighbor_point_id];
        if (neighbor_point_update_neighbor.size() > 0) {
          size_t search_num = neighbor_point_update_neighbor.effect_size(search_start_point_num_);
          for (size_t j = 0; j < search_num; j++) {
            start_point_list.push_back(neighbor_point_update_neighbor[j].id);
          }
          break;
        }
      }
    }

    // point search from key point in every bucket
    if (start_point_list.empty()) {
      for (IntIndex neighbor_bucket_id : neighbor_bucket_list) {
        GetNearestKeyPoint(neighbor_bucket_id, search_start_point_num_,
                           point_vec, start_point_list);
      }
    }

    visited_point_flag.assign(PointSize(), 0);
    for (IntIndex start_point_id : start_point_list) {
      GreedyFindKnnInGraph(point_vec, all_point_knn_graph_, start_point_id,
                           point_update_neighbor, visited_point_flag);
    }

    // neighbors search from current point's update candidates
    for (size_t i = 0; i < point_neighbor_num; i++) {
      IntIndex neighbor_point_id = point_neighbor[i].id;

      PointNeighbor& neighbor_point_update_neighbor = to_update_candidates[neighbor_point_id];
      // point has beed updated
      if (neighbor_point_update_neighbor.size() > 0) {
        continue;
      }

      const PointType& neighbor_point_vec = GetPoint(neighbor_point_id);
      visited_point_flag.assign(PointSize(), 0);
      size_t effect_size = point_update_neighbor.effect_size(search_start_point_num_);
      for (size_t j = 0; j < effect_size; j++) {
        IntIndex start_point_id = point_update_neighbor[j].id;
        GreedyFindKnnInGraph(neighbor_point_vec, all_point_knn_graph_,
                             start_point_id, neighbor_point_update_neighbor,
                             visited_point_flag);
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetBucket2ConnectBucketList(
    BucketId2PointList& bucket2point_list,
    BucketKnnGraph& bucket_knn_graph,
    BucketId2BucketIdList& bucket2connect_list) {

  util::PointPairSet<IntCode> has_selected_connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    IntIndex from_bucket = bucket_id;
    if (merged_bucket_map_.find(from_bucket) != merged_bucket_map_.end()) {
      from_bucket = merged_bucket_map_[from_bucket];
    }

    auto& neighbor_buckets_heap = bucket_knn_graph[bucket_id];
    for (auto iter = neighbor_buckets_heap.begin();
              iter != neighbor_buckets_heap.end(); iter++) {
      IntIndex target_bucket = iter->id;
      // from bucket and target bucket may be merged
      if (merged_bucket_map_.find(target_bucket) != merged_bucket_map_.end()) {
        target_bucket = merged_bucket_map_[target_bucket];
      }
      if (from_bucket == target_bucket) {
        continue;
      }
      if (has_selected_connect_pair_set.exist(from_bucket,target_bucket)) {
        continue;
      }
      has_selected_connect_pair_set.insert(from_bucket, target_bucket);
      // small bucket search in big bucket
      if (bucket2point_list[from_bucket].size() < bucket2point_list[target_bucket].size()) {
        bucket2connect_list[from_bucket].push_back(target_bucket);
      }
      else {
        bucket2connect_list[target_bucket].push_back(from_bucket);
      }
    }
  }
  std::cout << "connect bucket pair: " << has_selected_connect_pair_set.size() << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveSearch(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph) {

  util::Log("before get bucket search list");
  ContinuesPointKnnGraph to_update_candidates(PointSize(), PointNeighbor(max_point_neighbor_num_));

  BucketId2BucketIdList bucket2connect_list(OriginBucketSize(), IdList());
  GetBucket2ConnectBucketList(bucket2point_list, bucket_knn_graph, bucket2connect_list);

  util::Log("before real search");
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    IdList& to_connect_list = bucket2connect_list[bucket_id];
    if (to_connect_list.size() == 0) {
      continue;
    }

    ConnectBucket2BucketList(bucket2point_list, bucket_id, to_connect_list, to_update_candidates, true);
  }

  #pragma omp parallel for schedule(dynamic, 5)
  for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
    PointNeighbor& candidate_heap = to_update_candidates[point_id];
    if (candidate_heap.size() == 0) {
      continue;
    }

    int update_count = 0;
    for (auto iter = candidate_heap.begin(); iter != candidate_heap.end(); iter++) {
      update_count += UpdatePointKnn(point_id, iter->id, iter->distance);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveHashing(
    BucketId2PointList& bucket2point_list, IdList& point2bucket) {
  // encode
  std::vector<IntCode> point_code_list(PointSize());
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
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
    point2bucket.push_back(bucket_id);
  }

  bucket2key_point_ = BucketId2PointList(OriginBucketSize());
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeOneBucket(
    BucketId2PointList& bucket2point_list, IntCode cur_bucket,
    int max_bucket_size, int min_bucket_size,
    BucketNeighbor& bucket_neighbor_heap) {

  if (merged_bucket_map_.find(cur_bucket) != merged_bucket_map_.end() || 
      bucket2point_list[cur_bucket].size() >= min_bucket_size) {
    return;
  }

  auto bucket_neighbor_iter = bucket_neighbor_heap.begin();
  for (; bucket_neighbor_iter != bucket_neighbor_heap.end(); bucket_neighbor_iter++) {
    // merge cur_bucket into neighbor_buckt
    IntCode neighbor_bucket = bucket_neighbor_iter->id;
    // neighbor bucket may be merged and merged bucket may also be merged
    BucketId2BucketIdMap::iterator merged_iter;
    while ((merged_iter = merged_bucket_map_.find(neighbor_bucket)) != merged_bucket_map_.end()) {
      neighbor_bucket = merged_iter->second;
    }
    // for example: 3->8 8->5, then 5->3
    if (cur_bucket == neighbor_bucket) {
      continue;
    }
    
    // if not exceed split threshold
    if (bucket2point_list[neighbor_bucket].size() + bucket2point_list[cur_bucket].size() 
        <= max_bucket_size) {
        //<= max_bucket_size + min_bucket_size) {
      bucket2point_list[neighbor_bucket].insert(bucket2point_list[neighbor_bucket].end(),
                                                bucket2point_list[cur_bucket].begin(),
                                                bucket2point_list[cur_bucket].end());
      bucket2point_list[cur_bucket].clear();
      merged_bucket_map_[cur_bucket] = neighbor_bucket;
      break;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePoint2Bucket(
    IdList& point2bucket) {
  
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < point2bucket.size(); point_id++) {
    IntIndex bucket_id = point2bucket[point_id];

    BucketId2BucketIdMap::const_iterator merged_iter = merged_bucket_map_.find(bucket_id);
    if (merged_iter != merged_bucket_map_.end()) {
      bucket_id = merged_iter->second; 
      point2bucket[point_id] = bucket_id;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBuckets(
    BucketId2PointList& bucket2point_list, 
    int max_bucket_size, int min_bucket_size,
    BucketKnnGraph& bucket_knn_graph) {

  // merge
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    if (bucket2point_list[bucket_id].size() >= min_bucket_size) {
      continue;
    }
    BucketNeighbor& bucket_neighbor_heap = bucket_knn_graph[bucket_id];
    MergeOneBucket(bucket2point_list, bucket_id, max_bucket_size, min_bucket_size, 
                   bucket_neighbor_heap); 
  }

  // refine merged bucket map
  BucketId2BucketIdMap final_merged_bucket_map; 
  for (auto bucket_pair : merged_bucket_map_) {
    IntCode merged_bucket = bucket_pair.first;
    IntCode target_bucket = bucket_pair.second;
    BucketId2BucketIdMap::const_iterator merged_iter;
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
  #pragma omp parallel for schedule(dynamic, 5)
  for (IntIndex bucket_id_i = 0; bucket_id_i < OriginBucketSize(); bucket_id_i++) {
    for (IntIndex bucket_id_j = 0; bucket_id_j < OriginBucketSize(); bucket_id_j++) {
      if (bucket_id_i == bucket_id_j) {
        continue;
      }

      IntCode bucket_dist = encoder_ptr_->Distance(bucket_id2code_[bucket_id_i], bucket_id2code_[bucket_id_j]);
      bucket_knn_graph[bucket_id_i].insert_heap(BucketDistancePairItem(bucket_id_j, bucket_dist)); 
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
    const PointType& cur_point_vec = GetPoint(cur_point);
    for (int j = i+1; j < point_list.size(); j++) {
      IntIndex neighbor_point = point_list[j];
      // avoid repeated distance calculation
      DistanceType dist = distance_func_(cur_point_vec, GetPoint(neighbor_point));
      point_knn_graph[cur_point].insert_heap(PointDistancePairItem(neighbor_point, dist, false));
      point_knn_graph[neighbor_point].insert_heap(PointDistancePairItem(cur_point, dist, false));
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsApproximateKnnGraph(
    BucketId2PointList& bucket2point_list, int max_bucket_size, 
    DynamicBitset& refine_point_flag) {
  // build bucket knn graph
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    IdList& point_list = bucket2point_list[bucket_id];
    if (point_list.size() < max_bucket_size) {
      BuildPointsKnnGraph(point_list, all_point_knn_graph_);
    }
    else {
      util::IntRandomGenerator int_rand(0, point_list.size()-1);
      for (IntIndex point_id : point_list) {
        refine_point_flag[point_id] = 1;
        IdSet neighbor_set;
        size_t random_neighbor_size = std::min(static_cast<size_t>(point_neighbor_num_), point_list.size()-1);
        while (neighbor_set.size() < random_neighbor_size) {
          IntIndex neighbor_id = point_list[int_rand.Random()];
          if (neighbor_id == point_id || neighbor_set.find(neighbor_id) != neighbor_set.end()) {
            continue;
          }
          neighbor_set.insert(neighbor_id);
          DistanceType dist = distance_func_(GetPoint(neighbor_id), GetPoint(point_id));
          all_point_knn_graph_[point_id].insert_heap(PointDistancePairItem(neighbor_id, dist, true));
        }
      }
    }
  }

  // make heap becomes sorted array
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    all_point_knn_graph_[point_id].sort();
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints(
    BucketId2PointList& bucket2point_list) {

  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    IdList& point_list = bucket2point_list[bucket_id];
    if (point_list.empty()) {
      continue;
    }

    IdList& key_point_list = bucket2key_point_[bucket_id];
    DynamicBitset point_pass_flag(PointSize(), 0);
    for (IntIndex point_id : point_list) {
      if (point_pass_flag[point_id]) {
        continue;
      }

      // select current point
      key_point_list.push_back(point_id);
      point_pass_flag[point_id] = 1;

      // pass current key point's neighbor and neighbor's neighbor
      PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];
      size_t neighbor_effect_size = point_neighbor.effect_size(point_neighbor_num_);
      for (size_t i = 0; i < neighbor_effect_size; i++) {
        IntIndex neighbor_id = point_neighbor[i].id;
        point_pass_flag[neighbor_id] = 1;

        PointNeighbor& neighbor_neighbor = all_point_knn_graph_[neighbor_id];
        size_t neighbor2_effect_size = neighbor_neighbor.effect_size(point_neighbor_num_);
        for (size_t j = 0; j < neighbor2_effect_size; j++) {
          point_pass_flag[neighbor_neighbor[j].id] = 1;
        }
      }
    }
    size_t bucket_key_point_num = 1 + point_list.size() / 25;
    if (key_point_list.size() > bucket_key_point_num) {
      std::random_shuffle(key_point_list.begin(), key_point_list.end());
      key_point_list.resize(bucket_key_point_num); 
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SortBucketPointsByInDegree(
    BucketId2PointList& bucket2point_list) {

  // count points in-degree in [0, point_neighbor_num_)
  // which could represent local density
  std::vector<int> point_in_degree_count(PointSize(), 0);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& neighbor_heap = all_point_knn_graph_[point_id];
    size_t effect_size = neighbor_heap.effect_size(point_neighbor_num_);
    for (size_t i = 0; i < effect_size; i++) {
      IntIndex neighbor_id = neighbor_heap[i].id;
      #pragma omp atomic
      point_in_degree_count[neighbor_id]++;
    }
  }

  typedef util::Heap<PointDistancePair<IntIndex, int> > InDegreeHeap;
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    IdList& point_list = bucket2point_list[bucket_id];
    if (point_list.empty()) {
      continue;
    }

    // sort bucket points by in-degree desc
    InDegreeHeap in_degree_heap(point_list.size());
    for (auto point_id : point_list) {
      in_degree_heap.insert_heap(PointDistancePair<IntIndex, int>(point_id, -point_in_degree_count[point_id]));
    }
    in_degree_heap.sort();
    for (size_t i = 0; i < in_degree_heap.size(); i++) {
      point_list[i] = in_degree_heap[i].id;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetNearestKeyPoint(
    IntIndex bucket_id, int k, const PointType& point_vec, IdList& nearest_list) {

  const IdList& key_point_list = bucket2key_point_[bucket_id];
  if (key_point_list.size() <= k) {
    nearest_list.insert(nearest_list.end(), key_point_list.begin(), key_point_list.end());
    return;
  }

  PointNeighbor k_heap(k);
  for (auto key_point_id : key_point_list) {
    DistanceType key_dist = distance_func_(point_vec, GetPoint(key_point_id));
    k_heap.insert_heap(PointDistancePairItem(key_point_id, key_dist));
  }

  for (auto iter = k_heap.begin(); iter != k_heap.end(); iter++) {
    nearest_list.push_back(iter->id);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, int k, 
    std::vector<std::string>& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }

  IntCode bucket_code = encoder_ptr_->Encode(query);

  // if bucket not exist, use nearest bucket 
  IntIndex bucket_id = -1;
  if (bucket_code2id_.find(bucket_code) != bucket_code2id_.end()) {
    bucket_id = bucket_code2id_[bucket_code];
  }
  else {
    IntCode nearest_bucket_code_dist;
    for (IntIndex exist_bucket_id = 0; exist_bucket_id < bucket_id2code_.size(); exist_bucket_id++) {
      IntCode exist_bucket_code = bucket_id2code_[exist_bucket_id];
      IntCode exist_dist = encoder_ptr_->Distance(exist_bucket_code, bucket_code);
      if (bucket_id == -1 || nearest_bucket_code_dist < exist_dist) {
        bucket_id = exist_bucket_id;
        nearest_bucket_code_dist = exist_dist;
      }
    }
  }

  // bucket may be merged and merged bucket may also be merged
  if (merged_bucket_map_.find(bucket_id) != merged_bucket_map_.end()) {
    bucket_id= merged_bucket_map_[bucket_id];
  }

  // search from start points 
  PointNeighbor result_candidates_heap(k);
  DynamicBitset visited_point_flag(PointSize(), 0);
  for (auto start_point_id : bucket2key_point_[bucket_id]) {
    PointNeighbor tmp_result_candidates_heap(k);
    GreedyFindKnnInGraph(query, all_point_knn_graph_,
                         start_point_id, tmp_result_candidates_heap,
                         visited_point_flag); 
    for (auto iter = tmp_result_candidates_heap.begin(); iter != tmp_result_candidates_heap.end(); iter++) {
      result_candidates_heap.insert_heap(*iter);
    }
  }
  
  result_candidates_heap.sort();
  search_result.clear();
  auto result_iter = result_candidates_heap.begin();
  for (; result_iter != result_candidates_heap.end(); result_iter++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(result_iter->id));
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GreedyFindKnnInGraph(
    const PointType& query, PointKnnGraphType& knn_graph,
    IntIndex start_point_id, PointNeighbor& k_candidates_heap,
    DynamicBitset& visited_point_flag) {

  if (visited_point_flag[start_point_id]) {
    return;
  }

  DistanceType start_dist = distance_func_(GetPoint(start_point_id), query);
  PointDistancePairItem start_point(start_point_id, start_dist);

  PointNeighbor traverse_heap(1);
  traverse_heap.insert_heap(start_point);
  k_candidates_heap.insert_heap(start_point);
  bool is_last_traverse = false;
  while (traverse_heap.size() > 0) {
    // start from current point
    IntIndex cur_nearest_point = traverse_heap[0].id;

    // explore current nearest point's neighbor
    PointNeighbor& point_neighbor = knn_graph[cur_nearest_point];
    // only search search_point_neighbor_num_ neighbors
    size_t  neighbor_search_num = is_last_traverse ? point_neighbor_num_ : search_point_neighbor_num_;
    size_t effect_size = point_neighbor.effect_size(neighbor_search_num);
    for (size_t i = 0; i < effect_size; i++) {
      IntIndex neighbor_id = point_neighbor[i].id;
      if (visited_point_flag[neighbor_id]) {
        continue;
      }

      DistanceType dist = distance_func_(GetPoint(neighbor_id), query);
      PointDistancePairItem point_dist(neighbor_id, dist);

      k_candidates_heap.insert_heap(point_dist);
      traverse_heap.insert_heap(point_dist);
      visited_point_flag[neighbor_id] = 1;
    }

    if (is_last_traverse) {
      break;
    }

    // if nearest neighbor stays the same, enter last traverse
    if (traverse_heap[0].id == cur_nearest_point) {
      is_last_traverse = true;
    }
  }
}

} // namespace core 
} // namespace yannsa

#endif
