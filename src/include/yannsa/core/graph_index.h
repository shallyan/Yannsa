#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include "yannsa/util/point_pair.h"
#include "yannsa/util/logging.h"
#include "yannsa/util/random_generator.h"
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
    typedef std::unordered_set<IntIndex> IdSet;

    // bucket
    typedef std::vector<IdList> BucketId2PointList;
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketNeighbor;
    typedef std::vector<BucketNeighbor> BucketKnnGraph;
    typedef std::unordered_map<IntIndex, IntIndex> BucketId2BucketIdMap; 
    typedef std::unordered_map<IntIndex, IdList> BucketId2BucketIdListMap; 
    typedef std::vector<IdList> BucketId2BucketIdList;

    // point
    typedef std::vector<IdList> PointId2PointList;
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointNeighbor;
    typedef std::vector<PointNeighbor> ContinuesPointKnnGraph;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Clear(); 

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoderPtr<PointType>& encoder_ptr); 

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

    // for test
    void GraphKnn(IntIndex query_id, 
                  int k, 
                  std::vector<std::string>& search_result) {
      search_result.clear();

      // copy to not change heap
      auto& neighbor_heap = all_point_knn_graph_[query_id];
      auto iter = neighbor_heap.begin();
      for (; iter != neighbor_heap.end(); iter++) {
        search_result.push_back(this->dataset_ptr_->GetKeyById(iter->id));
        if (search_result.size() >= k) {
          break;
        }
      }
    }

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
                                  IdList& point_id2bucket_id); 

    void SplitMergeBuckets(BucketId2PointList& bucket2point_list, 
                           BucketKnnGraph& bucket_knn_graph,
                           int max_bucket_size, 
                           int min_bucket_size);

    void SplitOneBucket(BucketId2PointList& bucket2point_list, 
                     IntCode bucket_id, 
                     int max_bucket_size, 
                     int min_bucket_size);

    void SplitBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size);

    void MergeOneBucket(BucketId2PointList& bucket2point_list,
                        IntCode bucket_id,
                        int max_bucket_size,
                        int min_bucket_size,
                        BucketNeighbor& bucket_neighbor_heap);

    void MergeBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& bucket_knn_graph); 

    void BuildBucketsKnnGraph(BucketKnnGraph& bucket_knn_graph); 

    void BuildAllBucketsPointsKnnGraph(BucketId2PointList& bucket2point_list);

    template<typename PointKnnGraphType>
    void BuildPointsKnnGraph(const IdList& point_list, PointKnnGraphType& point_knn_graph); 

    IntIndex GetNearestKeyPoint(IntIndex bucket_id, const PointType& point_vec);

    void SortBucketPointsByInDegree(BucketId2PointList& bucket2point_list);

    void FindBucketKeyPoints(BucketId2PointList& bucket2point_list,
                             int key_point_num);

    template <typename PointKnnGraphType>
    void GreedyFindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                              IntIndex start_point_id, PointNeighbor& k_candidates_heap,
                              DynamicBitset& visited_point_flag);

    void LocalitySensitiveSearch(BucketId2PointList& bucket2point_list, 
                                 BucketKnnGraph& bucket_knn_graph,
                                 DynamicBitset& boundary_point_flag);

    void GetBucket2ConnectBucketList(BucketKnnGraph& bucket_knn_graph, 
                                     BucketId2BucketIdList& bucket2connect_list);

    void ConnectSplitedBuckets(BucketId2PointList& bucket2point_list,
                               ContinuesPointKnnGraph& to_update_candidates);

    void ConnectBucket2BucketList(BucketId2PointList& bucket2point_list, 
                                  IntIndex bucket_id, IdList& neighbor_bucket_list,
                                  ContinuesPointKnnGraph& to_update_candidates);

    void ConnectAllBuckets2BucketList(BucketId2PointList& bucket2point_list,
                                      BucketKnnGraph& bucket_knn_graph,
                                      ContinuesPointKnnGraph& to_update_candidates,
                                      DynamicBitset& boundary_point_flag); 

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);

    void LocalitySensitiveRefine(IdList& point_id2bucket_id, DynamicBitset& boundary_point_flag, int iteration_num); 

    void GetPointReverseNeighbors(PointId2PointList& point2point_list, 
                                  PointId2PointList& point2reverse_neighbors); 
  private:
    int point_neighbor_num_;
    int max_point_neighbor_num_;
    int search_point_neighbor_num_;

    ContinuesPointKnnGraph all_point_knn_graph_;

    BucketId2PointList bucket2key_point_;

    std::vector<IntCode> bucket_id2code_;
    std::unordered_map<IntCode, int> bucket_code2id_;

    BucketId2BucketIdMap merged_bucket_map_; 

    BucketId2BucketIdListMap splited_bucket_map_; 

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
  splited_bucket_map_.clear();
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

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_knn_graph_ = ContinuesPointKnnGraph(max_point_id, PointNeighbor(max_point_neighbor_num_));
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
  BucketId2PointList bucket2point_list;
  IdList point_id2bucket_id;
  LocalitySensitiveHashing(bucket2point_list, point_id2bucket_id);

  // construct bucket knn graph
  BucketKnnGraph bucket_knn_graph(OriginBucketSize(), BucketNeighbor(index_param.bucket_neighbor_num));
  BuildBucketsKnnGraph(bucket_knn_graph);

  {
  clock_t s, e;
  s = clock();
  SplitMergeBuckets(bucket2point_list, bucket_knn_graph, 
                    index_param.max_bucket_size, index_param.min_bucket_size); 
  e = clock();
  std::cout << "split and merge: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // build point knn graph
  {
  clock_t s, e;
  s = clock();
  BuildAllBucketsPointsKnnGraph(bucket2point_list);
  e = clock();
  std::cout << "build all point knn graph: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // find key points in bucket
  {
  clock_t s, e;
  s = clock();
  SortBucketPointsByInDegree(bucket2point_list);
  FindBucketKeyPoints(bucket2point_list, index_param.bucket_key_point_num);
  e = clock();
  std::cout << "sort in degree: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // join bucket knn graph graph
  DynamicBitset boundary_point_flag(PointSize(), 0);
  LocalitySensitiveSearch(bucket2point_list, bucket_knn_graph, boundary_point_flag);

  {
  clock_t s, e;
  s = clock();
  LocalitySensitiveRefine(point_id2bucket_id, boundary_point_flag, index_param.refine_iter_num);
  e = clock();
  std::cout << "refine: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // build
  this->have_built_ = true;

  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetPointReverseNeighbors(
    PointId2PointList& point2point_list, 
    PointId2PointList& point2reverse_neighbors) {

  for (IntIndex point_id = 0; point_id < point2point_list.size(); point_id++) {
    IdList& neighbor_point_list = point2point_list[point_id];
    for (auto neighbor_point_id : neighbor_point_list) {
      point2reverse_neighbors[neighbor_point_id].push_back(point_id);
    }
  }
  // won't repeat
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveRefine(
    IdList& point_id2bucket_id, DynamicBitset& boundary_point_flag, int iteration_num) {

  int max_point_id = PointSize();

  // find corner points from boundary points
  #pragma omp parallel for schedule(static)
  for (int point_id = 0; point_id < max_point_id; point_id++) {
    PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];

    if (!boundary_point_flag[point_id]) {
      point_neighbor.resize(point_neighbor_num_);
      continue;
    }

    // corner point neighbors come from more than 2 buckets
    size_t effect_size = point_neighbor.effect_size(point_neighbor_num_);
    IdSet neighbor_bucket_set;
    for (size_t i = 0; i < effect_size; i++) {
      neighbor_bucket_set.insert(point_id2bucket_id[point_neighbor[i].id]);
    }

    // only keep corner points
    if (neighbor_bucket_set.size() <= 2) {
      boundary_point_flag[point_id] = 0;
      point_neighbor.resize(point_neighbor_num_);
    }
  }

  PointId2PointList point2old(max_point_id), point2new(max_point_id),
                    point2old_reverse(max_point_id), point2new_reverse(max_point_id);
  for (int loop = 0; loop < iteration_num; loop++) {
    // init
    #pragma omp parallel for schedule(static)
    for (IntIndex cur_point_id = 0; cur_point_id < max_point_id; cur_point_id++) {
      point2old[cur_point_id].clear();
      point2old_reverse[cur_point_id].clear();
      point2new[cur_point_id].clear();
      point2new_reverse[cur_point_id].clear();

      PointNeighbor& neighbor_heap = all_point_knn_graph_[cur_point_id];
      for (auto iter = neighbor_heap.begin(); iter != neighbor_heap.end(); iter++) {
        if (iter->flag) {
          point2new[cur_point_id].push_back(iter->id);
          iter->flag = false;
        }
        else {
          point2old[cur_point_id].push_back(iter->id);
        }
      }
    }

    // reverse
    GetPointReverseNeighbors(point2old, point2old_reverse);
    GetPointReverseNeighbors(point2new, point2new_reverse);

    int update_count = 0;
    #pragma omp parallel for schedule(dynamic, 20) default(shared) reduction(+:update_count)
    for (IntIndex cur_point_id = 0; cur_point_id < max_point_id; cur_point_id++) {
      /*
      if (!boundary_point_flag[cur_point_id]) {
        continue;
      }
      */

      IdList& new_list = point2new[cur_point_id];
      IdList& new_reverse_list = point2new_reverse[cur_point_id];
      if (new_list.size() == 0 && new_reverse_list.size() == 0) {
        continue;
      }
      new_list.insert(new_list.end(), new_reverse_list.begin(), new_reverse_list.end());

      IdList& old_list = point2old[cur_point_id];
      IdList& old_reverse_list = point2old_reverse[cur_point_id];
      old_list.insert(old_list.end(), old_reverse_list.begin(), old_reverse_list.end());

      // unique
      IdSet old_set(old_list.begin(), old_list.end());
      IdSet new_set(new_list.begin(), new_list.end());
      IdList uniq_old_list(old_set.begin(), old_set.end());
      IdList uniq_new_list(new_set.begin(), new_set.end());

      // update new 
      for (IntIndex u1 : uniq_new_list) {
        const PointType& point_vec = GetPoint(u1);
        for (IntIndex u2 : uniq_new_list) {
          if (u1 < u2) {
            DistanceType dist = distance_func_(point_vec, GetPoint(u2));
            update_count += UpdatePointKnn(u1, u2, dist);
          }
        }
        for (IntIndex u2 : uniq_old_list) {
          DistanceType dist = distance_func_(point_vec, GetPoint(u2));
          update_count += UpdatePointKnn(u1, u2, dist);
        }
      }
    }
    if (update_count == 0) {
      util::Log("update count stop");
      break;
    }
  }
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
    ContinuesPointKnnGraph& to_update_candidates) {

  const IdList& bucket_point_list = bucket2point_list[bucket_id];
  // connect bucket_id to neighbor_bucket_list
  for (IntIndex point_id : bucket_point_list) {
    PointNeighbor& point_update_neighbor = to_update_candidates[point_id];

    // point has beed updated
    if (point_update_neighbor.size() > 0) {
      continue;
    }

    const PointType& point_vec = GetPoint(point_id);
    DynamicBitset visited_point_flag(PointSize(), 0);
    // point search from key point in every bucket
    for (IntIndex neighbor_bucket_id : neighbor_bucket_list) {
      const IdList& key_point_list = bucket2key_point_[neighbor_bucket_id];
      for (IntIndex start_point_id : key_point_list) {
        GreedyFindKnnInGraph(point_vec, all_point_knn_graph_, start_point_id, 
                             point_update_neighbor, visited_point_flag); 
      }
    }

    // neighbors search from current point's update candidates
    PointNeighbor& point_neighbor = all_point_knn_graph_[point_id];
    size_t point_neighbor_num = point_neighbor.effect_size(point_neighbor_num_);
    for (size_t i = 0; i < point_neighbor_num; i++) {
      IntIndex neighbor_point_id = point_neighbor[i].id;

      PointNeighbor& neighbor_point_update_neighbor = to_update_candidates[neighbor_point_id];
      if (neighbor_point_update_neighbor.size() == 0) {
        continue;
      }

      const PointType& neighbor_point_vec = GetPoint(neighbor_point_id);
      visited_point_flag.assign(PointSize(), 0);
      size_t effect_size = point_update_neighbor.effect_size(search_point_neighbor_num_);
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectSplitedBuckets(
    BucketId2PointList& bucket2point_list,
    ContinuesPointKnnGraph& to_update_candidates) {

  std::vector<std::pair<IntIndex, IdList> > bucket_connect_pair_list;
  for (auto iter = splited_bucket_map_.begin();
            iter != splited_bucket_map_.end(); iter++) {
    IdList bucket_list = iter->second;
    bucket_list.push_back(iter->first);

    for (size_t i = 0; i < bucket_list.size()-1; i++) {
      IdList connect_list(bucket_list.begin()+i, bucket_list.end());
      bucket_connect_pair_list.push_back(std::make_pair(bucket_list[i], connect_list));
    }
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t pair_id = 0; pair_id < bucket_connect_pair_list.size(); pair_id++) {
    IntIndex bucket_id = bucket_connect_pair_list[pair_id].first;
    IdList& to_connect_list = bucket_connect_pair_list[pair_id].second;
    ConnectBucket2BucketList(bucket2point_list, bucket_id, to_connect_list, to_update_candidates);
  }

  #pragma omp parallel for schedule(dynamic, 20)
  for (size_t point_id = 0; point_id < to_update_candidates.size(); point_id++) {
    PointNeighbor& candidate_heap = to_update_candidates[point_id];
    for (auto iter = candidate_heap.begin(); iter != candidate_heap.end(); iter++) {
      UpdatePointKnn(point_id, iter->id, iter->distance);
    }
    candidate_heap.clear();
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetBucket2ConnectBucketList(
    BucketKnnGraph& bucket_knn_graph,
    BucketId2BucketIdList& bucket2connect_list) {

  util::PointPairSet<IntCode> has_selected_connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    IntIndex from_bucket = bucket_id;
    if (merged_bucket_map_.find(from_bucket) != merged_bucket_map_.end()) {
      from_bucket = merged_bucket_map_[from_bucket];
    }
    IdList& to_connect_list = bucket2connect_list[from_bucket];
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
      to_connect_list.push_back(target_bucket);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectAllBuckets2BucketList(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    ContinuesPointKnnGraph& to_update_candidates,
    DynamicBitset& boundary_point_flag) {

  BucketId2BucketIdList bucket2connect_list(OriginBucketSize(), IdList());
  GetBucket2ConnectBucketList(bucket_knn_graph, bucket2connect_list);

  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    IdList& to_connect_list = bucket2connect_list[bucket_id];
    if (to_connect_list.size() == 0) {
      continue;
    }

    ConnectBucket2BucketList(bucket2point_list, bucket_id, to_connect_list, to_update_candidates);
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

    // only one side updated points are selected as boundary point
    if (update_count > 0) {
      boundary_point_flag[point_id] = 1;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveSearch(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    DynamicBitset& boundary_point_flag) {

  ContinuesPointKnnGraph to_update_candidates(PointSize(), PointNeighbor(max_point_neighbor_num_));

  ConnectSplitedBuckets(bucket2point_list, to_update_candidates);

  clock_t s, e;
  s = clock();
  // to_update_candidates will be cleared by ConnectSplitedBuckets
  // so here it needn't be cleare again
  ConnectAllBuckets2BucketList(bucket2point_list, bucket_knn_graph, to_update_candidates, boundary_point_flag);
  e = clock();
  std::cout << "connect neighbor update: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveHashing(
    BucketId2PointList& bucket2point_list, IdList& point_id2bucket_id) {
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
    point_id2bucket_id.push_back(bucket_id);
  }

  bucket2key_point_ = BucketId2PointList(OriginBucketSize());
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitMergeBuckets(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    int max_bucket_size, int min_bucket_size) {

  // split firstly for that too small buckets can be merged
  SplitBuckets(bucket2point_list, max_bucket_size, min_bucket_size); 

  // merge buckets
  MergeBuckets(bucket2point_list, max_bucket_size, min_bucket_size, bucket_knn_graph);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBuckets(
    BucketId2PointList& bucket2point_list, 
    int max_bucket_size, int min_bucket_size) {

  IntIndex origin_max_bucket_id = OriginBucketSize();
  for (IntIndex bucket_id = 0; bucket_id < origin_max_bucket_id; bucket_id++) {
    if (bucket2point_list[bucket_id].size() <= max_bucket_size + min_bucket_size) {
      continue;
    }

    SplitOneBucket(bucket2point_list, bucket_id, max_bucket_size, min_bucket_size);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitOneBucket(
    BucketId2PointList& bucket2point_list, IntCode cur_bucket, 
    int max_bucket_size, int min_bucket_size) {

  while (bucket2point_list[cur_bucket].size() > max_bucket_size + min_bucket_size) {
    auto bucket_begin_iter = bucket2point_list[cur_bucket].end() - max_bucket_size;
    auto bucket_end_iter = bucket2point_list[cur_bucket].end();

    // add new bucket
    int new_bucket = bucket2point_list.size();
    splited_bucket_map_[cur_bucket].push_back(new_bucket);
    bucket2key_point_.push_back(IdList());

    bucket2point_list.push_back(IdList(bucket_begin_iter, bucket_end_iter));
    bucket2point_list[cur_bucket].erase(bucket_begin_iter, bucket_end_iter); 
  }
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
        <= max_bucket_size + min_bucket_size) {
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsPointsKnnGraph(
    BucketId2PointList& bucket2point_list) {
  // build bucket knn graph
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    BuildPointsKnnGraph(bucket2point_list[bucket_id], all_point_knn_graph_);
  }

  // make heap becomes sorted array
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    all_point_knn_graph_[point_id].sort();
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints(
    BucketId2PointList& bucket2point_list, int key_point_num) {

  key_point_num = std::max(key_point_num, 1);

  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    IdList& point_list = bucket2point_list[bucket_id];
    if (point_list.empty()) {
      continue;
    }

    DynamicBitset point_pass_flag(PointSize(), 0);
    size_t neighbor_pass_size = point_list.size() / key_point_num;
    for (IntIndex point_id : point_list) {
      if (point_pass_flag[point_id]) {
        continue;
      }

      // select current point
      bucket2key_point_[bucket_id].push_back(point_id);

      // find enough num key points
      if (bucket2key_point_[bucket_id].size() > key_point_num) {
        break;
      }

      point_pass_flag[point_id] = 1;

      // pass current key point's neighbor (use max_point_neighbor_num_ for max margin)
      PointNeighbor& neighbor_heap = all_point_knn_graph_[point_id];
      size_t neighbor_effect_size = neighbor_heap.effect_size(neighbor_pass_size);
      for (size_t i = 0; i < neighbor_effect_size; i++) {
        IntIndex neighbor_id = neighbor_heap[i].id;
        point_pass_flag[neighbor_id] = 1;
      }
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
IntIndex GraphIndex<PointType, DistanceFuncType, DistanceType>::GetNearestKeyPoint(
    IntIndex bucket_id, const PointType& point_vec) {

  const IdList& key_point_list = bucket2key_point_[bucket_id];

  IntIndex start_point_id = -1;
  DistanceType nearest_dist;
  for (auto key_point_id : key_point_list) {
    DistanceType key_dist = distance_func_(point_vec, GetPoint(key_point_id));
    if (start_point_id == -1 || key_dist < nearest_dist) {
      start_point_id = key_point_id;
      nearest_dist = key_dist;
    }
  }
  return start_point_id;
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
