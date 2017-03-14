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
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::vector<BucketHeap> BucketKnnGraph;
    typedef std::unordered_map<IntIndex, IntIndex> BucketId2BucketId; 
    typedef std::unordered_map<IntIndex, IdList> BucketId2BucketIdList; 

    // point
    typedef std::vector<IdList> PointId2PointList;
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointHeap;
    typedef std::vector<PointHeap> ContinuesPointKnnGraph;

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

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
      auto neighbor_heap = all_point_knn_graph_[query_id];
      neighbor_heap.sort();
      auto iter = neighbor_heap.begin();
      for (; iter != neighbor_heap.end(); iter++) {
        search_result.push_back(this->dataset_ptr_->GetKeyById(iter->id));
      }
    }

  private:
    void InitPointIndex(int point_neighbor_num) {
      IntIndex max_point_id = this->dataset_ptr_->size();
      all_point_knn_graph_ = ContinuesPointKnnGraph(max_point_id, PointHeap(point_neighbor_num));
    }

    void InitBucketIndex() {
      bucket2key_point_ = BucketId2PointList(OriginBucketSize());
    }

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

    void clear() {
      // point
      all_point_knn_graph_.clear();

      // bucket
      bucket_code2id_.clear();
      bucket_id2code_.clear();
      bucket2key_point_.clear();
      merged_bucket_map_.clear();
      splited_bucket_map_.clear();
    }

    void Encode2Buckets(BucketId2PointList& bucket2point_list); 

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
                        BucketHeap bucket_neighbor_heap);

    void MergeBuckets(BucketId2PointList& bucket2point_list, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& bucket_knn_graph); 

    void BuildBucketsKnnGraph(BucketKnnGraph& bucket_knn_graph); 

    void BuildAllBucketsPointsKnnGraph(BucketId2PointList& bucket2point_list);

    template<typename PointKnnGraphType>
    void BuildPointsKnnGraph(const IdList& point_list, PointKnnGraphType& point_knn_graph); 

    IntIndex GetNearestKeyPoint(IntIndex bucket_id, const PointType& point_vec);

    void FindBucketKeyPoints(BucketId2PointList& bucket2point_list,
                             int key_point_num);

    template <typename PointKnnGraphType>
    void GreedyFindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                              IntIndex start_point_id, PointHeap& k_candidates_heap,
                              DynamicBitset& visited_point_flag);

    void ConnectBucketPoints(BucketId2PointList& bucket2point_list, 
                             BucketKnnGraph& bucket_knn_graph, int point_neighbor_num); 

    void ConnectTwoBucketPoints(BucketId2PointList& bucket2point_list, 
                                PointId2PointList& point2bi_neighbors,
                                IntIndex bucket_id, IntIndex neighbor_bucket_id,
                                ContinuesPointKnnGraph& to_update_candidates);

    void BatchNeighborConnectBucketPairs(BucketKnnGraph& bucket_knn_graph, 
                                         std::vector<util::PointPairList<IntCode> >& connect_pairs_batch);

    void BatchSplitedConnectBucketPairs(std::vector<util::PointPairList<IntCode> >& connect_pairs_batch);

    void DivideConnectPairList2Batch(util::PointPairList<IntCode>& connect_pair_list, 
                                     std::vector<util::PointPairList<IntCode> >& connect_pairs_batch); 

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);

    void Sample(IdList& point_list, IdList& sampled_point_list, int sample_num); 

    void RefineByExpansion(int iteration_num); 

    void GetPointReverseNeighbors(PointId2PointList& point2point_list, 
                                  PointId2PointList& point2reverse_neighbors); 

    void GetPointBidirectionalNeighbors(PointId2PointList& point2bi_neighbors); 
    void UniquePoint2PointList(PointId2PointList& point2point_list); 

  private:
    ContinuesPointKnnGraph all_point_knn_graph_;

    BucketId2PointList bucket2key_point_;

    std::vector<IntCode> bucket_id2code_;
    std::unordered_map<IntCode, int> bucket_code2id_;

    BucketId2BucketId merged_bucket_map_; 

    BucketId2BucketIdList splited_bucket_map_; 

    util::BaseEncoderPtr<PointType> encoder_ptr_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 
  encoder_ptr_ = encoder_ptr;

  clear();

  InitPointIndex(index_param.point_neighbor_num);

  // random 
  /*
  util::IntRandomGenerator rg(0, PointSize()-1);
  for (int i = 0; i < PointSize(); i++) {
    PointHeap& h = all_point_knn_graph_[i];
    for (int j = 0; j < index_param.point_neighbor_num*3; j++) {
      int nei = rg.Random();
      if (nei != i) {
        h.UniqInsert(PointDistancePairItem(nei, 1000000.0));
      }
    }
  }
  */

  // encode
  util::Log("before encode");
  BucketId2PointList bucket2point_list;
  Encode2Buckets(bucket2point_list);
  util::Log("encode done");

  InitBucketIndex();

  // construct bucket knn graph
  util::Log("before bucket knn graph");
  BucketKnnGraph bucket_knn_graph(OriginBucketSize(), BucketHeap(index_param.bucket_neighbor_num));
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
  RefineByExpansion(index_param.refine_iter_num);
  util::Log("end refine");

  // build
  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::UniquePoint2PointList(
    PointId2PointList& point2point_list) {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < point2point_list.size(); point_id++) {
    IdList& point_list = point2point_list[point_id];
    IdSet point_set(point_list.begin(), point_list.end());
    IdList uniq_point_list = IdList(point_set.begin(), point_set.end());
    point_list.swap(uniq_point_list);
  }
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetPointBidirectionalNeighbors(
    PointId2PointList& point2bi_neighbors) {

  for (IntIndex point_id = 0; point_id < all_point_knn_graph_.size(); point_id++) {
    PointHeap& neighbor_heap = all_point_knn_graph_[point_id];
    for (auto iter = neighbor_heap.begin(); iter != neighbor_heap.end(); iter++) {
      point2bi_neighbors[iter->id].push_back(point_id);
      point2bi_neighbors[point_id].push_back(iter->id);
    }
  }
   
  // unique neighbor list
  UniquePoint2PointList(point2bi_neighbors);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Sample(
    IdList& point_list, IdList& sampled_point_list, int sample_num) {
  util::IntRandomGenerator rg(0, sample_num-1);
  for (auto point_id : point_list) {
    if (sampled_point_list.size() < sample_num) {
      sampled_point_list.push_back(point_id);
    }
    else {
      sampled_point_list[rg.Random()] = point_id;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::RefineByExpansion(
    int iteration_num) {

  int max_point_id = PointSize();
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

      PointHeap& neighbor_heap = all_point_knn_graph_[cur_point_id];
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BatchSplitedConnectBucketPairs(
    std::vector<util::PointPairList<IntCode> >& connect_pairs_batch) {

  util::PointPairList<IntCode> connect_pair_list;
  for (auto iter = splited_bucket_map_.begin();
            iter != splited_bucket_map_.end(); iter++) {
    IdList bucket_list = iter->second;
    bucket_list.push_back(iter->first);

    for (int i = 0; i < bucket_list.size(); i++) {
      for (int j = i+1; j < bucket_list.size(); j++) {
        connect_pair_list.push_back(util::PointPair<IntCode>(bucket_list[i], bucket_list[j]));
      }
    }
  }

  DivideConnectPairList2Batch(connect_pair_list, connect_pairs_batch);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BatchNeighborConnectBucketPairs(
    BucketKnnGraph& bucket_knn_graph,
    std::vector<util::PointPairList<IntCode> >& connect_pairs_batch) {

  util::PointPairSet<IntCode> connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    auto& neighbor_buckets_heap = bucket_knn_graph[bucket_id];
    for (auto iter = neighbor_buckets_heap.begin();
              iter != neighbor_buckets_heap.end(); iter++) {
      IntIndex from_bucket = bucket_id, target_bucket = iter->id;
      // from bucket and target bucket may be merged
      if (merged_bucket_map_.find(from_bucket) != merged_bucket_map_.end()) {
        from_bucket = merged_bucket_map_[from_bucket];
      }
      if (merged_bucket_map_.find(target_bucket) != merged_bucket_map_.end()) {
        target_bucket = merged_bucket_map_[target_bucket];
      }
      if (from_bucket != target_bucket) {
        connect_pair_set.insert(from_bucket, target_bucket);
      }
    }
  }

  util::PointPairList<IntCode> connect_pair_list(connect_pair_set.begin(), connect_pair_set.end());
  DivideConnectPairList2Batch(connect_pair_list, connect_pairs_batch);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::DivideConnectPairList2Batch(
    util::PointPairList<IntCode>& connect_pair_list, 
    std::vector<util::PointPairList<IntCode> >& connect_pairs_batch) {

  DynamicBitset selected_connect_pair_flag(connect_pair_list.size(), 0);
  while (true) {
    util::PointPairList<IntCode> one_batch_pair_list;
    DynamicBitset selected_bucket_flag(AllBucketSize(), 0); 
    for (int con_pair_id = 0; con_pair_id < connect_pair_list.size(); con_pair_id++) {
      if (selected_connect_pair_flag[con_pair_id]) {
        continue;
      }
      auto& cur_pair = connect_pair_list[con_pair_id];
      if (selected_bucket_flag[cur_pair.first] || selected_bucket_flag[cur_pair.second]) {
        continue;
      }
      selected_bucket_flag[cur_pair.first] = 1;
      selected_bucket_flag[cur_pair.second] = 1;
      selected_connect_pair_flag[con_pair_id] = 1;
      one_batch_pair_list.push_back(cur_pair);
    }
    if (one_batch_pair_list.size() == 0) {
      break;
    }
    connect_pairs_batch.push_back(one_batch_pair_list);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointKnn(
    IntIndex point1, IntIndex point2, DistanceType dist) {

  if (point1 == point2) {
    return 0;
  }

  int update_count = 0;
  update_count += all_point_knn_graph_[point1].SafeUniqInsert(PointDistancePairItem(point2, dist, true));
  update_count += all_point_knn_graph_[point2].SafeUniqInsert(PointDistancePairItem(point1, dist, true));

  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectTwoBucketPoints(
    BucketId2PointList& bucket2point_list, 
    PointId2PointList& point2bi_neighbors,
    IntIndex bucket_id, IntIndex neighbor_bucket_id,
    ContinuesPointKnnGraph& to_update_candidates) {

  const IdList& bucket_point_list = bucket2point_list[bucket_id];

  DynamicBitset point_has_searched_flag(PointSize(), 0);
  DynamicBitset bucket_point_flag(PointSize(), 0);
  for (IntIndex point_id : bucket_point_list) {
    bucket_point_flag[point_id] = 1;
  }

  // connect bucket_id and neighbor_bucket_id
  // first search one point
  for (IntIndex point_id : bucket_point_list) {
    if (point_has_searched_flag[point_id]) {
      continue;
    }

    const PointType& point_data = GetPoint(point_id);

    // if no neighbor has been searched, start from key point
    IntIndex start_point_id = GetNearestKeyPoint(neighbor_bucket_id, point_data);

    {
      DynamicBitset visited_point_flag(bucket_point_flag);
      GreedyFindKnnInGraph(point_data, all_point_knn_graph_,
                           start_point_id, to_update_candidates[point_id],
                           visited_point_flag); 
    }

    point_has_searched_flag[point_id] = 1;

    // neighbor and reverse neighbor
    IdList& bi_neighbors = point2bi_neighbors[point_id];
    start_point_id = to_update_candidates[point_id].GetMinValue().id;
    for (auto neighbor_point_id : bi_neighbors) {
      if (point_has_searched_flag[neighbor_point_id] || !bucket_point_flag[neighbor_point_id]) {
        continue;
      }
      DynamicBitset visited_point_flag(bucket_point_flag);
      GreedyFindKnnInGraph(GetPoint(neighbor_point_id), all_point_knn_graph_,
                           start_point_id, to_update_candidates[neighbor_point_id],
                           visited_point_flag);
      point_has_searched_flag[neighbor_point_id] = 1;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectBucketPoints(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    int point_neighbor_num) {

  // get reverse neighbor
  PointId2PointList point2bi_neighbors(PointSize());
  GetPointBidirectionalNeighbors(point2bi_neighbors); 

  std::vector<util::PointPairList<IntCode> > connect_pairs_batch;
  // get need splited bucket pair firstly
  BatchSplitedConnectBucketPairs(connect_pairs_batch);
  // get need connected bucket pair
  BatchNeighborConnectBucketPairs(bucket_knn_graph, connect_pairs_batch);

  // then merge neighbors
  ContinuesPointKnnGraph to_update_candidates(all_point_knn_graph_.size(), PointHeap(point_neighbor_num));
  for (util::PointPairList<IntCode>& one_batch_pair_list : connect_pairs_batch) {
    #pragma omp parallel for schedule(static)
    for (int point_id  = 0; point_id < PointSize(); point_id++) {
      to_update_candidates[point_id].clear();
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int pair_id = 0; pair_id < one_batch_pair_list.size(); pair_id++) {
      IntCode bucket_id = one_batch_pair_list[pair_id].first;
      IntCode neighbor_bucket_id = one_batch_pair_list[pair_id].second;
      if (bucket_id >= OriginBucketSize() || 
          splited_bucket_map_.find(bucket_id) != splited_bucket_map_.end()) {
        std::swap(bucket_id, neighbor_bucket_id);
      }
      ConnectTwoBucketPoints(bucket2point_list, point2bi_neighbors, 
                             bucket_id, neighbor_bucket_id, to_update_candidates);
    }

    #pragma omp parallel for schedule(dynamic, 20)
    for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
      PointHeap& candidate_heap = to_update_candidates[point_id];
      for (auto iter = candidate_heap.begin(); iter != candidate_heap.end(); iter++) {
        UpdatePointKnn(point_id, iter->id, iter->distance);
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Encode2Buckets(
    BucketId2PointList& bucket2point_list) { 
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
  }
}


template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitMergeBuckets(
    BucketId2PointList& bucket2point_list, BucketKnnGraph& bucket_knn_graph,
    int max_bucket_size, int min_bucket_size) {

  // split firstly for that too small bucketscan be merged
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
    BucketHeap bucket_neighbor_heap) {

  if (merged_bucket_map_.find(cur_bucket) != merged_bucket_map_.end() || 
      bucket2point_list[cur_bucket].size() >= min_bucket_size) {
    return;
  }

  // copy bucket_neighbor_heap so origin heap is not changed
  bucket_neighbor_heap.sort();
  auto bucket_neighbor_iter = bucket_neighbor_heap.begin();
  for (; bucket_neighbor_iter != bucket_neighbor_heap.end(); bucket_neighbor_iter++) {
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
  #pragma omp parallel for schedule(dynamic, 5)
  for (IntIndex bucket_id_i = 0; bucket_id_i < OriginBucketSize(); bucket_id_i++) {
    for (IntIndex bucket_id_j = 0; bucket_id_j < OriginBucketSize(); bucket_id_j++) {
      if (bucket_id_i == bucket_id_j) {
        continue;
      }

      IntCode bucket_dist = encoder_ptr_->Distance(bucket_id2code_[bucket_id_i], bucket_id2code_[bucket_id_j]);
      bucket_knn_graph[bucket_id_i].insert(BucketDistancePairItem(bucket_id_j, bucket_dist)); 
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
      point_knn_graph[cur_point].insert(PointDistancePairItem(neighbor_point, dist, false));
      point_knn_graph[neighbor_point].insert(PointDistancePairItem(cur_point, dist, false));
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsPointsKnnGraph(
    BucketId2PointList& bucket2point_list) {
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < bucket2point_list.size(); bucket_id++) {
    BuildPointsKnnGraph(bucket2point_list[bucket_id], all_point_knn_graph_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints(
    BucketId2PointList& bucket2point_list, int key_point_num) {

  // count points in degree
  std::vector<int> point_in_degree_count(PointSize(), 0);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointHeap& neighbor_heap = all_point_knn_graph_[point_id];
    auto iter = neighbor_heap.begin();
    for (; iter != neighbor_heap.end(); iter++) {
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
      min_in_degree_heap.push(PointDistancePair<IntIndex, int>(point_id, point_in_degree_count[point_id]));
    }
    min_in_degree_heap.sort();

    DynamicBitset point_pass_flag(PointSize(), 0);
    for (auto point_iter = min_in_degree_heap.begin(); 
              point_iter != min_in_degree_heap.end(); point_iter++) {
      if (point_pass_flag[point_iter->id]) {
        continue;
      }
      point_pass_flag[point_iter->id] = 1;

      // check whether current point's neighbor has been selected
      bool is_selected = true;
      PointHeap& neighbor_heap = all_point_knn_graph_[point_iter->id];
      for (auto neighbor_iter = neighbor_heap.begin();
                neighbor_iter != neighbor_heap.end(); neighbor_iter++) {
        if (point_pass_flag[neighbor_iter->id]) {
          is_selected = false;
        }
        else {
          // pass current key point's neighbor
          point_pass_flag[neighbor_iter->id] = 1;
        }
      }

      // select current point
      if (is_selected) {
        bucket2key_point_[bucket_id].push_back(point_iter->id);
      }

      // find enough num key points
      if (bucket2key_point_[bucket_id].size() > key_point_num) {
        break;
      }
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
    throw IndexNotBuildError("Graph index hasn't been built!");
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
  PointHeap result_candidates_heap(k);
  DynamicBitset visited_point_flag(PointSize(), 0);
  for (auto start_point_id : bucket2key_point_[bucket_id]) {
    PointHeap tmp_result_candidates_heap(k);
    GreedyFindKnnInGraph(query, all_point_knn_graph_,
                         start_point_id, tmp_result_candidates_heap,
                         visited_point_flag); 
    for (auto iter = tmp_result_candidates_heap.begin(); iter != tmp_result_candidates_heap.end(); iter++) {
      result_candidates_heap.UniqInsert(*iter);
    }
  }
  
  search_result.clear();
  result_candidates_heap.sort();
  auto result_iter = result_candidates_heap.begin();
  for (; result_iter != result_candidates_heap.end(); result_iter++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(result_iter->id));
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

  // max heap, so get min top k
  PointHeap traverse_heap(1);
  traverse_heap.insert(start_point);
  k_candidates_heap.insert(start_point);
  while (traverse_heap.size() > 0) {
    // start from current point
    IntIndex cur_nearest_point = traverse_heap.front().id;

    // explore current nearest point's neighbor
    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.begin();
    for (; neighbor_iter != point_neighbor.end(); neighbor_iter++) {
      if (visited_point_flag[neighbor_iter->id]) {
        continue;
      }

      DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
      PointDistancePairItem point_dist(neighbor_iter->id, dist);

      k_candidates_heap.insert(point_dist);
      traverse_heap.insert(point_dist);
      visited_point_flag[neighbor_iter->id] = 1;
    }
    if (traverse_heap.front().id == cur_nearest_point) {
      break;
    }
  }
}

} // namespace core 
} // namespace yannsa

#endif
