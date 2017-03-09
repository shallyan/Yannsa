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
    typedef std::unordered_map<IntIndex, IdList> BucketId2BucketIdList; 

    // point
    typedef std::vector<IdList> PointId2PointList;
    // point knn graph : VECTOR --> continues
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointHeap;
    typedef std::vector<PointHeap> ContinuesPointKnnGraph;
    typedef std::unordered_map<IntIndex, PointHeap> DiscretePointKnnGraph;

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

    inline IntIndex OriginBucketSize() {
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
      bucket2key_point_ = BucketId2PointList(OriginBucketSize());
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

    void ConnectTwoBucketPoints(BucketId2PointList& bucket2point_list, 
                                PointId2PointList& point2bi_neighbors,
                                IntIndex bucket_id, IntIndex neighbor_bucket_id,
                                ContinuesPointKnnGraph& to_update_candidates);

    void BatchConnectBucketPairs(BucketKnnGraph& bucket_knn_graph, 
                                 std::vector<util::PointPairList<IntCode> >& connect_pairs_batch);

    void Sample(IdList& point_list, IdList& sampled_point_list, int sample_num); 

    void RefineByExpansion(int iteration_num); 

    void GetPointReverseNeighbors(PointId2PointList& point2point_list, 
                                  PointId2PointList& point2reverse_neighbors); 

    void GetPointBidirectionalNeighbors(PointId2PointList& point2bi_neighbors); 
    void UniquePoint2PointList(PointId2PointList& point2point_list); 

  private:
    std::vector<std::string> index2key_;
    ContinuesPointKnnGraph all_point_knn_graph_;

    BucketId2PointList bucket2key_point_;
    DiscretePointKnnGraph key_point_knn_graph_;

    std::vector<IntCode> bucket_id2code_;
    std::unordered_map<IntCode, int> bucket_code2id_;

    BucketId2BucketId merged_bucket_map_; 

    BucketId2BucketIdList splited_bucket_map_; 

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

  /*
  // random 
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
  RefineByExpansion(10);
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
    std::sort(point_list.begin(), point_list.end());
    point_list.resize(std::distance(point_list.begin(), std::unique(point_list.begin(), point_list.end())));
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
    for (auto iter = neighbor_heap.Begin(); iter != neighbor_heap.End(); iter++) {
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

  float ratio = 0.5;
  int sample_num = 8;

  int max_point_id = PointSize();
  for (int loop = 0; loop < iteration_num; loop++) {
    // init
    PointId2PointList point2old(max_point_id), point2new(max_point_id);
    #pragma omp parallel for schedule(static)
    for (IntIndex cur_point_id = 0; cur_point_id < max_point_id; cur_point_id++) {
      PointHeap& neighbor_heap = all_point_knn_graph_[cur_point_id];
      for (auto iter = neighbor_heap.Begin(); iter != neighbor_heap.End(); iter++) {
        if (iter->flag) {
          //if (point2new[cur_point_id].size() <= sample_num) {
            point2new[cur_point_id].push_back(iter->id);
            iter->flag = false;
          //}
        }
        else {
          point2old[cur_point_id].push_back(iter->id);
        }
      }
    }

    // reverse
    PointId2PointList point2old_reverse(max_point_id), point2new_reverse(max_point_id);
    GetPointReverseNeighbors(point2old, point2old_reverse);
    GetPointReverseNeighbors(point2new, point2new_reverse);

    int update_count = 0;
    //#pragma omp parallel for schedule(dynamic, 20)
    for (IntIndex cur_point_id = 0; cur_point_id < max_point_id; cur_point_id++) {
      // sample reverse  
      /*
      IdList old_reverse_sampled_list, new_reverse_sampled_list;
      Sample(point2old_reverse[cur_point_id], old_reverse_sampled_list, sample_num);
      Sample(point2new_reverse[cur_point_id], new_reverse_sampled_list, sample_num);
      */

      IdList old_reverse_sampled_list = point2old_reverse[cur_point_id];
      IdList new_reverse_sampled_list = point2new_reverse[cur_point_id];

      // merge
      IdList& old_list = point2old[cur_point_id];
      IdList& new_list = point2new[cur_point_id];
      old_list.insert(old_list.end(), old_reverse_sampled_list.begin(), old_reverse_sampled_list.end());
      new_list.insert(new_list.end(), new_reverse_sampled_list.begin(), new_reverse_sampled_list.end());

      // unique
      std::sort(old_list.begin(), old_list.end());
      std::sort(new_list.begin(), new_list.end());
      old_list.resize(std::distance(old_list.begin(), std::unique(old_list.begin(), old_list.end())));
      new_list.resize(std::distance(new_list.begin(), std::unique(new_list.begin(), new_list.end())));

      // update new new
      for (IntIndex u1 : new_list) {
        const PointType& point_vec = GetPoint(u1);
        for (IntIndex u2 : new_list) {
          if (u1 < u2) {
            DistanceType dist = distance_func_(point_vec, GetPoint(u2));
            //#pragma omp critical
            {
              update_count += all_point_knn_graph_[u1].UniqInsert(PointDistancePairItem(u2, dist, true));
              update_count += all_point_knn_graph_[u2].UniqInsert(PointDistancePairItem(u1, dist, true));
            }
          }
        }
        for (IntIndex u2 : old_list) {
          DistanceType dist = distance_func_(point_vec, GetPoint(u2));
          //#pragma omp critical
          {
            update_count += all_point_knn_graph_[u1].UniqInsert(PointDistancePairItem(u2, dist, true));
            update_count += all_point_knn_graph_[u2].UniqInsert(PointDistancePairItem(u1, dist, true));
          }
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BatchConnectBucketPairs(
    BucketKnnGraph& bucket_knn_graph,
    std::vector<util::PointPairList<IntCode> >& connect_pairs_batch) {

  util::PointPairTable<IntCode> connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < OriginBucketSize(); bucket_id++) {
    auto& neighbor_buckets_heap = bucket_knn_graph[bucket_id];
    for (auto iter = neighbor_buckets_heap.Begin();
              iter != neighbor_buckets_heap.End(); iter++) {
      IntIndex from_bucket = bucket_id, target_bucket = iter->id;
      // from bucket and target bucket may be merged
      if (merged_bucket_map_.find(from_bucket) != merged_bucket_map_.end()) {
        from_bucket = merged_bucket_map_[from_bucket];
      }
      if (merged_bucket_map_.find(target_bucket) != merged_bucket_map_.end()) {
        target_bucket = merged_bucket_map_[target_bucket];
      }
      if (from_bucket != target_bucket) {
        connect_pair_set.Insert(from_bucket, target_bucket);
      }
    }
  }

  util::PointPairList<IntCode> connect_pair_list(connect_pair_set.Begin(), connect_pair_set.End());
  DynamicBitset selected_connect_pair_flag(connect_pair_list.size(), 0);
  while (true) {
    util::PointPairList<IntCode> one_batch_pair_list;
    DynamicBitset selected_bucket_flag(OriginBucketSize(), 0); 
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectTwoBucketPoints(
    BucketId2PointList& bucket2point_list, 
    PointId2PointList& point2bi_neighbors,
    IntIndex bucket_id, IntIndex neighbor_bucket_id,

    ContinuesPointKnnGraph& to_update_candidates) {

  const IdList& bucket_point_list = bucket2point_list[bucket_id];
  const IdList& key_point_list = bucket2key_point_[neighbor_bucket_id];

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
    IntIndex start_point_id = -1;
    DistanceType nearest_dist;
    for (auto key_point_id : key_point_list) {
      DistanceType key_dist = distance_func_(point_data, GetPoint(key_point_id));
      if (start_point_id == -1 || key_dist < nearest_dist) {
        start_point_id = key_point_id;
        nearest_dist = key_dist;
      }
    }

    {
      DynamicBitset visited_point_flag(bucket_point_flag);
      GreedyFindKnnInGraph(point_data, all_point_knn_graph_,
                           start_point_id, to_update_candidates[point_id],
                           visited_point_flag); 
    }

    point_has_searched_flag[point_id] = 1;
    to_update_candidates[point_id].Sort();

    // neighbor and reverse neighbor
    IdList& bi_neighbors = point2bi_neighbors[point_id];
    for (auto neighbor_point_id : bi_neighbors) {
      if (point_has_searched_flag[neighbor_point_id] || !bucket_point_flag[neighbor_point_id]) {
        continue;
      }
      DynamicBitset visited_point_flag(bucket_point_flag);
      start_point_id = to_update_candidates[point_id].Front().id;
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

  // get need connected bucket pair
  std::vector<util::PointPairList<IntCode> > connect_pairs_batch;
  BatchConnectBucketPairs(bucket_knn_graph, connect_pairs_batch);

  PointId2PointList point2bi_neighbors(PointSize());
  GetPointBidirectionalNeighbors(point2bi_neighbors); 

  // merge splited buckets firstly
  std::vector<IdList> splited_bucket_list;
  for (auto iter = splited_bucket_map_.begin();
            iter != splited_bucket_map_.end(); iter++) {
    IdList bucket_list = iter->second;
    bucket_list.push_back(iter->first);
    splited_bucket_list.push_back(bucket_list);
  }

  {
    ContinuesPointKnnGraph to_update_candidates(all_point_knn_graph_.size(), PointHeap(point_neighbor_num));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int split_id = 0; split_id < splited_bucket_list.size(); split_id++) {
      IdList& bucket_list = splited_bucket_list[split_id];
      for (int i = 0; i < bucket_list.size(); i++) {
        for (int j = i+1; j < bucket_list.size(); j++) {
          ConnectTwoBucketPoints(bucket2point_list, point2bi_neighbors, 
                                 bucket_list[i], bucket_list[j], to_update_candidates);
        }
      }
    }
    // update
    for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
      PointHeap& candidate_heap = to_update_candidates[point_id];
      for (auto iter = candidate_heap.Begin(); iter != candidate_heap.End(); iter++) {
        all_point_knn_graph_[point_id].UniqInsert(*iter);
        all_point_knn_graph_[iter->id].UniqInsert(PointDistancePairItem(point_id, iter->distance));
      }
    }
  }

  // then merger neighbors
  for (util::PointPairList<IntCode>& one_batch_pair_list : connect_pairs_batch) {
    ContinuesPointKnnGraph to_update_candidates(all_point_knn_graph_.size(), PointHeap(point_neighbor_num));

    #pragma omp parallel for schedule(dynamic, 5)
    for (int pair_id = 0; pair_id < one_batch_pair_list.size(); pair_id++) {
      IntCode bucket_id = one_batch_pair_list[pair_id].first;
      IntCode neighbor_bucket_id = one_batch_pair_list[pair_id].second;
      ConnectTwoBucketPoints(bucket2point_list, point2bi_neighbors, 
                             bucket_id, neighbor_bucket_id, to_update_candidates);
    }

    // update
    for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
      PointHeap& candidate_heap = to_update_candidates[point_id];
      for (auto iter = candidate_heap.Begin(); iter != candidate_heap.End(); iter++) {
        all_point_knn_graph_[point_id].UniqInsert(*iter);
        all_point_knn_graph_[iter->id].UniqInsert(PointDistancePairItem(point_id, iter->distance));
      }
    }
  }
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
  SplitBuckets(bucket2point_list, max_bucket_size, min_bucket_size); 

  int before_size = bucket2point_list.size();
  std::cout << "before merge size " << before_size << std::endl;
  // merge buckets
  MergeBuckets(bucket2point_list, max_bucket_size, min_bucket_size, bucket_knn_graph);

  std::cout << "after merge size " << before_size - merged_bucket_map_.size() << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBuckets(
    BucketId2PointList& bucket2point_list, 
    int max_bucket_size, int min_bucket_size) {

  IntIndex origin_max_bucket_id = bucket2point_list.size();
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
      bucket_knn_graph[bucket_id_i].Insert(BucketDistancePairItem(bucket_id_j, bucket_dist)); 
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
    BucketId2PointList& bucket2point_list, int key_point_num) {

  // count points in degree
  std::vector<int> point_in_degree_count(PointSize(), 0);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointHeap& neighbor_heap = all_point_knn_graph_[point_id];
    auto iter = neighbor_heap.Begin();
    for (; iter != neighbor_heap.End(); iter++) {
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

      // check whether current point's neighbor has been selected
      bool is_selected = true;
      PointHeap& neighbor_heap = all_point_knn_graph_[point_iter->id];
      for (auto neighbor_iter = neighbor_heap.Begin();
                neighbor_iter != neighbor_heap.End(); neighbor_iter++) {
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
  if (merged_bucket_map_.find(bucket_code) != merged_bucket_map_.end()) {
    bucket_code = merged_bucket_map_[bucket_code];
  }

  // if bucket not exist, use near bucket 
  IntIndex bucket_id = -1;
  if (bucket_code2id_.find(bucket_code) != bucket_code2id_.end()) {
    bucket_id = bucket_code2id_[bucket_code];
  }
  else {
    for (bucket_id2code_) {
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

  PointHeap traverse_heap(1);
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

    // explore current nearest point's neighbor
    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.Begin();
    for (; neighbor_iter != point_neighbor.End(); neighbor_iter++) {
      if (visited_point_flag[neighbor_iter->id]) {
        continue;
      }

      DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
      PointDistancePairItem point_dist(neighbor_iter->id, dist);

      k_candidates_heap.Insert(point_dist);
      traverse_heap.Insert(point_dist);
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
