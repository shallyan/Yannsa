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
    // bucket
    typedef std::vector<IntCode> BucketList;
    typedef std::unordered_set<IntCode> BucketSet;
    typedef std::unordered_map<IntCode, IntCode> MergedBucketMap; 
    typedef std::unordered_map<IntCode, std::vector<IntIndex> > Bucket2Point; 
    // bucket knn graph : MAP --> discrete and not too many items
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::unordered_map<IntCode, BucketHeap> BucketKnnGraph;
    typedef std::unordered_map<IntCode, std::unordered_set<IntCode> > Bucket2ConnectedBuckets;

    // point
    typedef std::vector<IntIndex> PointList;
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

    void SplitMergeBuckets(Bucket2Point& bucket2point, 
                           int bucket_neighbor_num,
                           int max_bucket_size, 
                           int min_bucket_size);

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
    
    void GetBucketList(Bucket2Point& bucket2point, 
                       BucketList& bucket_list); 

    template <typename PointKnnGraphType>
    void FindKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                        PointDistancePairItem& start_point, PointHeap& k_candidates_heap, 
                        std::unordered_set<IntIndex>& has_visited_point); 

    template <typename PointKnnGraphType>
    void FindGreedyKnnInGraph(const PointType& query, PointKnnGraphType& knn_graph,
                        PointDistancePairItem& start_point, PointHeap& k_candidates_heap, 
                        std::unordered_set<IntIndex>& has_visited_point); 

    void ConnectBucketPoints(Bucket2Point& bucket2point, int point_neighbor_num, int bucket_neighbor_num); 

    void Sample(PointSet& point_set, PointList& sampled_point_list, int sample_num); 

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
    DiscretePointKnnGraph key_point_knn_graph_;
    Bucket2Point bucket2key_point_;
    MergedBucketMap merged_bucket_map_; 
    util::BaseEncoderPtr<PointType> encoder_ptr_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param,
    util::BaseEncoderPtr<PointType>& encoder_ptr) { 
  Clear();

  // record encoder for query search
  encoder_ptr_ = encoder_ptr;

  // encode
  Bucket2Point bucket2point;
  util::Log("before encode");
  Encode2Buckets(bucket2point, index_param.point_neighbor_num); 
  util::Log("encode done");

  // get bucket list
  SplitMergeBuckets(bucket2point, index_param.bucket_neighbor_num, 
                    index_param.max_bucket_size, index_param.min_bucket_size); 

  // build point knn graph
  util::Log("before build all knn graph");
  BuildAllBucketsPointsKnnGraph(bucket2point);
  util::Log("build all point knn graph done");

  // find key points in bucket
  //FindBucketKeyPoints(bucket2point, index_param.bucket_neighbor_num);
  FindBucketKeyPoints(bucket2point, 1);

  ConnectBucketPoints(bucket2point, index_param.point_neighbor_num, index_param.bucket_neighbor_num); 

  util::Log("start refine ");

  //RefineByExpansion(2);

  util::Log("end refine ");

  /*
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
  */

  // build
  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Sample(
    PointSet& point_set, PointList& sampled_point_list, int sample_num) {
  util::IntRandomGenerator rg(0, sample_num-1);
  for (auto iter = point_set.begin(); iter != point_set.end(); iter++) {
    if (sampled_point_list.size() < sample_num) {
      sampled_point_list.push_back(*iter);
    }
    else {
      sampled_point_list[rg.Random()] = *iter;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectBucketPoints(
    Bucket2Point& bucket2point, int point_neighbor_num, int bucket_neighbor_num) {
  // reverse knn for each data 
  Point2PointSet point_reverse_neighbor;
  GetPointReverseNeighbors(point_reverse_neighbor); 

  BucketList bucket_list;
  GetBucketList(bucket2point, bucket_list);

  util::Log("before build bucket knn graph");
  BucketKnnGraph bucket_knn_graph;
  BuildBucketsKnnGraph(bucket_list, bucket_list,
                       bucket_neighbor_num, bucket_knn_graph);
  util::Log("build bucket knn graph done");

  // get each bucket connect bucket
  util::PointPairTable<IntCode> connect_pair_set;
  for (auto& bucket_id : bucket_list) {
    auto& neighbor_buckets_heap = bucket_knn_graph[bucket_id];
    for (auto iter = neighbor_buckets_heap.Begin();
              iter != neighbor_buckets_heap.End(); iter++) {
      connect_pair_set.Insert(bucket_id, iter->id);
    }
  }
  
  util::PointPairList<IntCode> connect_pair_list(connect_pair_set.Begin(), connect_pair_set.End());
  std::vector<bool> connect_pair_flag(connect_pair_list.size(), true);
  std::vector<util::PointPairList<IntCode> > batch_connect_pairs;
  while (true) {
    util::PointPairList<IntCode> one_batch_pair_list;
    BucketSet con_bucket_set;
    for (int con_pair_id = 0; con_pair_id < connect_pair_flag.size(); con_pair_id++) {
      if (connect_pair_flag[con_pair_id]) {
        auto& cur_pair = connect_pair_list[con_pair_id];
        if (con_bucket_set.find(cur_pair.first) == con_bucket_set.end() &&
            con_bucket_set.find(cur_pair.second) == con_bucket_set.end()) {
          con_bucket_set.insert(cur_pair.first);
          con_bucket_set.insert(cur_pair.second);
          connect_pair_flag[con_pair_id] = false;
          one_batch_pair_list.push_back(cur_pair);
        }
      }
    }
    if (one_batch_pair_list.size() == 0) {
      break;
    }
    batch_connect_pairs.push_back(one_batch_pair_list);
  }

  util::Log("start lsh search");
  int cc_count = 0;
  for (util::PointPairList<IntCode>& one_batch_pair_list : batch_connect_pairs) {
    cc_count++;
    if (cc_count % 50 == 0) {
      std::cout << "finish lsh buckets " << cc_count << std::endl;
    }
    #pragma omp parallel for schedule(dynamic, 5)
    for (int pair_id = 0; pair_id < one_batch_pair_list.size(); pair_id++) {
      IntCode bucket_id = one_batch_pair_list[pair_id].first;
      IntCode neighbor_bucket_id = one_batch_pair_list[pair_id].second;

      #pragma omp critical
      PointList& start_point_list = bucket2key_point_[neighbor_bucket_id];
      #pragma omp critical
      PointList& bucket_point_list = bucket2point[bucket_id];

      for (auto& point_id : bucket_point_list) {
        /*
        PointSet pass_point;
        if (pass_point.find(point_id) != pass_point.end()) {
          continue;
        }
        */

        IntIndex start_point_id = start_point_list[0];
        auto& start_point_vec = GetPoint(start_point_id);

        PointHeap k_candidates_heap(point_neighbor_num);
        std::unordered_set<IntIndex> has_visited_point;

        auto& point_vec = GetPoint(point_id);
        DistanceType dist = distance_func_(start_point_vec, point_vec);
        PointDistancePairItem start_point_dist(start_point_id, dist);
        has_visited_point.insert(start_point_id);
      
        k_candidates_heap.Insert(start_point_dist);
        FindKnnInGraph(point_vec,
                             all_point_knn_graph_,
                             start_point_dist, k_candidates_heap,
                             has_visited_point); 

        // update
        auto update_iter = k_candidates_heap.Begin();
        for (; update_iter != k_candidates_heap.End(); update_iter++) {
          all_point_knn_graph_[point_id].Insert(*update_iter);
          all_point_knn_graph_[update_iter->id].Insert(PointDistancePairItem(point_id, update_iter->distance));
        }

        /*
        if (update_count > 0) {
          auto iter = point_reverse_neighbor[point_id].begin();
          for (; iter != point_reverse_neighbor[point_id].end(); iter++) {
            pass_point.insert(*iter);
          }
        }
        */
      }
    }
  }

  util::Log("end lsh search");
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetBucketList(
    Bucket2Point& bucket2point, BucketList& bucket_list) { 
  for (auto& item : bucket2point) {
    bucket_list.push_back(item.first);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitMergeBuckets(
    Bucket2Point& bucket2point, 
    int bucket_neighbor_num,
    int max_bucket_size,
    int min_bucket_size) {

  BucketList bucket_list, to_split_bucket_list, to_merge_bucket_list;
  GetSplitMergeBucketsList(bucket2point, bucket_list, to_split_bucket_list, 
                           to_merge_bucket_list, max_bucket_size, min_bucket_size);

  // construct bucket knn graph which need be merged
  BucketKnnGraph to_merge_bucket_knn_graph;
  BuildBucketsKnnGraph(to_merge_bucket_list, bucket_list, 
                       bucket_neighbor_num, 
                       to_merge_bucket_knn_graph);

  // split firstly for that too small bucketscan be merged
  //SplitBuckets(bucket2point, max_bucket_size, min_bucket_size, to_split_bucket_list); 

  std::cout << "before merge size " << bucket2point.size() << std::endl;
  // merge buckets
  MergeBuckets(bucket2point, max_bucket_size, min_bucket_size, to_merge_bucket_knn_graph);

  std::cout << "after merge size " << bucket2point.size() << std::endl;
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
      key_heap.Insert(PointDistancePair<int, int>(point_id, point_in_degree_count[point_id]));
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

  PointList start_points;
  IntCode bucket_code = encoder_ptr_->Encode(query);

  // bucket may be merged and merged bucket may also be merged
  MergedBucketMap::iterator merged_iter;
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
      /*
      PointHeap nearest_heap(100);
      FindKnnInGraph(query, key_point_knn_graph_, point_dist, nearest_heap, has_visited_point); 
      for (auto ii = nearest_heap.Begin(); ii != nearest_heap.End(); ii++) {
        key_candidates_heap.Insert(*ii);
      }
      */
    }
  }

  // search from key points in all point knn graph
  //has_visited_point.clear();
  PointHeap result_candidates_heap(k);
  auto key_iter = key_candidates_heap.Begin();
  for (; key_iter != key_candidates_heap.End(); key_iter++) {
    PointHeap k_candidates_heap(k);
    FindKnnInGraph(query, all_point_knn_graph_, *key_iter, k_candidates_heap, has_visited_point); 
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
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindKnnInGraph(
    const PointType& query, PointKnnGraphType& knn_graph,
    PointDistancePairItem& start_point, PointHeap& k_candidates_heap,
    std::unordered_set<IntIndex>& has_visited_point) {
  PointHeap traverse_heap(0);
  traverse_heap.Insert(start_point);
  k_candidates_heap.Insert(start_point);
  while (traverse_heap.Size() > 0) {
    IntIndex cur_nearest_point = traverse_heap.Front().id;
    traverse_heap.Pop();

    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.Begin();
    for (; neighbor_iter != point_neighbor.End(); neighbor_iter++) {
      if (has_visited_point.find(neighbor_iter->id) == has_visited_point.end()) {
        DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
        PointDistancePairItem point_dist(neighbor_iter->id, dist);
        int update_count = k_candidates_heap.Insert(point_dist);
        if (update_count > 0) {
          traverse_heap.Insert(point_dist);
        }
        has_visited_point.insert(neighbor_iter->id);
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
template <typename PointKnnGraphType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindGreedyKnnInGraph(
    const PointType& query, PointKnnGraphType& knn_graph,
    PointDistancePairItem& start_point, PointHeap& k_candidates_heap,
    std::unordered_set<IntIndex>& has_visited_point) {
  PointHeap traverse_heap(1);
  traverse_heap.Insert(start_point);
  k_candidates_heap.Insert(start_point);
  while (true) {
    IntIndex cur_nearest_point = traverse_heap.Front().id;

    PointHeap& point_neighbor = knn_graph[cur_nearest_point];
    auto neighbor_iter = point_neighbor.Begin();
    int update_count = 0;
    for (; neighbor_iter != point_neighbor.End(); neighbor_iter++) {
      if (has_visited_point.find(neighbor_iter->id) == has_visited_point.end()) {
        DistanceType dist = distance_func_(GetPoint(neighbor_iter->id), query);
        PointDistancePairItem point_dist(neighbor_iter->id, dist);
        k_candidates_heap.Insert(point_dist);
        update_count += traverse_heap.Insert(point_dist);
        has_visited_point.insert(neighbor_iter->id);
      }
    }
    if (update_count == 0) {
      break;
    }
  }
}

} // namespace core 
} // namespace yannsa

#endif
