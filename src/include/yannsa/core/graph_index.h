#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/binary_encoder.h"
#include "yannsa/util/point_pair.h"
#include "yannsa/util/logging.h"
#include "yannsa/util/lock.h"
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
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketNeighbor;
    typedef std::unordered_map<IntIndex, IntIndex> BucketId2BucketIdMap; 
    typedef std::unordered_map<IntCode, IntIndex> BucketCode2BucketIdMap; 
    typedef std::vector<IdList> BucketId2BucketIdList;

    // point
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointNeighbor;

  private:
    struct BucketInfo {
      IdList knn_list;
      IdList point_list;
      IdList key_point_list;
      IntCode code;

      BucketInfo(IntCode c) {
        code = c;
      }

      void clear() {
        point_list.clear();
      }
    };

    struct PointInfo {
      // graph
      PointNeighbor knn;
      IdList nav_list;

      // basic info
      IntIndex bucket_id;

      // status
      bool is_updated;
      bool is_join;

      // for local join
      IdList old_list;
      IdList new_list;
      IdList reverse_old_list;
      IdList reverse_new_list;
      DistanceType radius;
      size_t effect_size;

      util::Mutex lock;

      PointInfo(int point_neighbor_num) {
        knn = PointNeighbor(point_neighbor_num);
      }

      void reset(size_t s) {
        old_list.clear();
        new_list.clear();
        reverse_new_list.clear();
        reverse_old_list.clear();
        effect_size = s;
        if (effect_size > 0) {
          radius = knn[effect_size-1].distance;
        }
        else {
          radius = 0;
        }
      }

      void insert(IntIndex point_id, bool new_flag) {
        if (new_flag) {
          new_list.push_back(point_id);
        }
        else {
          old_list.push_back(point_id);
        }
      }

      void parallel_insert_reverse(IntIndex point_id, bool new_flag) {
        util::ScopedLock sl(lock);
        if (new_flag) {
          reverse_new_list.push_back(point_id);
        }
        else {
          reverse_old_list.push_back(point_id);
        }
      }
    };

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Clear(); 

    void Build(const util::GraphIndexParameter& index_param, 
               util::BinaryEncoderPtr<PointType>& encoder_ptr); 

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

    void Save(const std::string file_path);
    void SaveBinary(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param,
              util::BinaryEncoderPtr<PointType>& encoder_ptr); 

    inline const PointType& GetPoint(IntIndex point_id) {
      return (*this->dataset_ptr_)[point_id];
    }
    
    inline IntIndex PointSize() {
      return all_point_info_.size();
    }

    inline IntIndex BucketSize() {
      return all_bucket_info_.size();
    }

    void LocalitySensitiveHashing();

    void MergeOneBucket(IntIndex bucket_id,
                        int max_bucket_size,
                        int min_bucket_size);

    void MergeBuckets(int max_bucket_size,
                      int min_bucket_size);

    void UpdatePointBucket(); 

    void BuildBucketsKnnGraph();

    void BuildAllBucketsApproximateKnnGraph(int max_bucket_size);

    void SortBucketPointsByInDegree();

    void FindBucketKeyPoints();

    void BuildNavigateGraph();

    void GreedyFindKnnInGraph(const PointType& query,
                              IntIndex start_point_id, PointNeighbor& k_candidates_heap,
                              DynamicBitset& visited_point_flag);

    void LocalitySensitiveSearch();

    void GetBucket2ConnectBucketList(BucketId2BucketIdList& bucket2connect_list);

    void ConnectBucket2BucketList(IntIndex bucket_id, IdList& neighbor_bucket_list,
                                  std::vector<PointNeighbor>& to_update_candidates);

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);
    int UpdatePointKnn(IntIndex point1, IntIndex point2);

    void InitPointNeighborInfo(bool is_global);
    void UpdatePointNeighborInfo(); 
    int LocalJoin(bool is_global);

  private:
    int point_neighbor_num_;
    int max_point_neighbor_num_;
    int search_point_neighbor_num_;
    int search_start_point_num_;

    std::vector<PointInfo> all_point_info_;
    std::vector<BucketInfo> all_bucket_info_;

    BucketCode2BucketIdMap bucket_code2id_;
    BucketId2BucketIdMap merged_bucket_map_; 

    util::BinaryEncoderPtr<PointType> encoder_ptr_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Clear() {
  this->have_built_ = false;
  // point
  all_point_info_.clear();
  all_bucket_info_.clear();

  // bucket
  bucket_code2id_.clear();
  merged_bucket_map_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Init(
    const util::GraphIndexParameter& index_param,
    util::BinaryEncoderPtr<PointType>& encoder_ptr) { 

  encoder_ptr_ = encoder_ptr;
  point_neighbor_num_ = index_param.point_neighbor_num;

  max_point_neighbor_num_ = std::max(index_param.max_point_neighbor_num,
                                     point_neighbor_num_);
  search_point_neighbor_num_ = std::min(index_param.search_point_neighbor_num,
                                        point_neighbor_num_);
  search_start_point_num_ = index_param.search_start_point_num;

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_info_ = std::vector<PointInfo>(max_point_id, PointInfo(point_neighbor_num_));
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SaveBinary(
    const std::string file_path) {

  std::ofstream save_file(file_path, std::ios::binary);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    int k = point_neighbor_num_;
    save_file.write((char*)&k, sizeof(int));
    for (size_t i = 0; i < k; i++) {
      int id = 0;
      if (i < point_neighbor_num.size()) {
        id = point_neighbor[i].id;
      }
      save_file.write((char*)&id, sizeof(int));
    }
  }
  save_file.close();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Save(
    const std::string file_path) {

  std::ofstream save_file(file_path);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    save_file << this->dataset_ptr_->GetKeyById(point_id) << " ";
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
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
    util::BinaryEncoderPtr<PointType>& encoder_ptr) { 

  util::Log("before build");

  if (this->have_built_) {
    throw IndexBuildError("Graph index has already been built!");
  }

  Init(index_param, encoder_ptr);

  // encode and init bukcet
  util::Log("before hashing");
  LocalitySensitiveHashing();
  std::cout << "bucket size: " << BucketSize() << std::endl;
  for (int i = 0; i < BucketSize(); i++) {
    std::cout << all_bucket_info_[i].point_dist.size() << " ";
  }
  std::cout << std::endl;

  // construct bucket knn graph
  util::Log("before bucket knn");
  BuildBucketsKnnGraph();

  // merge buckets
  MergeBuckets(index_param.max_bucket_size, index_param.min_bucket_size);
  UpdatePointBucket();
  std::cout << "merged bucket size: " << merged_bucket_map_.size() << std::endl;

  // build point knn graph
  util::Log("before build buckets knn graph");
  BuildAllBucketsApproximateKnnGraph(index_param.max_bucket_size);
  util::Log("end build buckets knn graph");

  util::Log("before local refine");
  // only refine local points
  InitPointNeighborInfo(false);
  for (int loop = 0; loop < index_param.local_refine_iter_num; loop++) {
    UpdatePointNeighborInfo();
    clock_t s, e;
    s = clock();
    LocalJoin(false);
    e = clock();
    std::cout << "refine update: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }
  util::Log("end local refine");

  InitPointNeighborInfo(true);
  util::Log("before search");
  LocalitySensitiveSearch();

  util::Log("before global refine");
  for (int loop = 0; loop < index_param.global_refine_iter_num; loop++) {
    clock_t s, e, e1;
    s = clock();
    UpdatePointNeighborInfo();
    e = clock();
    std::cout << "refine init: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
    LocalJoin(true);
    e1 = clock();
    std::cout << "refine update: " << (e1-e)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // build
  this->have_built_ = true;

  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::InitPointNeighborInfo(
    bool is_global) {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointInfo& point_info = all_point_info_[point_id];
    if (is_global) {
      point_info.is_join = true;
    }
    else if (!point_info.is_join) {
      continue;
    }

    point_info.is_updated = false;

    point_info.effect_size = point_info.knn.size();
    // generally, size >= 1
    if (point_info.effect_size > 0) {
      point_info.radius = point_info.knn[point_info.effect_size-1].distance;
    }
    else {
      point_info.radius = 0;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointNeighborInfo() {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointInfo& point_info = all_point_info_[point_id];
    if (!point_info.is_join) {
      continue;
    }
    
    IntIndex effect_size = point_info.knn.size();
    /*
    IntIndex effect_size = point_info.effect_size;
    auto& point_neighbor = point_info.knn;
    if (point_info.is_updated && effect_size < point_neighbor.size()) {
      int new_point_count = 0;
      for (IntIndex i = 0; i < point_neighbor.size(); i++) {
        if (point_neighbor[i].flag) {
          new_point_count++;
          if (new_point_count >= search_start_point_num_) {
            effect_size = i+1;
            break;
          }
        }
      }
    }
    */
    point_info.reset(effect_size);
  }

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointInfo& point_info = all_point_info_[point_id];
    if (!point_info.is_join) {
      continue;
    }

    auto& point_neighbor = point_info.knn;
    IntIndex effect_size = point_neighbor.effect_size(point_info.effect_size);
    for (IntIndex i = 0; i < effect_size; i++) {
      auto& neighbor = point_neighbor[i];
      PointInfo& neighbor_info = all_point_info_[neighbor.id];
      // neighbor
      point_info.insert(neighbor.id, neighbor.flag);
      // reverse neighbor, avoid repeat element
      if (neighbor.distance > neighbor_info.radius) {
        neighbor_info.parallel_insert_reverse(point_id, neighbor.flag);
      }
      neighbor.flag = false;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalJoin(
    bool is_global) {

  int sample_num = 100;
  int update_count = 0;
  #pragma omp parallel for schedule(dynamic, 5) default(shared) reduction(+:update_count)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointInfo& point_info = all_point_info_[point_id];

    IdList& new_list = point_info.new_list;
    IdList& reverse_new_list = point_info.reverse_new_list;
    if (new_list.size() == 0 && reverse_new_list.size() == 0) {
      continue;
    }
    if (reverse_new_list.size() > sample_num) {
      std::random_shuffle(reverse_new_list.begin(), reverse_new_list.end());
      reverse_new_list.resize(sample_num);
    }
    new_list.insert(new_list.end(), reverse_new_list.begin(), reverse_new_list.end());

    IdList& old_list = point_info.old_list;
    IdList& reverse_old_list = point_info.reverse_old_list;
    if (reverse_old_list.size() > sample_num) {
      std::random_shuffle(reverse_old_list.begin(), reverse_old_list.end());
      reverse_old_list.resize(sample_num);
    }
    old_list.insert(old_list.end(), reverse_old_list.begin(), reverse_old_list.end());

    // update new 
    int cur_update_count = 0;
    for (size_t i = 0; i < new_list.size(); i++) {
      IntIndex p1 = new_list[i];

      for (size_t j = i+1; j < new_list.size(); j++) {
        IntIndex p2 = new_list[j];
        cur_update_count += UpdatePointKnn(p1, p2);
      }

      for (size_t j = 0; j < old_list.size(); j++) {
        IntIndex p2 = old_list[j];
        cur_update_count += UpdatePointKnn(p1, p2);
      }
    }
    update_count += cur_update_count;
    point_info.is_updated = cur_update_count > 0;
  }
  return update_count;
}

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
  int update_pos = all_point_info_[point1].knn.parallel_insert_array(PointDistancePairItem(point2, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;
  update_pos = all_point_info_[point2].knn.parallel_insert_array(PointDistancePairItem(point1, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;

  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ConnectBucket2BucketList(
    IntIndex bucket_id, IdList& neighbor_bucket_list,
    std::vector<PointNeighbor>& to_update_candidates) {

  DynamicBitset visited_point_flag(PointSize(), 0);
  const BucketInfo& bucket_info = all_bucket_info_[bucket_id];
  // connect bucket_id to neighbor_bucket_list
  for (IntIndex point_id : bucket_info.key_point_list) {
    PointNeighbor& point_update_neighbor = to_update_candidates[point_id];

    // point has beed updated
    if (point_update_neighbor.size() > 0) {
      continue;
    }
    
    const PointType& point_vec = GetPoint(point_id);
    
    IdList start_point_list;
    // point search from key point in every bucket
    for (IntIndex neighbor_bucket_id : neighbor_bucket_list) {
      const IdList& key_point_list = all_bucket_info_[neighbor_bucket_id].key_point_list;
      size_t key_point_size = std::min(static_cast<size_t>(search_start_point_num_), key_point_list.size());
      start_point_list.insert(start_point_list.end(), key_point_list.begin(), key_point_list.begin()+key_point_size);
    }

    visited_point_flag.assign(PointSize(), 0);
    for (IntIndex start_point_id : start_point_list) {
      GreedyFindKnnInGraph(point_vec, start_point_id,
                           point_update_neighbor, visited_point_flag);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetBucket2ConnectBucketList(
    BucketId2BucketIdList& bucket2connect_list) {

  util::PointPairSet<IntCode> has_selected_connect_pair_set;
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    IntIndex from_bucket = bucket_id;
    if (merged_bucket_map_.find(from_bucket) != merged_bucket_map_.end()) {
      from_bucket = merged_bucket_map_[from_bucket];
    }

    IdList& neighbor_bucket_list = all_bucket_info_[bucket_id].knn_list;
    for (auto target_bucket : neighbor_bucket_list) {
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
      if (all_bucket_info_[from_bucket].point_list.size() < all_bucket_info_[target_bucket].point_list.size()) {
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveSearch() {

  SortBucketPointsByInDegree();
  FindBucketKeyPoints();

  BucketId2BucketIdList bucket2connect_list(BucketSize(), IdList());
  GetBucket2ConnectBucketList(bucket2connect_list);

  std::vector<PointNeighbor> to_update_candidates(PointSize(), 
              PointNeighbor(max_point_neighbor_num_ - point_neighbor_num_));

  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    IdList& to_connect_list = bucket2connect_list[bucket_id];
    if (to_connect_list.size() == 0) {
      continue;
    }

    ConnectBucket2BucketList(bucket_id, to_connect_list, to_update_candidates);
  }

  #pragma omp parallel for schedule(static)
  for (int point_id = 0; point_id < PointSize(); point_id++) {
    all_point_info_[point_id].knn.remax_size(max_point_neighbor_num_);
  }

  #pragma omp parallel for schedule(dynamic, 5)
  for (int point_id = 0; point_id < to_update_candidates.size(); point_id++) {
    PointNeighbor& candidate_heap = to_update_candidates[point_id];
    if (candidate_heap.size() == 0) {
      continue;
    }

    for (auto iter = candidate_heap.begin(); iter != candidate_heap.end(); iter++) {
      UpdatePointKnn(point_id, iter->id, iter->distance);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveHashing() {

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
      IntIndex new_bucket_id = all_bucket_info_.size();
      all_bucket_info_.push_back(BucketInfo(point_code));

      bucket_code2id_[point_code] = new_bucket_id; 
    }

    IntIndex bucket_id = bucket_code2id_[point_code];
    all_bucket_info_[bucket_id].point_list.push_back(point_id);
    all_point_info_[point_id].bucket_id = bucket_id;
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeOneBucket(
    IntIndex cur_bucket, int max_bucket_size, int min_bucket_size) {

  if (merged_bucket_map_.find(cur_bucket) != merged_bucket_map_.end() || 
      all_bucket_info_[cur_bucket].point_list.size() >= min_bucket_size) {
    return;
  }

  IdList& bucket_knn_list = all_bucket_info_[cur_bucket].knn_list;
  // merge cur_bucket into neighbor_buckt
  for (auto neighbor_bucket: bucket_knn_list) {
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
    IdList& neighbor_point_list = all_bucket_info_[neighbor_bucket].point_list;
    IdList& cur_point_list = all_bucket_info_[cur_bucket].point_list;
    if (neighbor_point_list.size() + cur_point_list.size() <= max_bucket_size) {
        //<= max_bucket_size + min_bucket_size) {
      neighbor_point_list.insert(neighbor_point_list.end(),
                                 cur_point_list.begin(), cur_point_list.end());
      cur_point_list.clear();
      merged_bucket_map_[cur_bucket] = neighbor_bucket;
      break;
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointBucket() {
  
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    IntIndex& bucket_id = all_point_info_[point_id].bucket_id;

    BucketId2BucketIdMap::const_iterator merged_iter = merged_bucket_map_.find(bucket_id);
    if (merged_iter != merged_bucket_map_.end()) {
      bucket_id = merged_iter->second; 
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBuckets(
    int max_bucket_size, int min_bucket_size) {

  // merge
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    if (all_bucket_info_[bucket_id].point_list.size() >= min_bucket_size) {
      continue;
    }

    MergeOneBucket(bucket_id, max_bucket_size, min_bucket_size); 
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildBucketsKnnGraph() {

  int code_length = encoder_ptr_->code_length();
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    BucketInfo& bucket_info = all_bucket_info_[bucket_id];
    IntCode bucket_code = bucket_info.code; 
    int mask = 1;
    for (int i = 0; i < code_length; i++) {
      IntCode neighbor_bucket_code = bucket_code ^ mask;
      mask <<= 1;

      BucketCode2BucketIdMap::const_iterator neighbor_bucket_iter = bucket_code2id_.find(neighbor_bucket_code);
      if (neighbor_bucket_iter != bucket_code2id_.end()) {
        bucket_info.knn_list.push_back(neighbor_bucket_iter->second);
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildAllBucketsApproximateKnnGraph(
    int max_bucket_size) { 

  // build bucket knn graph
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    const IdList& point_list = all_bucket_info_[bucket_id].point_list;
    if (point_list.size() < max_bucket_size) {
      for (int i = 0; i < point_list.size(); i++) {
        IntIndex cur_point = point_list[i];
        all_point_info_[cur_point].is_join = false;
        const PointType& cur_point_vec = GetPoint(cur_point);
        for (int j = i+1; j < point_list.size(); j++) {
          IntIndex neighbor_point = point_list[j];
          DistanceType dist = distance_func_(cur_point_vec, GetPoint(neighbor_point));
          all_point_info_[cur_point].knn.insert_heap(PointDistancePairItem(neighbor_point, dist, false));
          all_point_info_[neighbor_point].knn.insert_heap(PointDistancePairItem(cur_point, dist, false));
        }
      }
    }
    else {
      util::IntRandomGenerator int_rand(0, point_list.size()-1);
      for (IntIndex point_id : point_list) {
        all_point_info_[point_id].is_join = true;
        IdSet neighbor_set;
        size_t random_neighbor_size = std::min(static_cast<size_t>(point_neighbor_num_), point_list.size()-1);
        while (neighbor_set.size() < random_neighbor_size) {
          IntIndex neighbor_id = point_list[int_rand.Random()];
          if (neighbor_id == point_id || neighbor_set.find(neighbor_id) != neighbor_set.end()) {
            continue;
          }
          neighbor_set.insert(neighbor_id);
          DistanceType dist = distance_func_(GetPoint(neighbor_id), GetPoint(point_id));
          all_point_info_[point_id].knn.insert_heap(PointDistancePairItem(neighbor_id, dist, true));
        }
      }
    }
  }

  // make heap becomes sorted array
  #pragma omp parallel for schedule(dynamic, 1)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    all_point_info_[point_id].knn.sort();
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildNavigateGraph() {

}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::FindBucketKeyPoints() {

  UpdatePointNeighborInfo();
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    BucketInfo& bucket_info = all_bucket_info_[bucket_id];
    IdList& point_list = bucket_info.point_list;
    if (point_list.empty()) {
      continue;
    }

    IdList& key_point_list = bucket_info.key_point_list;
    DynamicBitset point_pass_flag(PointSize(), 0);
    for (IntIndex point_id : point_list) {
      if (point_pass_flag[point_id]) {
        continue;
      }

      // select current point
      key_point_list.push_back(point_id);
      point_pass_flag[point_id] = 1;

      // pass neighbor and reverse neighbor
      PointInfo& point_info = all_point_info_[point_id];
      for (auto neighbor_id : point_info.new_list) {
        point_pass_flag[neighbor_id] = 1;
      }
      for (auto neighbor_id : point_info.old_list) {
        point_pass_flag[neighbor_id] = 1;
      }
      for (auto neighbor_id : point_info.reverse_new_list) {
        point_pass_flag[neighbor_id] = 1;
      }
      for (auto neighbor_id : point_info.reverse_old_list) {
        point_pass_flag[neighbor_id] = 1;
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SortBucketPointsByInDegree() {

  // count points in-degree in [0, point_neighbor_num_)
  // which could represent local density
  std::vector<int> point_in_degree_count(PointSize(), 0);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& neighbor_heap = all_point_info_[point_id].knn;
    size_t effect_size = neighbor_heap.effect_size(point_neighbor_num_);
    for (size_t i = 0; i < effect_size; i++) {
      IntIndex neighbor_id = neighbor_heap[i].id;
      #pragma omp atomic
      point_in_degree_count[neighbor_id]++;
    }
  }

  typedef util::Heap<PointDistancePair<IntIndex, int> > InDegreeHeap;
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    IdList& point_list = all_bucket_info_[bucket_id].point_list;
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, int k, 
    std::vector<std::string>& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }

  IntCode bucket_code = encoder_ptr_->Encode(query);

  /*
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
    GreedyFindKnnInGraph(query,
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
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GreedyFindKnnInGraph(
    const PointType& query, IntIndex start_point_id, PointNeighbor& k_candidates_heap,
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
    PointNeighbor& point_neighbor = all_point_info_[cur_nearest_point].knn;
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
