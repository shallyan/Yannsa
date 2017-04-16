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
    typedef std::vector<IntCode> CodeList;
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

      // bucket info
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
                   const util::GraphSearchParameter& search_param,
                   std::vector<std::string>& search_result);

    void Save(const std::string file_path);
    void SaveBinary(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param,
              util::BinaryEncoderPtr<PointType>& encoder_ptr); 

    void LocalitySensitiveInit();

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

    void GetNearestBucketList(IdList& bucket_list, IntCode bucket_code, int max_num);

    void BuildBucketsKnnGraph();

    void SortBucketPointsByInDegree();

    void FindBucketKeyPoints();

    /*
    void GreedyFindKnnInGraph(const PointType& query,
                              IntIndex start_point_id, PointNeighbor& k_candidates_heap,
                              DynamicBitset& visited_point_flag);
                              */


    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);
    int UpdatePointKnn(IntIndex point1, IntIndex point2, bool join_same_bucket);

    void InitPointNeighborInfo(bool is_global);
    void UpdatePointNeighborInfo(); 
    int LocalJoin(bool is_global);

  private:
    int point_neighbor_num_;
    int max_point_neighbor_num_;

    std::vector<PointInfo> all_point_info_;
    std::vector<BucketInfo> all_bucket_info_;

    BucketCode2BucketIdMap bucket_code2id_;

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
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Init(
    const util::GraphIndexParameter& index_param,
    util::BinaryEncoderPtr<PointType>& encoder_ptr) { 

  encoder_ptr_ = encoder_ptr;
  point_neighbor_num_ = index_param.point_neighbor_num;

  max_point_neighbor_num_ = std::max(index_param.max_point_neighbor_num,
                                     point_neighbor_num_);

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_info_ = std::vector<PointInfo>(max_point_id, PointInfo(max_point_neighbor_num_));
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
      if (i < point_neighbor.size()) {
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveInit() {
  size_t max_point_id = PointSize();
  //util::IntRandomGenerator int_rand(0, max_point_id-1);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
    IntIndex bucket_id = all_point_info_[point_id].bucket_id;
    IdList& bucket_knn_list = all_bucket_info_[bucket_id].knn_list;
    for (IntIndex neighbor_bucket : bucket_knn_list) {
      IdList& point_list = all_bucket_info_[neighbor_bucket].point_list;
      if (point_list.size() > 0) {
        IntIndex selected_point_id = point_list[point_id%point_list.size()];
        all_point_info_[point_id].knn.insert_array(PointDistancePairItem(selected_point_id, 100000, true));
      }
    }
  }
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

  /*
  std::cout << "bucket size: " << BucketSize() << std::endl;
  for (int i = 0; i < BucketSize(); i++) {
    std::cout << all_bucket_info_[i].point_list.size() << " ";
  }
  std::cout << std::endl;
  */

  // construct bucket knn graph
  util::Log("before bucket knn");
  BuildBucketsKnnGraph();

  util::Log("before locality init");
  LocalitySensitiveInit();

  InitPointNeighborInfo(true);
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

    //point_info.effect_size = std::min(static_cast<size_t>(point_neighbor_num_), point_info.knn.size());
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
    
    IntIndex effect_size = point_info.effect_size;
    auto& point_neighbor = point_info.knn;
    if (point_info.is_updated) {
      int new_point_count = 0;
      for (IntIndex i = 0; i < point_neighbor.size(); i++) {
        if (point_neighbor[i].flag) {
          new_point_count++;
          if (new_point_count >= 10) {
            effect_size = i+1;
            break;
          }
        }
      }
      if (effect_size < point_neighbor_num_) {
        effect_size = std::min(point_neighbor.size(), static_cast<size_t>(point_neighbor_num_));
      }
    }
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

// LocalJoin is proposed by NNDescent(KGraph)
// Many thanks
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
        cur_update_count += UpdatePointKnn(p1, p2, !is_global);
      }

      for (size_t j = 0; j < old_list.size(); j++) {
        IntIndex p2 = old_list[j];
        cur_update_count += UpdatePointKnn(p1, p2, !is_global);
      }
    }
    update_count += cur_update_count;
    point_info.is_updated = cur_update_count > 0;
  }
  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::UpdatePointKnn(
    IntIndex point1, IntIndex point2, bool join_same_bucket) {

  if (point1 == point2) {
    return 0;
  }

  /*
  if (!join_same_bucket && (all_point_info_[point1].bucket_id == all_point_info_[point2].bucket_id)) {
    return 0;
  }
  */

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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetNearestBucketList(
    IdList& bucket_list, IntCode bucket_code, int max_num) {

  bucket_list.clear();
  int code_length = encoder_ptr_->code_length();
  // use 1, 11 and 101 as mask
  for (int dist = 0, init_mask = 1; dist < 3; dist++, init_mask += 2) {
    int mask = init_mask;
    for (int i = 0; i < code_length-dist; i++) {
      IntCode neighbor_bucket_code = bucket_code ^ mask;
      mask <<= 1;

      BucketCode2BucketIdMap::const_iterator neighbor_bucket_iter = bucket_code2id_.find(neighbor_bucket_code);
      if (neighbor_bucket_iter != bucket_code2id_.end()) {
        bucket_list.push_back(neighbor_bucket_iter->second);
      }

      if (bucket_list.size() >= max_num) {
        return;
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalitySensitiveHashing() {

  // encode
  CodeList point_code_list(PointSize());
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildBucketsKnnGraph() {

  int code_length = encoder_ptr_->code_length();
  #pragma omp parallel for schedule(static)
  for (IntIndex bucket_id = 0; bucket_id < BucketSize(); bucket_id++) {
    BucketInfo& bucket_info = all_bucket_info_[bucket_id];
    GetNearestBucketList(bucket_info.knn_list, bucket_info.code, code_length);
  }
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
    std::random_shuffle(key_point_list.begin(), key_point_list.end());
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
    const PointType& query,
    const util::GraphSearchParameter& search_param,
    std::vector<std::string>& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }

  IntCode bucket_code = encoder_ptr_->Encode(query);

  // if bucket not exist, use nearest bucket 
  IntIndex bucket_id = 0;
  if (bucket_code2id_.find(bucket_code) != bucket_code2id_.end()) {
    bucket_id = bucket_code2id_[bucket_code];
  }
  else {
    IdList bucket_list;
    GetNearestBucketList(bucket_list, bucket_code, 1);
    if (!bucket_list.empty()) {
      bucket_id = bucket_list[0];
    }
  }

  // search from start points
  DynamicBitset visited_point_flag(PointSize(), 0);
  const IdList& key_list = all_bucket_info_[bucket_id].key_point_list;
  IdList start_list(key_list.begin(), key_list.end());
  if (start_list.size() < search_param.search_start_point_num) {
    const IdList& knn_bucket = all_bucket_info_[bucket_id].knn_list;
    for (auto neighbor_bucket : knn_bucket) {
      const IdList& neighbor_bucket_key = all_bucket_info_[neighbor_bucket].key_point_list;
      start_list.insert(start_list.end(), neighbor_bucket_key.begin(), neighbor_bucket_key.end());
      if (start_list.size() >= search_param.search_start_point_num) {
        break;
      }
    }
  }
  if (start_list.size() > search_param.search_start_point_num) {
    start_list.resize(search_param.search_start_point_num);
  }
  
  PointNeighbor result_candidates(search_param.search_k);
  for (auto start_point_id : start_list) {
    DistanceType dist = distance_func_(GetPoint(start_point_id), query);
    result_candidates.insert_array(PointDistancePairItem(start_point_id, dist));

    /*
    PointNeighbor tmp_result_candidates(search_param.search_k);
    GreedyFindKnnInGraph(query,
                         start_point_id, tmp_result_candidates,
                         visited_point_flag);
    for (auto iter = tmp_result_candidates.begin();
              iter != tmp_result_candidates.end(); iter++) {
      result_candidates.insert_array(*iter);
    }
    */
  }

  DynamicBitset pass_point_flag(PointSize(), 0);
  size_t start_index = 0; 
  while (true) {
    IdList next_point_list;
    size_t i, count = 0;
    for (i = start_index; i < result_candidates.size(); i++) {
      IntIndex point_id = result_candidates[i].id;
      if (pass_point_flag[point_id]) {
        continue;
      }
      pass_point_flag[point_id] = 1;
      PointNeighbor& knn = all_point_info_[point_id].knn;
      size_t effect_size = knn.effect_size(point_neighbor_num_);
      for (size_t j = 0; j < effect_size; j++) {
        if (!visited_point_flag[knn[j].id]) {
          next_point_list.push_back(knn[j].id);
        }
      }
      count++;
      if (count > search_param.k) {
        break;
      }
    }

    start_index = i;

    for (auto point : next_point_list) {
      visited_point_flag[point] = 1;
      DistanceType dist = distance_func_(GetPoint(point), query);
      PointDistancePairItem point_dist(point, dist);
      size_t update_pos = result_candidates.insert_array(point_dist);
      start_index = std::min(update_pos, start_index);
    }

    if (start_index >= result_candidates.size()) {
      break;
    }
  }
  
  search_result.clear();
  for (int i = 0; i < result_candidates.effect_size(search_param.k); i++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(result_candidates[i].id));
  }
}

/*
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
    size_t neighbor_search_num = is_last_traverse ? point_neighbor_num_ : search_point_neighbor_num_;
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
*/

} // namespace core 
} // namespace yannsa

#endif
