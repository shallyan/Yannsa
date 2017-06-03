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
    // point
    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    typedef util::Heap<PointDistancePairItem> PointNeighbor;

  private:
    struct PointInfo {
      // graph
      PointNeighbor knn;

      // bucket info
      IntIndex bucket_id;

      // status
      bool is_updated;

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

    void Build(const util::GraphIndexParameter& index_param); 

    void SearchKnn(const PointType& query,
                   const util::GraphSearchParameter& search_param,
                   std::vector<std::string>& search_result);

    void Save(const std::string file_path);
    void SaveBinary(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param);

    inline const PointType& GetPoint(IntIndex point_id) {
      return (*this->dataset_ptr_)[point_id];
    }
    
    inline IntIndex PointSize() {
      return all_point_info_.size();
    }

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);
    int UpdatePointKnn(IntIndex point1, IntIndex point2);

    void InitPointNeighborInfo();
    void UpdatePointNeighborInfo(); 
    int LocalJoin();

  private:
    int point_neighbor_num_;
    int max_point_neighbor_num_;

    std::vector<PointInfo> all_point_info_;
    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Clear() {
  this->have_built_ = false;
  // point
  all_point_info_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Init(
    const util::GraphIndexParameter& index_param) {

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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param) {

  util::Log("before build");

  if (this->have_built_) {
    throw IndexBuildError("Graph index has already been built!");
  }

  Init(index_param);

  util::Log("before init");
  InitPointNeighborInfo();

  util::Log("before refine");
  for (int loop = 0; loop < index_param.refine_iter_num; loop++) {
    clock_t s, e, e1;
    s = clock();
    UpdatePointNeighborInfo();
    e = clock();
    std::cout << "refine init: " << (e-s)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
    LocalJoin();
    e1 = clock();
    std::cout << "refine update: " << (e1-e)*1.0 / CLOCKS_PER_SEC << "s" << std::endl;
  }

  // build
  this->have_built_ = true;

  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::InitPointNeighborInfo() {
  size_t max_point_id = PointSize();
  util::IntRandomGenerator int_rand(0, max_point_id-1);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
    PointInfo& point_info = all_point_info_[point_id];
    size_t random_neighbor_size = std::min(static_cast<size_t>(point_neighbor_num_), max_point_id-1);
    IdSet neighbor_set;
    while (neighbor_set.size() < random_neighbor_size) {
      IntIndex neighbor_id = int_rand.Random();
      if (neighbor_id == point_id || neighbor_set.find(neighbor_id) != neighbor_set.end()) {
        continue;
      }
      neighbor_set.insert(neighbor_id);
      DistanceType dist = distance_func_(GetPoint(neighbor_id), GetPoint(point_id));
      point_info.knn.insert_array(PointDistancePairItem(neighbor_id, dist, true));
    }

    point_info.is_updated = false;
    point_info.effect_size = std::min(static_cast<size_t>(point_neighbor_num_), point_info.knn.size());
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
    IntIndex effect_size = point_info.effect_size;
    auto& point_neighbor = point_info.knn;
    if (point_info.is_updated) {
      int new_point_count = 0;
      for (IntIndex i = 0; i < point_neighbor.size(); i++) {
        if (point_neighbor[i].flag) {
          new_point_count++;
          if (new_point_count >= point_neighbor_num_/2) {
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
int GraphIndex<PointType, DistanceFuncType, DistanceType>::LocalJoin() {

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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query,
    const util::GraphSearchParameter& search_param,
    std::vector<std::string>& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }

  /*
  DynamicBitset visited_point_flag(PointSize(), 0);

  const BucketInfo& bucket_info = all_bucket_info_[bucket_id];
  const IdList& bucket_point_list = bucket_info.point_list;
  size_t bucket_point_effect_size = std::min(bucket_point_list.size(), 
                                             static_cast<size_t>(search_param.search_start_point_num));
  IdList start_list(bucket_point_list.begin(), bucket_point_list.begin()+bucket_point_effect_size);
  if (start_list.size() < search_param.search_start_point_num) {
    const IdList& bucket_knn = bucket_info.knn_list;
    for (auto neighbor_bucket : bucket_knn) {
      const IdList& neighbor_bucket_point_list = all_bucket_info_[neighbor_bucket].point_list;
      start_list.insert(start_list.end(), neighbor_bucket_point_list.begin(), neighbor_bucket_point_list.end());
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
  */
  
  /*
  PointNeighbor result_candidates(search_param.k);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    size_t indegree = all_point_info_[point_id].in_degree;
    if (indegree < 3) {
      DistanceType dist = distance_func_(GetPoint(point_id), query);
      PointDistancePairItem point_dist(point_id, dist);
      size_t update_pos = result_candidates.insert_array(point_dist);
    }
  }
  */
  /*
  search_result.clear();
  for (int i = 0; i < result_candidates.effect_size(search_param.k); i++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(result_candidates[i].id));
  }
  */
}

} // namespace core 
} // namespace yannsa

#endif
