#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/base/constant_definition.h"
#include "yannsa/util/sorted_array.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/logging.h"
#include "yannsa/util/lock.h"
#include "yannsa/util/random_generator.h"
#include "yannsa/core/base_index.h"
#include <omp.h>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

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
    typedef util::SortedArray<PointDistancePairItem> PointNeighbor;

  private:
    struct PointInfo {
      // graph
      PointNeighbor knn;

      // for local join
      IdList old_list;
      IdList new_list;
      IdList reverse_old_list;
      IdList reverse_new_list;
      DistanceType radius;
      size_t effect_size;

      util::Mutex lock;

      PointInfo(int k) {
        knn = PointNeighbor(k);
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

    void Build(const util::GraphIndexParameter& index_param); 

    void SearchKnn(const PointType& query,
                   const util::GraphSearchParameter& search_param,
                   std::vector<std::string>& search_result);

    void SaveIndex(const std::string file_path);

    void LoadIndex(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param);
    void Clear(); 

    void BuildKnnGraphIndex(const util::GraphIndexParameter& index_param); 

    void ExtractIndex(); 

    void Prune(double lambda);

    void Reverse();

    inline const PointType& GetPoint(IntIndex point_id) {
      return (*this->dataset_ptr_)[point_id];
    }
    
    inline IntIndex PointSize() {
      return this->dataset_ptr_->size();
    }

    int UpdatePointKnn(IntIndex point1, IntIndex point2);

    void InitPointNeighborInfo();
    void UpdatePointNeighborInfo(); 
    int LocalJoin();

  private:
    // index
    int k_;
    int join_k_;

    std::vector<PointInfo> all_point_info_;

    // search
    IdList shuffle_point_id_list_;
    std::vector<IdList> all_point_index_;

    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Clear() {
  this->have_built_ = false;
  // point
  all_point_info_.clear();
  all_point_index_.clear();
  shuffle_point_id_list_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Init(
    const util::GraphIndexParameter& index_param) {

  k_ = index_param.k;
  join_k_ = std::max(index_param.join_k, k_);

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_info_ = std::vector<PointInfo>(max_point_id, PointInfo(join_k_));
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LoadIndex(
    const std::string file_path) {

  Clear();

  size_t total_cnt = 0;
  size_t max_cnt = 0;

  std::ifstream load_file(file_path);
  std::string buff;
  while (std::getline(load_file, buff)) {
    std::stringstream point_info_stream(buff);
    size_t neighbor_num;
    point_info_stream >> neighbor_num;

    max_cnt = std::max(max_cnt, neighbor_num);
    total_cnt += neighbor_num;

    IdList knn_list;
    knn_list.reserve(neighbor_num);
    IntIndex point_id;
    for (size_t i = 0; i < neighbor_num; i++) {
      point_info_stream >> point_id;
      knn_list.push_back(point_id);
    }

    all_point_index_.push_back(knn_list);
  }

  std::cout << "Point num: " << all_point_index_.size() << std::endl;
  std::cout << "Max knn: " << max_cnt << std::endl;
  std::cout << "Average knn: " << total_cnt*1.0/PointSize() << std::endl;

  // shuffle data
  for (IntIndex i = 0; i < PointSize(); i++) {
    shuffle_point_id_list_.push_back(i);
  }

  std::random_shuffle(shuffle_point_id_list_.begin(), shuffle_point_id_list_.end());

  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SaveIndex(
    const std::string file_path) {

  std::ofstream save_file(file_path);
  // knn_size neighbor1 neighbor2 
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    const IdList& knn_list = all_point_index_[point_id];
    save_file << knn_list.size() << " ";
    for (auto neighbor_id : knn_list) {
      save_file << neighbor_id << " ";
    }
    save_file << std::endl;
  }
  save_file.close();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param) {

  this->CheckIndexIsNotBuilt();

  util::Log("before build");

  Clear();

  BuildKnnGraphIndex(index_param);
  Prune(index_param.lambda);
  Reverse();

  ExtractIndex();

  this->SetIndexBuiltFlag();
  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildKnnGraphIndex(
    const util::GraphIndexParameter& index_param) {

  Init(index_param);

  util::Log("before init");
  InitPointNeighborInfo();

  util::Log("before refine");
  for (size_t loop = 0; loop < index_param.refine_iter_num; loop++) {
    util::Log(std::to_string(loop) + " iteration ");
    UpdatePointNeighborInfo();
    LocalJoin();
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ExtractIndex() {

  all_point_index_.resize(PointSize());

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& neighbor = all_point_info_[point_id].knn;

    IdList& knn_list = all_point_index_[point_id];
    knn_list.resize(neighbor.size());

    for (size_t i = 0; i < neighbor.size(); i++) {
      knn_list[i] = neighbor[i].id;
    }
  }

  // clear old knn graph 
  all_point_info_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Prune(double lambda) {

  size_t knn_candidate_size = join_k_;
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& knn = all_point_info_[point_id].knn;

    std::vector<double> max_cosine(knn_candidate_size, 0.0);
    std::vector<double> proximity(knn_candidate_size, 0.0);

    DistanceType max_dist = knn[0].distance;
    DistanceType min_dist = knn[0].distance;
    for (size_t i = 1; i < knn_candidate_size; i++) {
      max_dist = std::max(knn[i].distance, max_dist);
      min_dist = std::min(knn[i].distance, min_dist);
    }

    for (size_t i = 0; i < knn_candidate_size; i++) {
      // for cosine, distance = - similarity
      double dist = static_cast<double>(knn[i].distance);
      proximity[i] = - dist / (max_dist - min_dist + constant::epsilon);
    }

    // select i-th neighbors
    for (size_t i = 1; i < k_; i++) {
      // update diversity score
      PointType added_dir = GetPoint(knn[i-1].id) - GetPoint(point_id);
      PointType added_dir_norm = added_dir.normalized();
      for (size_t j = i; j < knn_candidate_size; j++) {
        PointType cur_dir = GetPoint(knn[j].id) - GetPoint(point_id);
        PointType cur_dir_norm = cur_dir.normalized();
        max_cosine[j] = std::max(max_cosine[j], static_cast<double>(cur_dir_norm.dot(added_dir_norm)));
      }

      // select max mmr
      double max_mmr = 0.0;
      int max_mmr_id = -1;
      for (size_t j = i; j < knn_candidate_size; j++) {
        double mmr = lambda * proximity[j] + (1.0 - lambda) * 0.5 * (-max_cosine[j]);
        if (max_mmr_id == -1 || max_mmr < mmr) {
          max_mmr_id = j;
          max_mmr = mmr;
        }
      }

      std::swap(knn[max_mmr_id], knn[i]);
      std::swap(max_cosine[max_mmr_id], max_cosine[i]);
      std::swap(proximity[max_mmr_id], proximity[i]);
    }

    knn.remax_size(k_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Reverse() {

  // reverse
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    for (size_t i = 0; i < k_; i++) {
      IntIndex neighbor_id = point_neighbor[i].id;
      PointNeighbor& neighbor = all_point_info_[neighbor_id].knn;
      PointDistancePairItem reverse_neighbor(point_id, point_neighbor[i].distance, true);
      neighbor.parallel_push(reverse_neighbor);
    }
  }

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    point_neighbor.unique(k_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::InitPointNeighborInfo() {

  size_t max_point_id = PointSize();
  util::IntRandomGenerator int_rand(0, max_point_id-1);
  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
    PointInfo& point_info = all_point_info_[point_id];
    size_t random_neighbor_size = std::min(static_cast<size_t>(k_), max_point_id-1);
    IdSet neighbor_set;
    while (neighbor_set.size() < random_neighbor_size) {
      IntIndex neighbor_id = int_rand.Random();
      if (neighbor_id == point_id || neighbor_set.find(neighbor_id) != neighbor_set.end()) {
        continue;
      }
      neighbor_set.insert(neighbor_id);
      DistanceType dist = distance_func_(GetPoint(neighbor_id), GetPoint(point_id));
      point_info.knn.insert(PointDistancePairItem(neighbor_id, dist, true));
    }

    point_info.effect_size = std::min(static_cast<size_t>(k_), point_info.knn.size());
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
    auto& knn = point_info.knn;
    size_t max_effect_size = std::min(knn.size(), static_cast<size_t>(join_k_));
    IntIndex effect_size = max_effect_size;
    int new_point_count = 0;
    for (IntIndex i = 0; i < max_effect_size; i++) {
      if (knn[i].flag) {
        new_point_count++;
        if (new_point_count >= k_) {
          effect_size = i+1;
          break;
        }
      }
    }
    point_info.reset(effect_size);
  }

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointInfo& point_info = all_point_info_[point_id];

    auto& knn = point_info.knn;
    IntIndex effect_size = knn.effect_size(point_info.effect_size);
    for (IntIndex i = 0; i < effect_size; i++) {
      auto& neighbor = knn[i];
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

  int update_count = 0;
  int update_pos = all_point_info_[point1].knn.parallel_insert(PointDistancePairItem(point2, dist, true));
  update_count += update_pos < k_ ? 1 : 0;

  update_pos = all_point_info_[point2].knn.parallel_insert(PointDistancePairItem(point1, dist, true));
  update_count += update_pos < k_ ? 1 : 0;

  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, const util::GraphSearchParameter& search_param,
    std::vector<std::string>& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }
    
  DynamicBitset visited_point_flag(PointSize(), 0);

  util::IntRandomGenerator int_rand(0, PointSize()-1);
  size_t random_start = int_rand.Random();

  PointNeighbor result_candidates(std::max(search_param.search_K, search_param.K));
  
  for (size_t random_index = random_start; 
              random_index < random_start + search_param.search_K; random_index++) {
    IntIndex start_point_id = shuffle_point_id_list_[random_index % PointSize()];
    visited_point_flag[start_point_id] = 1;
    DistanceType dist = distance_func_(GetPoint(start_point_id), query);
    result_candidates.insert(PointDistancePairItem(start_point_id, dist, true));
  }
  
  size_t start_index = 0;
  while (start_index < search_param.search_K) {
    auto& current_point = result_candidates[start_index];
    if (current_point.flag == false) {
      start_index++;
      continue;
    }

    current_point.flag = false;
    const IdList& knn_list = all_point_index_[current_point.id];

    for (size_t i = 0; i < knn_list.size(); i++) {
      IntIndex neighbor_id = knn_list[i];
      if (visited_point_flag[neighbor_id]) {
        continue;
      }
      visited_point_flag[neighbor_id] = 1;
      DistanceType neighbor_dist = distance_func_(GetPoint(neighbor_id), query);

      size_t update_pos = result_candidates.insert(PointDistancePairItem(neighbor_id, neighbor_dist, true));
      if (update_pos <= start_index) {
        start_index = update_pos;
      }
    }
  }
    
  search_result.clear();
  for (size_t i = 0; i < result_candidates.effect_size(search_param.K); i++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(result_candidates[i].id));
  }
}

} // namespace core 
} // namespace yannsa

#endif
