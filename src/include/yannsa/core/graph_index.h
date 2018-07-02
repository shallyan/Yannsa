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

    void AddNewPoint(const std::string& key, const PointType& point_vec, int search_K);

    void SearchKnn(const PointType& query, const util::GraphSearchParameter& search_param,
                   std::vector<std::string>& search_result);

    void SaveIndex(const std::string file_path);

    void LoadIndex(const std::string file_path);

  private:
    void SearchKnn(const PointType& query, int search_K, PointNeighbor& knn_results);

    void MMRRanking(const PointType& vertex_vec, PointNeighbor& knn, size_t knn_candidate_size, double lambda);

    void Init(const util::GraphIndexParameter& index_param);

    void Clear(); 

    void BuildKnnGraphIndex(int refine_iter_num);

    void ExtractIndex(); 

    void Prune();

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
    double lambda_;

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
  lambda_ = index_param.lambda;

  IntIndex max_point_id = this->dataset_ptr_->size();
  all_point_index_.reserve(max_point_id);
  all_point_info_ = std::vector<PointInfo>(max_point_id, PointInfo(join_k_));
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LoadIndex(
    const std::string file_path) {

  util::Log("load the index from " + file_path);

  Clear();

  IntIndex total_cnt = 0;
  IntIndex max_cnt = 0;

  std::ifstream load_file(file_path, std::ios::binary);
  // magic number
  int magic_number = 0;
  load_file.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
  if (magic_number != constant::magic_number) {
    throw IndexReadError("index is corrupted!");
  }

  // meta information
  // all point number
  IntIndex point_number = 0;
  load_file.read(reinterpret_cast<char*>(&point_number), sizeof(IntIndex));

  // parameters for updating
  load_file.read(reinterpret_cast<char*>(&k_), sizeof(int));
  load_file.read(reinterpret_cast<char*>(&join_k_), sizeof(int));
  load_file.read(reinterpret_cast<char*>(&lambda_), sizeof(double));

  all_point_index_.reserve(point_number);
  all_point_index_.resize(point_number);
  IntIndex neighbor_num = 0;
  for (IntIndex point_id = 0; point_id < point_number; point_id++) {
    load_file.read(reinterpret_cast<char*>(&neighbor_num), sizeof(IntIndex));

    max_cnt = std::max(max_cnt, neighbor_num);
    total_cnt += neighbor_num;

    IdList& knn_list = all_point_index_[point_id];
    knn_list.resize(neighbor_num);

    for (size_t i = 0; i < neighbor_num; i++) {
      load_file.read(reinterpret_cast<char*>(&knn_list[i]), sizeof(IntIndex));
    }
  }

  util::Log("Point: " + std::to_string(all_point_index_.size()) + 
            " Max nn = " + std::to_string(max_cnt) +
            " Average nn = " + std::to_string(total_cnt * 1.0 / PointSize()) +
            " k = " + std::to_string(k_) + " join_k = " + std::to_string(join_k_));

  // shuffle data
  for (IntIndex i = 0; i < PointSize(); i++) {
    shuffle_point_id_list_.push_back(i);
  }

  std::random_shuffle(shuffle_point_id_list_.begin(), shuffle_point_id_list_.end());

  this->SetIndexBuiltFlag();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SaveIndex(
    const std::string file_path) {

  this->CheckIndexIsBuilt();

  util::Log("save the index to " + file_path);
  std::ofstream save_file(file_path, std::ios::binary);
  // magic number
  int magic_number = constant::magic_number;
  save_file.write(reinterpret_cast<char*>(&magic_number), sizeof(int));

  // meta information
  // all point number
  IntIndex point_number = PointSize();
  save_file.write(reinterpret_cast<char*>(&point_number), sizeof(IntIndex));

  // parameters for updating
  save_file.write(reinterpret_cast<char*>(&k_), sizeof(int));
  save_file.write(reinterpret_cast<char*>(&join_k_), sizeof(int));
  save_file.write(reinterpret_cast<char*>(&lambda_), sizeof(double));

  // neighbor_num neighbor1 neighbor2 
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    const IdList& knn_list = all_point_index_[point_id];
    IntIndex neighbor_num = knn_list.size();
    save_file.write(reinterpret_cast<char*>(&neighbor_num), sizeof(IntIndex));

    for (IntIndex neighbor_id : knn_list) {
      save_file.write(reinterpret_cast<char*>(&neighbor_id), sizeof(IntIndex));
    }
  }
  save_file.close();
}

/*
   TODO
   reserve the container and point index to reduce the memory cost
*/
template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::AddNewPoint(
    const std::string& key, const PointType& point_vec, int search_K) {

  this->CheckIndexIsBuilt();

  // find join_k_ neighbors
  PointNeighbor new_point_knn(std::max(search_K, join_k_));
  SearchKnn(point_vec, search_K, new_point_knn);

  // rank neighbors of new point and keep top-k
  MMRRanking(point_vec, new_point_knn, join_k_, lambda_);

  // insert data point into dataset 
  // insert must after search
  IntIndex new_point_id = this->dataset_ptr_->insert(key, point_vec);
  // insert new point's knn list
  int new_neighbor_num = std::max(join_k_, k_*2);
  IdList new_point_knn_list(new_neighbor_num);
  for (size_t i = 0; i < new_neighbor_num; i++) {
    new_point_knn_list[i] = new_point_knn[i].id;
    all_point_index_[new_point_knn[i].id].push_back(new_point_id);
  }
  all_point_index_.push_back(new_point_knn_list);
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param) {

  this->CheckIndexIsNotBuilt();

  Clear();

  Init(index_param);

  util::Log("compute initial neighbor candidates");
  BuildKnnGraphIndex(index_param.refine_iter_num);

  util::Log("re-rank neighbor candidates");
  Prune();

  util::Log("reverse k-diverse nearest neighbors");
  Reverse();

  ExtractIndex();

  this->SetIndexBuiltFlag();
  util::Log("complete the building of k-DNN graph");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildKnnGraphIndex(
    int refine_iter_num) {

  InitPointNeighborInfo();
  for (size_t loop = 0; loop < refine_iter_num; loop++) {
    util::Log("iteration " + std::to_string(loop));
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MMRRanking(
    const PointType& vertex_vec, PointNeighbor& knn, 
    size_t knn_candidate_size, double lambda) {

  knn_candidate_size = std::min(knn_candidate_size, knn.size());

  std::vector<double> max_cosine(knn_candidate_size, 0.0);
  std::vector<double> proximity(knn_candidate_size, 0.0);

  DistanceType max_dist = knn[0].distance;
  DistanceType min_dist = knn[0].distance;
  for (size_t i = 1; i < knn_candidate_size; i++) {
    max_dist = std::max(knn[i].distance, max_dist);
    min_dist = std::min(knn[i].distance, min_dist);
  }

  for (size_t i = 0; i < knn_candidate_size; i++) {
    double dist = static_cast<double>(knn[i].distance);
    proximity[i] = - dist / (max_dist - min_dist + constant::epsilon);
  }

  // select i-th neighbors
  for (size_t i = 1; i < k_; i++) {
    // update diversity score
    PointType added_dir = GetPoint(knn[i-1].id) - vertex_vec;
    PointType added_dir_norm = added_dir.normalized();
    for (size_t j = i; j < knn_candidate_size; j++) {
      PointType cur_dir = GetPoint(knn[j].id) - vertex_vec;
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
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Prune() {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& knn = all_point_info_[point_id].knn;
    MMRRanking(GetPoint(point_id), knn, join_k_, lambda_);
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
    const PointType& query, int search_K, PointNeighbor& knn_results) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }
    
  DynamicBitset visited_point_flag(PointSize(), 0);

  util::IntRandomGenerator int_rand(0, PointSize()-1);
  size_t random_start = int_rand.Random();
  
  for (size_t random_index = random_start; 
              random_index < random_start + search_K; random_index++) {
    IntIndex start_point_id = shuffle_point_id_list_[random_index % PointSize()];
    visited_point_flag[start_point_id] = 1;
    DistanceType dist = distance_func_(GetPoint(start_point_id), query);
    knn_results.insert(PointDistancePairItem(start_point_id, dist, true));
  }
  
  size_t start_index = 0;
  while (start_index < search_K) {
    auto& current_point = knn_results[start_index];
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

      size_t update_pos = knn_results.insert(PointDistancePairItem(neighbor_id, neighbor_dist, true));
      if (update_pos <= start_index) {
        start_index = update_pos;
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, const util::GraphSearchParameter& search_param,
    std::vector<std::string>& search_result) {

  PointNeighbor knn_results(std::max(search_param.search_K, search_param.K));
  SearchKnn(query, search_param.search_K, knn_results);

  search_result.clear();
  for (size_t i = 0; i < knn_results.effect_size(search_param.K); i++) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(knn_results[i].id));
  }
}

} // namespace core 
} // namespace yannsa

#endif
