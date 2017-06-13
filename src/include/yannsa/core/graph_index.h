#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/base/error_definition.h"
#include "yannsa/util/sorted_array.h"
#include "yannsa/util/parameter.h"
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
    struct PointIndex {
      PointNeighbor knn;
      IdList next;
      // tmp
      std::vector<DistanceType> next_dist;

      PointIndex(int point_neighbor_num) {
        knn = PointNeighbor(point_neighbor_num);
      }
    };

    struct PointInfo {
      // graph
      PointNeighbor knn;

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

    void BuildKnnGraphIndex(const util::GraphIndexParameter& index_param); 

    void ExtractIndex(); 

    void BuildExtendIndex2(const util::GraphSearchParameter& extend_index_param); 

    void BuildExtendIndex(const util::GraphSearchParameter& extend_index_param); 

    void Prune();

    void Reverse();

    int SearchKnn(const PointType& query,
                   const util::GraphSearchParameter& search_param,
                   std::vector<std::string>& search_result,
                   bool is_extend=false);

    int inSearchKnn(const PointType& query,
                   IntIndex start_point_id,
                   const util::GraphSearchParameter& search_param,
                   IdList& search_result);

    int SearchKnn(const PointType& query,
                   const util::GraphSearchParameter& search_param,
                   IdList& search_result,
                   bool is_extend=false);

    void SaveIndex(const std::string file_path);
    void LoadIndex(const std::string file_path);

    void SaveKnnGraph(const std::string file_path);
    void SaveBinary(const std::string file_path);

  private:
    void Init(const util::GraphIndexParameter& index_param);

    inline const PointType& GetPoint(IntIndex point_id) {
      return (*this->dataset_ptr_)[point_id];
    }
    
    inline IntIndex PointSize() {
      return this->dataset_ptr_->size();
    }

    int UpdatePointKnn(IntIndex point1, IntIndex point2, DistanceType dist);
    int UpdatePointKnn(IntIndex point1, IntIndex point2);

    void InitPointNeighborInfo();
    void UpdatePointNeighborInfo(); 
    int LocalJoin();

  private:
    // index
    int point_neighbor_num_;
    int max_point_neighbor_num_;

    std::vector<PointInfo> all_point_info_;

    // search
    IdList shuffle_point_id_list_;
    std::vector<PointIndex> all_point_index_;

    DistanceFuncType distance_func_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Clear() {
  this->have_built_ = false;
  // point
  all_point_info_.clear();
  shuffle_point_id_list_.clear();
  all_point_index_.clear();
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
    PointNeighbor& point_neighbor = all_point_index_[point_id].knn;
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
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SaveKnnGraph(
    const std::string file_path) {

  std::ofstream save_file(file_path);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    save_file << this->dataset_ptr_->GetKeyById(point_id) << " ";
    PointNeighbor& point_neighbor = all_point_index_[point_id].knn;
    size_t effect_size = point_neighbor.effect_size(point_neighbor_num_);
    for (size_t i = 0; i < effect_size; i++) {
      save_file << this->dataset_ptr_->GetKeyById(point_neighbor[i].id) << " "; 
    }
    save_file << std::endl;
  }
  save_file.close();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::LoadIndex(
    const std::string file_path) {

  Clear();

  size_t total_cnt = 0;
  size_t max_cnt = 0;
  size_t next_cnt = 0;

  std::ifstream load_file(file_path);
  std::string buff;
  while (std::getline(load_file, buff)) {
    std::stringstream point_info_stream(buff);
    size_t neighbor_num;
    point_info_stream >> neighbor_num;

    max_cnt = std::max(max_cnt, neighbor_num);
    total_cnt += neighbor_num;

    PointIndex point_index(neighbor_num);
    IntIndex point_id;
    DistanceType dist;
    for (size_t i = 0; i < neighbor_num; i++) {
      point_info_stream >> point_id >> dist; 
      point_index.knn.push(PointDistancePairItem(point_id, dist));
    }

    size_t next_num;
    point_info_stream >> next_num;
    IntIndex next_id;
    for (size_t i = 0; i < next_num; i++) {
      point_info_stream >> next_id; 
      point_index.next.push_back(next_id);

      // discard dist
      point_info_stream >> dist; 
    }
    next_cnt += point_index.next.size();

    all_point_index_.push_back(point_index);
  }

  std::cout << "Point num: " << all_point_index_.size() << std::endl;
  std::cout << "Max knn: " << max_cnt << std::endl;
  std::cout << "Average knn: " << total_cnt*1.0/PointSize() << std::endl;
  std::cout << "Next num: " << next_cnt*1.0/PointSize() << std::endl;

  // shuffle data
  for (IntIndex i = 0; i < PointSize(); i++) {
    shuffle_point_id_list_.push_back(i);
  }

  // fix shuffle
  std::srand(1);
  for (int i = 1; i < PointSize(); i++) {
    std::swap(shuffle_point_id_list_[i], shuffle_point_id_list_[std::rand() % (i+1)]);
  }
  //std::random_shuffle(shuffle_point_id_list_.begin(), shuffle_point_id_list_.end());

  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SaveIndex(
    const std::string file_path) {

  // point knn_size neighbor1 dist1 neighbor2 dist2 extend_size id1 id2 id3
  std::ofstream save_file(file_path);
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_index_[point_id].knn;
    save_file << point_neighbor.size() << " ";
    for (size_t i = 0; i < point_neighbor.size(); i++) {
      save_file << point_neighbor[i].id << " "
                << point_neighbor[i].distance << " ";
    }

    IdList& next = all_point_index_[point_id].next;
    save_file << next.size() << " ";
    for (size_t i = 0; i < next.size(); i++) {
      save_file << next[i] << " "
                << all_point_index_[point_id].next_dist[i] << " ";
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

  // build basic knn graph
  BuildKnnGraphIndex(index_param);

  // build extend knn graph
  // BuildExtendIndex();
  
  this->have_built_ = true;
  util::Log("end build");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildExtendIndex2(
    const util::GraphSearchParameter& extend_index_param) {

  util::Log("before build extend index");

  point_neighbor_num_ = 10;

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& knn = all_point_index_[point_id].knn;
    const PointType& cur_point = GetPoint(point_id); 
    for (size_t i = 0; i < knn.size()/*point_neighbor_num_*/; i++) {
      const PointType& target_point = GetPoint(knn[i].id);
      PointType direction = target_point - cur_point;
      auto direction_norm = direction.normalized();

      PointNeighbor& target_knn = all_point_index_[knn[i].id].knn;
      DistanceType n_dist; 
      IntIndex n_id = -1;
      for (size_t j = 0; j < target_knn.size()/*point_neighbor_num_*/; j++) {
        PointType candidate_direction = GetPoint(target_knn[j].id) - target_point;
        // cal cosine
        auto candidate_direction_norm = candidate_direction.normalized();
        DistanceType dist = direction_norm.dot(candidate_direction_norm);
        if (n_id == -1 || n_dist < dist) {
          n_id = target_knn[j].id;
          n_dist = dist;
        }
      }

      all_point_index_[point_id].next.push_back(n_id);
      all_point_index_[point_id].next_dist.push_back(n_dist);
    }
  }

  util::Log("end build extend index");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildExtendIndex(
    const util::GraphSearchParameter& extend_index_param) {

  util::Log("before build extend index");

  point_neighbor_num_ = 10;

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& knn = all_point_index_[point_id].knn;
    const PointType& cur_point = GetPoint(point_id); 
    for (size_t i = 0; i < point_neighbor_num_; i++) {
      IdList search_result;
      DistanceType dist = 0;
      const PointType& target_point = GetPoint(knn[i].id);
      //for (int step = 2; step <= 32; step *= 2) {
        int step = 2;
        PointType next_point = cur_point + step * (target_point - cur_point);
        search_result.clear();
        inSearchKnn(next_point, knn[i].id, extend_index_param, search_result);
        // check this point
        //IntIndex next_nn = search_result[0];
        //dist = distance_func_(cur_point, GetPoint(next_nn));
        //if (dist >= knn[i].distance*2.0) {
        //  break;
        //}
      //}

      for (IntIndex next_candidate : search_result) {
        if (next_candidate != knn[i].id) {
          //DistanceType dist = distance_func_(cur_point, GetPoint(next_candidate));
          all_point_index_[point_id].next.push_back(next_candidate);

          PointType direction = GetPoint(next_candidate) - target_point;
          auto direction_norm = direction.normalized();

          PointType direction1 = target_point - cur_point;
          auto direction_norm1 = direction1.normalized();
          DistanceType dist = direction_norm1.dot(direction_norm);

          all_point_index_[point_id].next_dist.push_back(dist);
          break;
        }
      }
    }
  }

  util::Log("end build extend index");
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildKnnGraphIndex(
    const util::GraphIndexParameter& index_param) {

  Clear();

  Init(index_param);

  util::Log("before init");
  InitPointNeighborInfo();

  util::Log("before refine");
  for (int loop = 0; loop < index_param.refine_iter_num; loop++) {
    util::Log(std::to_string(loop) + " iteration ");
    UpdatePointNeighborInfo();
    LocalJoin();
  }

  // keep point_neighbor_num nn graph
  Prune();

  // reverse 
  Reverse();

  // keep index
  ExtractIndex();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::ExtractIndex() {

  all_point_index_.clear();

  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& neighbor = all_point_info_[point_id].knn;
    PointIndex point_index(neighbor.size());

    for (size_t i = 0; i < neighbor.size(); i++) {
      point_index.knn.push(PointDistancePairItem(neighbor[i].id, neighbor[i].distance, true));
    }

    all_point_index_.push_back(point_index);
  }

  // clear old knn graph 
  all_point_info_.clear();
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Prune() {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    point_neighbor.remax_size(point_neighbor_num_);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Reverse() {

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    for (size_t i = 0; i < point_neighbor_num_; i++) {
      IntIndex neighbor_id = point_neighbor[i].id;
      PointNeighbor& neighbor = all_point_info_[neighbor_id].knn;
      PointDistancePairItem reverse_neighbor(point_id, point_neighbor[i].distance, true);
      neighbor.parallel_push(reverse_neighbor);
    }
  }

  #pragma omp parallel for schedule(static)
  for (IntIndex point_id = 0; point_id < PointSize(); point_id++) {
    PointNeighbor& point_neighbor = all_point_info_[point_id].knn;
    point_neighbor.unique();
  }
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
      point_info.knn.insert(PointDistancePairItem(neighbor_id, dist, true));
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
          if (new_point_count >= point_neighbor_num_) {
            effect_size = i+1;
            break;
          }
        }
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
  int update_pos = all_point_info_[point1].knn.parallel_insert(PointDistancePairItem(point2, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;

  update_pos = all_point_info_[point2].knn.parallel_insert(PointDistancePairItem(point1, dist, true));
  update_count += update_pos < point_neighbor_num_ ? 1 : 0;

  return update_count;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::inSearchKnn(
    const PointType& query,
    IntIndex start_point_id,
    const util::GraphSearchParameter& search_param,
    IdList& search_result) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }
    
  int num_cnt = 0;
  DynamicBitset visited_point_flag(PointSize(), 0);

  PointNeighbor result_candidates(search_param.search_k);

  visited_point_flag[start_point_id] = 1;
  DistanceType dist = distance_func_(GetPoint(start_point_id), query);
  num_cnt++;
  result_candidates.insert(PointDistancePairItem(start_point_id, dist, true));

  util::IntRandomGenerator int_rand(0, PointSize()-1);
  size_t random_start = int_rand.Random();
  for (size_t random_index = random_start; 
              random_index < random_start + search_param.search_k; random_index++) {
    IntIndex start_point_id = shuffle_point_id_list_[random_index % PointSize()];
    if (visited_point_flag[start_point_id]) {
      continue;
    }
    visited_point_flag[start_point_id] = 1;
    DistanceType dist = distance_func_(GetPoint(start_point_id), query);
    num_cnt++;
    result_candidates.insert(PointDistancePairItem(start_point_id, dist, true));
  }

  size_t start_index = 0;
  while (start_index < search_param.search_k) {
    auto& current_point = result_candidates[start_index];
    if (current_point.flag == false) {
      start_index++;
      continue;
    }

    PointNeighbor& knn = all_point_index_[current_point.id].knn;
    size_t first_range = current_point.m;
    size_t last_range = current_point.m + search_param.start_neighbor_num; 
    if (last_range >= knn.size()) {
      last_range = knn.size();
      // has deal with this point's neighbor
      current_point.flag = false;
    }
    current_point.m = last_range;
    for (size_t i = first_range; i < last_range; i++) {
      if (visited_point_flag[knn[i].id]) {
        continue;
      }
      visited_point_flag[knn[i].id] = 1;
      DistanceType dist = distance_func_(GetPoint(knn[i].id), query);
      num_cnt++;
      size_t update_pos = result_candidates.insert(PointDistancePairItem(knn[i].id, dist, true));
      if (update_pos <= start_index) {
        start_index = update_pos;
      }
    }
  }

  search_result.clear();
  for (size_t i = 0; i < result_candidates.effect_size(search_param.k); i++) {
    search_result.push_back(result_candidates[i].id);
  }

  return num_cnt;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, const util::GraphSearchParameter& search_param,
    IdList& search_result, bool is_extend) {

  if (!this->have_built_) {
    throw IndexBuildError("Graph index hasn't been built!");
  }
    
  int num_cnt = 0;
  DynamicBitset visited_point_flag(PointSize(), 0);

  util::IntRandomGenerator int_rand(0, PointSize()-1);
  size_t random_start = int_rand.Random();

  PointNeighbor result_candidates(search_param.search_k);
  for (size_t random_index = random_start; 
              random_index < random_start + search_param.search_k; random_index++) {
    IntIndex start_point_id = shuffle_point_id_list_[random_index % PointSize()];
    visited_point_flag[start_point_id] = 1;
    DistanceType dist = distance_func_(GetPoint(start_point_id), query);
    num_cnt++;
    result_candidates.insert(PointDistancePairItem(start_point_id, dist, true));
  }

  point_neighbor_num_ = 10;

  size_t start_index = 0;
  while (start_index < search_param.search_k) {
    auto& current_point = result_candidates[start_index];
    if (current_point.flag == false) {
      start_index++;
      continue;
    }

    PointNeighbor& knn = all_point_index_[current_point.id].knn;
    size_t first_range = current_point.m;
    size_t last_range = current_point.m + search_param.start_neighbor_num; 
    if (last_range >= knn.size()) {
      last_range = knn.size();
      // has deal with this point's neighbor
      current_point.flag = false;
    }
    current_point.m = last_range;

    // keep current point id because it may be replaced
    IntIndex cur_point_id = current_point.id;
    DistanceType min_dist = current_point.distance;
    IntIndex next_point_id = -1;
    for (size_t i = first_range; i < last_range; i++) {
      if (visited_point_flag[knn[i].id]) {
        continue;
      }
      visited_point_flag[knn[i].id] = 1;
      DistanceType neighbor_dist = distance_func_(GetPoint(knn[i].id), query);
      num_cnt++;
      size_t update_pos = result_candidates.insert(PointDistancePairItem(knn[i].id, neighbor_dist, true));
      if (update_pos <= start_index) {
        start_index = update_pos;
      }

      if (is_extend && i < point_neighbor_num_) {
        // this neighbor direction can update 
        if (neighbor_dist < min_dist) {
          min_dist = neighbor_dist;
          next_point_id = all_point_index_[cur_point_id].next[i]; 
        }
        
        /*
        if (neighbor_dist < min_dist) {
          min_dist = neighbor_dist;
          IntIndex next_point_id = all_point_index_[cur_point_id].next[i]; 
          if (visited_point_flag[next_point_id]) {
            continue;
          }
          visited_point_flag[next_point_id] = 1;
          DistanceType next_dist = distance_func_(GetPoint(next_point_id), query);
          num_cnt++;
          size_t update_pos = result_candidates.insert(PointDistancePairItem(next_point_id, next_dist, true));
          if (update_pos <= start_index) {
            start_index = update_pos;
          }
        }
        */
      }
    }
    
    if (next_point_id == -1) continue;
    if (visited_point_flag[next_point_id]) {
      continue;
    }
    visited_point_flag[next_point_id] = 1;
    DistanceType next_dist = distance_func_(GetPoint(next_point_id), query);
    num_cnt++;
    size_t update_pos = result_candidates.insert(PointDistancePairItem(next_point_id, next_dist, true));
    if (update_pos <= start_index) {
      start_index = update_pos;
    }
  }

  search_result.clear();
  for (size_t i = 0; i < result_candidates.effect_size(search_param.k); i++) {
    search_result.push_back(result_candidates[i].id);
  }

  return num_cnt;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
int GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query,
    const util::GraphSearchParameter& search_param,
    std::vector<std::string>& search_result,
    bool is_extend) {

  IdList search_result_ids;
  int num_cnt = SearchKnn(query, search_param, search_result_ids, is_extend);

  search_result.clear();
  for (auto point_id : search_result_ids) {
    search_result.push_back(this->dataset_ptr_->GetKeyById(point_id));
  }

  return num_cnt;
}

} // namespace core 
} // namespace yannsa

#endif
