#ifndef YANNSA_BI_NEIGHBOR_H
#define YANNSA_BI_NEIGHBOR_H

#include <vector>
#include "yannsa/util/lock.h" 
#include "yannsa/base/type_definition.h" 

namespace yannsa {
namespace core {

template <typename DistanceType>
struct BiNeighbor {
  IdList old_list;
  IdList new_list;
  IdList reverse_old_list;
  IdList reverse_new_list;
  DistanceType radius;
  size_t effect_size;
  bool is_updated;
  bool is_join;
  util::Mutex lock;

  BiNeighbor() {
    is_join = false;
  }

  void reset(size_t s, DistanceType r) {
    old_list.clear();
    new_list.clear();
    reverse_new_list.clear();
    reverse_old_list.clear();
    effect_size = s;
    radius = r;
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

template <typename ContinuesPointKnnGraph, typename DistanceType>
struct ContinuesBiNeighborInfo {
    typedef BiNeighbor<DistanceType> PointBiNeighbor;
    typedef std::vector<PointBiNeighbor> BiNeighborInfo;

    BiNeighborInfo point2bi_neighbor;
    ContinuesPointKnnGraph& point_knn_graph;
    IntIndex max_point_id;

    ContinuesBiNeighborInfo(ContinuesPointKnnGraph& graph):
        point_knn_graph(graph) {
      max_point_id = graph.size();
      point2bi_neighbor = BiNeighborInfo(max_point_id);
    }

    void Init(DynamicBitset& refine_point_flag) {
      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        point2bi_neighbor[point_id].is_join = refine_point_flag[point_id];
      }
      InitStatus();
    }

    void Init() {
      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        point2bi_neighbor[point_id].is_join = true;
      }
      InitStatus();
    }

    void Update(int new_point_num, int sample_num) {
      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        PointBiNeighbor& point_bi_neighbor = point2bi_neighbor[point_id];
        if (!point_bi_neighbor.is_join) {
          continue;
        }
        
        auto& point_neighbor = point_knn_graph[point_id];
        IntIndex effect_size = point_bi_neighbor.effect_size;
        if (point_bi_neighbor.is_updated && effect_size < point_neighbor.size()) {
          int new_point_count = 0;
          for (IntIndex i = 0; i < point_neighbor.size(); i++) {
            if (point_neighbor[i].flag) {
              new_point_count++;
              if (new_point_count >= new_point_num) {
                effect_size = i+1;
                break;
              }
            }
          }
        }
        point_bi_neighbor.reset(effect_size, point_neighbor[effect_size-1].distance);
      }

      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        PointBiNeighbor& point_bi_neighbor = point2bi_neighbor[point_id];
        if (!point_bi_neighbor.is_join) {
          continue;
        }

        auto& point_neighbor = point_knn_graph[point_id];
        
        IntIndex effect_size = point_neighbor.effect_size(point_bi_neighbor.effect_size);
        for (IntIndex i = 0; i < effect_size; i++) {
          auto& neighbor = point_neighbor[i];
          PointBiNeighbor& neighbor_bi_neighbor = point2bi_neighbor[neighbor.id];
          // neighbor
          point_bi_neighbor.insert(neighbor.id, neighbor.flag);
          // reverse neighbor, avoid repeat element
          if (neighbor.distance > neighbor_bi_neighbor.radius) {
            neighbor_bi_neighbor.parallel_insert_reverse(point_id, neighbor.flag);
          }
          neighbor.flag = false;
        }
      }

      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        PointBiNeighbor& point_bi_neighbor = point2bi_neighbor[point_id];
        if (!point_bi_neighbor.is_join) {
          continue;
        }

        IdList& new_list = point_bi_neighbor.new_list;
        IdList& reverse_new_list = point_bi_neighbor.reverse_new_list;
        if (new_list.size() == 0 && reverse_new_list.size() == 0) {
          continue;
        }

        if (reverse_new_list.size() > sample_num) {
          std::random_shuffle(reverse_new_list.begin(), reverse_new_list.end());
          reverse_new_list.resize(sample_num);
        }

        IdList& reverse_old_list = point_bi_neighbor.reverse_old_list;
        if (reverse_old_list.size() > sample_num) {
          std::random_shuffle(reverse_old_list.begin(), reverse_old_list.end());
          reverse_old_list.resize(sample_num);
        }
      }
    }

    void InitStatus() {
      #pragma omp parallel for schedule(static)
      for (IntIndex point_id = 0; point_id < max_point_id; point_id++) {
        PointBiNeighbor& point_bi_neighbor = point2bi_neighbor[point_id];
        if (!point2bi_neighbor[point_id].is_join) {
          continue;
        }

        auto& point_neighbor = point_knn_graph[point_id];
        // size >= 1
        IntIndex effect_size = point_neighbor.size();
        point_bi_neighbor.reset(effect_size, point_neighbor[effect_size-1].distance);
        point_bi_neighbor.is_updated = false;
      }
    }
};

} // namespace core 
} // namespace yannsa

#endif
