#ifndef YANNSA_BI_NEIGHBOR_H
#define YANNSA_BI_NEIGHBOR_H

#include <vector>
#include "yannsa/util/lock.h" 
#include "yannsa/base/type_definition.h" 

namespace yannsa {
namespace util {

template <typename DistanceType>
struct BiNeighbor {
  std::vector<IntIndex> old_list;
  std::vector<IntIndex> new_list;
  std::vector<IntIndex> reverse_old_list;
  std::vector<IntIndex> reverse_new_list;
  DistanceType radius;
  Mutex lock;

  void reset(DistanceType r) {
    old_list.clear();
    new_list.clear();
    reverse_new_list.clear();
    reverse_old_list.clear();
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
    ScopedLock sl(lock);
    if (new_flag) {
      reverse_new_list.push_back(point_id);
    }
    else {
      reverse_old_list.push_back(point_id);
    }
  }

};

} // namespace util
} // namespace yannsa

#endif
