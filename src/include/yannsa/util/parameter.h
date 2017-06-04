#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int point_neighbor_num;
  int max_point_neighbor_num;
  int refine_iter_num;
};

struct GraphSearchParameter {
  int k;
  int search_k;
  int start_neighbor_num;
};

} // namespace util
} // namespace yannsa

#endif
