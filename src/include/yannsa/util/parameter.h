#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int point_neighbor_num;
  int join_point_neighbor_num;
  int refine_iter_num;
  float lambda;
};

struct GraphSearchParameter {
  int k;
  int search_k;
};

} // namespace util
} // namespace yannsa

#endif
