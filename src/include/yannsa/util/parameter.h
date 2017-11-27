#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int k;
  int join_k;
  int refine_iter_num;
  float lambda;
};

struct GraphSearchParameter {
  int K;
  int search_K;
};

} // namespace util
} // namespace yannsa

#endif
