#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int point_neighbor_num;
  int search_point_neighbor_num;
  int search_start_point_num;
  int max_point_neighbor_num;
  int min_bucket_size;
  int max_bucket_size;
  int global_refine_iter_num;
  int local_refine_iter_num;
};

} // namespace util
} // namespace yannsa

#endif
