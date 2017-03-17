#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int point_neighbor_num;
  int search_point_neighbor_num;
  int max_point_neighbor_num;
  int bucket_neighbor_num;
  int bucket_key_point_num;
  int min_bucket_size;
  int max_bucket_size;
  int refine_iter_num;
};

} // namespace util
} // namespace yannsa

#endif
