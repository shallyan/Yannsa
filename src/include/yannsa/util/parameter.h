#ifndef YANNSA_PARAMETER_H
#define YANNSA_PARAMETER_H 

namespace yannsa {
namespace util {

struct GraphIndexParameter {
  int point_neighbor_num;
  int bucket_neighbor_num;
  int min_bucket_size;
  int max_bucket_size;
};

} // namespace util
} // namespace yannsa

#endif
