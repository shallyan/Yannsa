#ifndef YANNSA_POINT_VECTOR_H
#define YANNSA_POINT_VECTOR_H 

#include <eigen/Dense>

namespace yannsa {
namespace util {

// use Eigen to represent vector 
template <typename CoordinateType>
using PointVector = Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

} // namespace util
} // namespace yannsa

#endif
