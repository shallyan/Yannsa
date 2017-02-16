#ifndef YANNSA_REPRESENTATION_H
#define YANNSA_REPRESENTATION_H

#include "yannsa/util/common.h"
#include <eigen/Dense>
#include <map>
#include <vector>

namespace yannsa {
namespace util {

// use Eigen to represent vector temporarily
template <typename PointCoordinateType>
using PointVector = Eigen::Matrix<PointCoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

} // namespace util
} // namespace yannsa

#endif
