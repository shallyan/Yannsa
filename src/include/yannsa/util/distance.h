#ifndef YANNSA_DISTANCE_H
#define YANNSA_DISTANCE_H

#include "yannsa/util/common.h"
#include <eigen/Dense>
#include <map>
#include <vector>

namespace yannsa {
namespace util {

// use Eigen to represent vector temporarily
template <typename PointCoordinateType>
using PointVector = Eigen::Matrix<PointCoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

// - dot
template <typename DistanceType>
struct DotDistance {
  template <typename PointCoordinateType>
  DistanceType operator()(const PointVector<PointCoordinateType>& point_a, 
                          const PointVector<PointCoordinateType>& point_b) {
    return -point_a.dot(point_b);
  }
};

// - cosine similarity
template <typename DistanceType>
struct CosineDistance {
  template <typename PointCoordinateType>
  DistanceType operator()(const PointVector<PointCoordinateType>& point_a, 
                          const PointVector<PointCoordinateType>& point_b) {
    auto normalized_point_a = point_a.normalized();
    auto normalized_point_b = point_b.normalized();
    return -normalized_point_a.dot(normalized_point_b);
  }
};

// euclidean distance
template <typename DistanceType>
struct EuclideanDistance {
  template <typename PointCoordinateType>
  DistanceType operator()(const PointVector<PointCoordinateType>& point_a, 
                          const PointVector<PointCoordinateType>& point_b) {
    return (point_a - point_b).norm();
  }
};

} // namespace util
} // namespace yannsa

#endif
