#ifndef YANNSA_DISTANCE_H
#define YANNSA_DISTANCE_H

#include <eigen/Dense>
#include <map>
#include <vector>

namespace yannsa {
namespace util {

// use Eigen to represent vector 
template <typename CoordinateType>
using PointVector = Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

// - dot
template <typename DistanceType>
struct DotDistance {
  template <typename CoordinateType>
  DistanceType operator()(const PointVector<CoordinateType>& point_a, 
                          const PointVector<CoordinateType>& point_b) {
    return -point_a.dot(point_b);
  }
};

// - cosine similarity
template <typename DistanceType>
struct CosineDistance {
  template <typename CoordinateType>
  DistanceType operator()(const PointVector<CoordinateType>& point_a, 
                          const PointVector<CoordinateType>& point_b) {
    auto normalized_point_a = point_a.normalized();
    auto normalized_point_b = point_b.normalized();
    return -normalized_point_a.dot(normalized_point_b);
  }
};

// euclidean distance
template <typename DistanceType>
struct EuclideanDistance {
  template <typename CoordinateType>
  DistanceType operator()(const PointVector<CoordinateType>& point_a, 
                          const PointVector<CoordinateType>& point_b) {
    return (point_a - point_b).norm();
  }
};

} // namespace util
} // namespace yannsa

#endif
