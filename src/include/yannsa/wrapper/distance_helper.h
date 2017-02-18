#ifndef YANNSA_DISTANCE_HELPER_H
#define YANNSA_DISTANCE_HELPER_H

#include "yannsa/wrapper/representation.h"

namespace yannsa {
namespace wrapper {

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

} // namespace wrapper 
} // namespace yannsa

#endif
