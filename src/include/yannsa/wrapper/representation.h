#ifndef YANNSA_REPRESENTATION_H
#define YANNSA_REPRESENTATION_H 

#include <eigen/Dense>
#include "yannsa/util/container.h"
#include <memory>

namespace yannsa {
namespace wrapper {

// use Eigen to represent vector 
template <typename CoordinateType>
using PointVector = Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename CoordinateType>
using Dataset = util::Container<PointVector<CoordinateType> > ; 

template <typename CoordinateType>
using DatasetPtr = std::shared_ptr<Dataset<CoordinateType> >;

} // namespace wrapper 
} // namespace yannsa

#endif
