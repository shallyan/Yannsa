#ifndef YANNSA_REPRESENTATION_H
#define YANNSA_REPRESENTATION_H 

#include <Eigen/Dense>
#include "yannsa/util/container.h"
#include <memory>

namespace yannsa {
namespace wrapper {

// use Eigen to represent vector 
template <typename CoordinateType>
using PointVector = Eigen::Matrix<CoordinateType, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename CoordinateType>
using Dataset = util::Container<PointVector<CoordinateType> > ; 

template <typename CoordinateType>
using DatasetPtr = std::shared_ptr<Dataset<CoordinateType> >;

} // namespace wrapper 
} // namespace yannsa

#endif
