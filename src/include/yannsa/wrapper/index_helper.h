#ifndef YANNSA_INDEX_HELPER_H
#define YANNSA_INDEX_HELPER_H

#include "yannsa/util/point_vector.h"
#include "yannsa/core/index.h"
#include "yannsa/wrapper/distance_helper.h"
#include <string>

namespace yannsa {
namespace wrapper {

template <typename CoordinateType>
using CosineIndex = core::Index<std::string, 
                                util::PointVector<CoordinateType>, 
                                CosineDistance<CoordinateType>, 
                                CoordinateType>;

template <typename CoordinateType>
using CosineIndexPtr = std::shared_ptr<CosineIndex<CoordinateType> >;

} // namespace wrapper 
} // namespace yannsa

#endif
