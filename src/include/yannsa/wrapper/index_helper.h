#ifndef YANNSA_INDEX_HELPER_H
#define YANNSA_INDEX_HELPER_H

#include "yannsa/util/point_vector.h"
#include "yannsa/core/graph_index.h"
#include "yannsa/wrapper/distance_helper.h"
#include <string>

namespace yannsa {
namespace wrapper {

template <typename CoordinateType>
using CosineGraphIndex = core::GraphIndex<std::string, 
                                          util::PointVector<CoordinateType>, 
                                          CosineDistance<CoordinateType>, 
                                          CoordinateType>;

template <typename CoordinateType>
using CosineGraphIndexPtr = std::shared_ptr<CosineGraphIndex<CoordinateType> >;

template <typename CoordinateType>
using CosineBruteForceIndex = core::BruteForceIndex<std::string, 
                                                    util::PointVector<CoordinateType>, 
                                                    CosineDistance<CoordinateType>, 
                                                    CoordinateType>;

template <typename CoordinateType>
using CosineBruteForceIndexPtr = std::shared_ptr<CosineBruteForceIndex<CoordinateType> >;

} // namespace wrapper 
} // namespace yannsa

#endif
