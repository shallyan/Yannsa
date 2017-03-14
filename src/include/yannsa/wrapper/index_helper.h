#ifndef YANNSA_INDEX_HELPER_H
#define YANNSA_INDEX_HELPER_H

#include "yannsa/core/brute_force_index.h"
#include "yannsa/core/graph_index.h"
#include "yannsa/wrapper/representation.h"
#include "yannsa/wrapper/distance.h"
#include <memory>
#include <string>

namespace yannsa {
namespace wrapper {

template <typename CoordinateType>
using DotGraphIndex = core::GraphIndex<PointVector<CoordinateType>, 
                                       DotDistance<CoordinateType>, 
                                       CoordinateType>;

template <typename CoordinateType>
using DotGraphIndexPtr = std::shared_ptr<DotGraphIndex<CoordinateType> >;

template <typename CoordinateType>
using DotBruteForceIndex = core::BruteForceIndex<PointVector<CoordinateType>, 
                                                 DotDistance<CoordinateType>, 
                                                 CoordinateType>;

template <typename CoordinateType>
using DotBruteForceIndexPtr = std::shared_ptr<DotBruteForceIndex<CoordinateType> >;

} // namespace wrapper 
} // namespace yannsa

#endif
