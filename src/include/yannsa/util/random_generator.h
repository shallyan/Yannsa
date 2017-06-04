#ifndef YANNSA_RANDOM_H
#define YANNSA_RANDOM_H

#include <random>
#include <algorithm>
#include "yannsa/base/type_definition.h"

namespace yannsa {
namespace util {

class IntRandomGenerator {
  public:
    IntRandomGenerator(int begin, int end) 
        : distribution_generator_(begin, end), random_generator_(std::random_device()()) {
    }

    int Random() {
      return distribution_generator_(random_generator_);
    }
  
  private:
    std::uniform_int_distribution<> distribution_generator_;
    std::mt19937 random_generator_;
};

template <typename CoordinateType>
class GaussRealRandomGenerator {
  public:
    GaussRealRandomGenerator(CoordinateType mean, CoordinateType stddev) 
        : distribution_generator_(mean, stddev), random_generator_(/*std::random_device()()*/2) {
    }

    CoordinateType Random() {
      return distribution_generator_(random_generator_);
    }
  
  private:
    std::normal_distribution<CoordinateType> distribution_generator_;
    std::mt19937 random_generator_;
};

} // namespace util
} // namespace yannsa

#endif
