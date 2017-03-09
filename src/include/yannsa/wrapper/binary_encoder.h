#ifndef YANNSA_BINARY_CODE_H
#define YANNSA_BINARY_CODE_H 

#include "yannsa/wrapper/representation.h"
#include "yannsa/util/base_encoder.h"
#include <random>
#include <iostream>

namespace yannsa {
namespace wrapper {

template <typename CoordinateType>
using Hyperplane = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename CoordinateType>
class RealRandomGenerator {
  public:
    RealRandomGenerator(CoordinateType begin, CoordinateType end) 
        : distribution_generator_(begin, end), random_generator_(/*std::random_device()()*/0) {
    }

    CoordinateType Random() {
      return distribution_generator_(random_generator_);
    }
  
  private:
    std::uniform_real_distribution<CoordinateType> distribution_generator_;
    std::mt19937 random_generator_;
};

template <typename PointType, typename CoordinateType>
class BinaryEncoder : public util::BaseEncoder<PointType> {
  public:
    BinaryEncoder(int point_dim, int code_length) 
        : util::BaseEncoder<PointType>(code_length), hash_func_set_(point_dim, code_length) {
      RealRandomGenerator<CoordinateType> random_generator(-1.0, 1.0);
      for (int col = 0; col < code_length; col++) {
        for (int row = 0; row < point_dim; row++) {
          hash_func_set_(row, col) = random_generator.Random();
        }
      }
    }

    IntCode Encode(const PointType& point) {
      auto hash_results = point.transpose() * hash_func_set_;
      IntCode code_result = 0;

      for (int col = 0; col < this->code_length_; col++) {
        CoordinateType one_hash_result = hash_results(0, col);
        IntCode code_binary = one_hash_result > 0.0 ? 1 : 0; 
        code_result = (code_result << 1) + code_binary; 
      }

      return code_result;
    }

    IntCode Distance(const IntCode& a, const IntCode& b) {
      IntCode hamming_dist = 0;
      IntCode xor_result = a ^ b;
      while (xor_result) {
        xor_result &= xor_result-1;
        hamming_dist++;
      }
      return hamming_dist;
    }

  private:
    Hyperplane<CoordinateType> hash_func_set_;
};

} // namespace wrapper 
} // namespace yannsa

#endif
