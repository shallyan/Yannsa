#ifndef YANNSA_BINARY_CODE_IMP_H
#define YANNSA_BINARY_CODE_IMP_H 

#include "yannsa/wrapper/representation.h"
#include "yannsa/util/random_generator.h"
#include "yannsa/util/binary_encoder.h"
#include <random>

namespace yannsa {
namespace wrapper {

template <typename CoordinateType>
using Hyperplane = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename PointType, typename CoordinateType>
class RandomBinaryEncoder : public util::BinaryEncoder<PointType> {
  public:
    RandomBinaryEncoder(int point_dim, int code_length) 
        : util::BinaryEncoder<PointType>(code_length), hash_func_set_(point_dim, code_length) {

      util::GaussRealRandomGenerator<CoordinateType> random_generator(0.0, 1.0);
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

  private:
    Hyperplane<CoordinateType> hash_func_set_;
};

} // namespace wrapper 
} // namespace yannsa

#endif