#ifndef YANNSA_TEST_HELPER_H
#define YANNSA_TEST_HELPER_H

#include "yannsa/util/container.h"
#include "yannsa/util/random_generator.h"
#include "yannsa/core/graph_index.h"
#include <vector>
#include <memory>
#include <cmath>

// not speed up distance calculation 
// for test only

namespace yannsa {
namespace test {

// representation
template <typename CoordinateType>
using PointVector = std::vector<CoordinateType>;

template <typename CoordinateType>
using Dataset = util::Container<PointVector<CoordinateType> > ; 

template <typename CoordinateType>
using DatasetPtr = std::shared_ptr<Dataset<CoordinateType> >;

// distance
template <typename DistanceType>
struct EuclideanDistance {
  template <typename CoordinateType>
  DistanceType operator()(const PointVector<CoordinateType>& point_a, 
                          const PointVector<CoordinateType>& point_b) {

    CoordinateType dist = 0;
    for (int i = 0; i < point_a.size(); i++) {
      CoordinateType deta = point_a[i] - point_b[i];
      dist += deta * deta;
    }
    return sqrt(dist);
  }
};

template <typename DistanceType>
struct DotDistance {
  template <typename CoordinateType>
  DistanceType operator()(const PointVector<CoordinateType>& point_a, 
                          const PointVector<CoordinateType>& point_b) {
    CoordinateType dist = 0;
    for (int i = 0; i < point_a.size(); i++) {
      dist += point_a[i] * point_b[i];
    }
    return -dist;
  }
};

template <typename PointType, typename CoordinateType>
class BinaryEncoder : public util::BaseEncoder<PointType> {
  public:
    BinaryEncoder(int point_dim, int code_length, DatasetPtr<CoordinateType>& dataset_ptr) 
        : util::BaseEncoder<PointType>(code_length) {
      util::RealRandomGenerator<CoordinateType> random_generator(-1.0, 1.0);
      util::IntRandomGenerator int_random(0, dataset_ptr->size()-1);

      for (int col = 0; col < code_length; col++) {
        std::vector<CoordinateType> one_hash_func;
        for (int row = 0; row < point_dim; row++) {
          one_hash_func.push_back(random_generator.Random());
        }
        PointType& random_point = (*dataset_ptr)[int_random.Random()];
        CoordinateType b = 0;
        if (col > 2) {
        for (int j = 0; j < one_hash_func.size(); j++) {
          b += (random_point[j]/2) * one_hash_func[j];
        }
        }
        one_hash_func.push_back(-b);

        hash_func_set_.push_back(one_hash_func);
      }
    }

    IntCode Encode(const PointType& point) {
      std::vector<CoordinateType> hash_results;
      for (int i = 0; i < hash_func_set_.size(); i++) {
        CoordinateType result = 0;
        int point_dim = hash_func_set_[i].size()-1;
        for (int j = 0; j < point_dim; j++) {
          result += point[j] * hash_func_set_[i][j];
        }
        result += hash_func_set_[i][point_dim];
        hash_results.push_back(result);
      }

      IntCode code_result = 0;
      for (int col = 0; col < this->code_length_; col++) {
        CoordinateType one_hash_result = hash_results[col];
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
    std::vector<std::vector<CoordinateType> > hash_func_set_;
};

template <typename CoordinateType>
using DotGraphIndex = core::GraphIndex<PointVector<CoordinateType>, 
                                       DotDistance<CoordinateType>, 
                                       CoordinateType>;
template <typename CoordinateType>
using DotGraphIndexPtr = std::shared_ptr<DotGraphIndex<CoordinateType> >;

template <typename CoordinateType>
using EuclideanGraphIndex = core::GraphIndex<PointVector<CoordinateType>, 
                                             EuclideanDistance<CoordinateType>, 
                                             CoordinateType>;
template <typename CoordinateType>
using EuclideanGraphIndexPtr = std::shared_ptr<EuclideanGraphIndex<CoordinateType> >;

} // namespace test 
} // namespace yannsa

#endif
