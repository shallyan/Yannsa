#ifndef YANNSA_BINARY_CODE_H
#define YANNSA_BINARY_CODE_H

#include "yannsa/base/type_definition.h"
#include <memory>

namespace yannsa {
namespace util {

template <typename PointType>
class BinaryEncoder {
  public:
    BinaryEncoder(int code_length) : code_length_(code_length) {}

    int code_length() {
      return code_length_;
    }

    virtual IntCode Encode(const PointType& point) = 0; 

  protected:
    int code_length_;
};

template <typename PointType>
using BinaryEncoderPtr = std::shared_ptr<BinaryEncoder<PointType> >;

} // namespace util
} // namespace yannsa

#endif
