#ifndef YANNSA_CODE_H
#define YANNSA_CODE_H

#include "yannsa/base/type_definition.h"
#include <memory>

namespace yannsa {
namespace util {

template <typename PointType>
class BaseEncoder {
  public:
    BaseEncoder(int code_length) : code_length_(code_length) {}

    virtual IntCode Encode(const PointType& point) = 0; 

  protected:
    int code_length_;
};

template <typename PointType>
using BaseEncoderPtr = std::shared_ptr<BaseEncoder<PointType> >;

} // namespace util
} // namespace yannsa

#endif
