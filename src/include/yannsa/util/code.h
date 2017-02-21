#ifndef YANNSA_CODE_H
#define YANNSA_CODE_H

#include "yannsa/base/type_definition.h"

namespace yannsa {
namespace util {

template <typename PointType>
class BaseCoder {
  public:
    BaseCoder(int code_length) : code_length_(code_length) {}

    virtual IntCode Code(const PointType& point) = 0; 

  protected:
    int code_length_;
};

} // namespace util
} // namespace yannsa

#endif
