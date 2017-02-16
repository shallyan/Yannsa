#ifndef YANNSA_COMMON_H
#define YANNSA_COMMON_H

#include <string>
#include <stdexcept>

namespace yannsa {
namespace util {

class YannsaError : public std::logic_error {
  public:
    YannsaError(const std::string& error_msg) : logic_error(error_msg) {}
};

class DataKeyExistError : public YannsaError {
  public:
    DataKeyExistError(const std::string& error_msg) : YannsaError(error_msg) {}
};

} // namespace util
} // namespace yannsa

#endif
