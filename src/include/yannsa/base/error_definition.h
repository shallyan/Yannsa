#ifndef YANNSA_COMMON_H
#define YANNSA_COMMON_H

#include <string>
#include <stdexcept>

namespace yannsa {

class YannsaError : public std::logic_error {
  public:
    YannsaError(const std::string& error_msg) : logic_error(error_msg) {}
};

class KeyExistError : public YannsaError {
  public:
    KeyExistError(const std::string& error_msg) : YannsaError(error_msg) {}
};

class KeyNotExistError : public YannsaError {
  public:
    KeyNotExistError(const std::string& error_msg) : YannsaError(error_msg) {}
};

} // namespace yannsa

#endif
