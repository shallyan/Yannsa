#ifndef YANNSA_ERROR_DEFINITION_H
#define YANNSA_ERROR_DEFINITION_H 

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

class IndexBuildError : public YannsaError {
  public:
    IndexBuildError(const std::string& error_msg) : YannsaError(error_msg) {}
};

class IndexReadError : public YannsaError {
  public:
    IndexReadError(const std::string& error_msg) : YannsaError(error_msg) {}
};

} // namespace yannsa

#endif
