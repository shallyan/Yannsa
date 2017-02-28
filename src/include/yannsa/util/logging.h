#ifndef YANNSA_LOGGING_H
#define YANNSA_LOGGING_H

#include <ctime>
#include <iostream>

namespace yannsa {
namespace util {

void Log(const std::string& prompt) {
  time_t now = time(0);
  char* dt = ctime(&now);
  std::cout << prompt << ": " << dt;
}

} // namespace util
} // namespace yannsa

#endif

