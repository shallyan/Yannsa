#ifndef YANNSA_LOCK_H
#define YANNSA_LOCK_H

#include <omp.h>

namespace yannsa {
namespace util {

class Mutex {
  public:
    Mutex() {
      omp_init_lock(&lock);
    }

    void Lock() {
      omp_set_lock(&lock);
    }

    void UnLock() {
      omp_unset_lock(&lock);
    }

    ~Mutex() {
      omp_destroy_lock(&lock);
    }

  public:
    omp_lock_t lock;
};

class ScopedLock {
  public:
    ScopedLock(Mutex& mutex) : mutex_(mutex) {
      mutex_.Lock();
    }

    ~ScopedLock() {
      mutex_.UnLock();
    }

  private:
    Mutex& mutex_;
};

} // namespace util
} // namespace yannsa

#endif
