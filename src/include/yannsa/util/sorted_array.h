#ifndef YANNSA_SORTED_ARRAY_H
#define YANNSA_SORTED_ARRAY_H

#include <vector>
#include <algorithm>
#include "yannsa/util/lock.h" 

namespace yannsa {
namespace util {

template <typename PointType>
class SortedArray {
  public:
    typedef typename std::vector<PointType>::iterator iterator;

  public:
    // = 0 for default construction
    SortedArray(int max_size = 0) {
      max_size_ = max_size;
      sorted_points_.reserve(max_size_);
    }

    inline void clear() {
      sorted_points_.clear();
    }

    inline size_t size() {
      return sorted_points_.size();
    }

    inline void resize(int new_size) {
      max_size_ = new_size;
      sorted_points_.resize(new_size);
    }

    inline const PointType& operator[](int i) {
      return sorted_points_[i];
    }

    inline iterator begin() {
      return sorted_points_.begin();
    }

    inline iterator end() {
      return sorted_points_.end();
    }

    int SafeInsert(const PointType& new_point) {
      ScopedLock lock = ScopedLock(lock_);
      return insert(new_point);
    }

    int insert(const PointType& new_point) {
      // find insert postion
      int pos = size();
      while (pos-1 >= 0 && new_point < sorted_points_[pos-1]) {
        pos--;
      }

      // check repeat
      if (pos-1 >= 0 && sorted_points_[pos-1] == new_point) {
        return max_size_;
      }

      // resize array size+1
      if (size() < max_size_) {
        sorted_points_.push_back(new_point);
      }

      // put new point at pos
      for (int i = size()-1; i > pos; i--) {
        sorted_points_[i] = sorted_points_[i-1];
      }
      sorted_points_[pos] = new_point;
      return pos;
    }

  private:
    std::vector<PointType> sorted_points_;
    int max_size_;
    Mutex lock_;
};

} // namespace util
} // namespace yannsa

#endif
