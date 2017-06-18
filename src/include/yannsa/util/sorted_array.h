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
    SortedArray(size_t max_size = 0) {
      max_size_ = max_size;
      sorted_array_.reserve(max_size_);
    }

    inline void clear() {
      sorted_array_.clear();
    }

    inline size_t size() {
      return sorted_array_.size();
    }

    inline bool full() {
      return sorted_array_.size() == max_size_;
    }

    inline size_t effect_size(size_t used_size) {
      return std::min(size(), used_size);
    }

    inline void remax_size(size_t new_size) {
      size_t old_size = max_size_;
      max_size_ = new_size; 
      if (new_size > old_size) {
        sorted_array_.reserve(new_size);
      }
      else if (new_size < old_size) {
        sorted_array_.resize(new_size);
      }
    }

    inline PointType& operator[](int i) {
      return sorted_array_[i];
    }

    // assume at lease one element in array
    inline const PointType& min_array() {
      return sorted_array_[0];
    }

    inline const PointType& max_array() {
      return sorted_array_[size()-1];
    }

    inline iterator begin() {
      return sorted_array_.begin();
    }

    inline iterator end() {
      return sorted_array_.end();
    }

    void unique() {
      std::sort(sorted_array_.begin(), sorted_array_.end());
      sorted_array_.resize(std::unique(sorted_array_.begin(), sorted_array_.end()) - sorted_array_.begin());
    }

    void unique(size_t start) {
      std::sort(sorted_array_.begin()+start, sorted_array_.end());
      size_t cur_pos = start;
      for (size_t i = start; i < sorted_array_.size(); i++) {
        // check neighbor
        if (i >= 1 && sorted_array_[i] == sorted_array_[i-1]) {
          continue;
        }
        // check front part
        bool is_duplicate = false;
        for (size_t e = 0; e < cur_pos; e++) {
          if (sorted_array_[e] == sorted_array_[i]) {
            is_duplicate = true;
            break;
          }
        }
        if (!is_duplicate) {
          sorted_array_[cur_pos++] = sorted_array_[i];
        }
      }
      sorted_array_.resize(cur_pos);
    }

    void parallel_push(const PointType& new_point) {
      ScopedLock lock(lock_);
      return push(new_point);
    }

    void push(const PointType& new_point) {
      sorted_array_.push_back(new_point);
    }

    size_t parallel_insert(const PointType& new_point) {
      ScopedLock lock(lock_);
      return insert(new_point);
    }

    size_t insert(const PointType& new_point) {
      // find insert postion
      size_t pos = size();
      while (pos >= 1 && new_point < sorted_array_[pos-1]) {
        pos--;
      }

      // check repeat
      if (pos >= 1) {
        for (int before = pos-1; before >=0; before--) {
          if (sorted_array_[before] < new_point) {
            break;
          }
          if (sorted_array_[before] == new_point) {
            return max_size_;
          }
        }
      }

      // resize array size+1
      if (size() < max_size_) {
        sorted_array_.push_back(new_point);
      }
      else if (pos == max_size_) {
        // can not insert
        return max_size_;
      }

      // put new point at pos
      for (size_t i = size(); i > pos+1; i--) {
        sorted_array_[i-1] = sorted_array_[i-2];
      }
      sorted_array_[pos] = new_point;
      return pos;
    }

  private:
    std::vector<PointType> sorted_array_;
    size_t max_size_;
    Mutex lock_;
};

} // namespace util
} // namespace yannsa

#endif
