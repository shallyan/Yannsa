#ifndef YANNSA_HEAP_H
#define YANNSA_HEAP_H

#include <vector>
#include <algorithm>
#include "yannsa/util/lock.h" 

namespace yannsa {
namespace util {

template <typename PointType>
class Heap {
  public:
    typedef typename std::vector<PointType>::iterator iterator;

  public:
    Heap(int max_size = 0) {
      max_size_ = max_size;
      heap_.reserve(max_size_);
    }

    inline void clear() {
      heap_.clear();
    }

    inline size_t size() {
      return heap_.size();
    }

    inline size_t effect_size(size_t used_size) {
      return std::min(size(), used_size);
    }

    inline void resize(size_t new_size) {
      max_size_ = std::min(new_size, size());
      heap_.resize(max_size_);
    }

    inline PointType& operator[](int i) {
      return heap_[i];
    }

    // assume at lease one element in array
    inline const PointType& min_array() {
      return heap_[0];
    }

    inline const PointType& max_array() {
      return heap_[size()-1];
    }

    inline iterator begin() {
      return heap_.begin();
    }

    inline iterator end() {
      return heap_.end();
    }

    inline void sort() {
      std::sort_heap(heap_.begin(), heap_.end());
    }

    size_t parallel_insert_array(const PointType& new_point) {
      ScopedLock lock(lock_);
      return insert_array(new_point);
    }

    size_t insert_array(const PointType& new_point) {
      // find insert postion
      size_t pos = size();
      while (pos >= 1 && new_point < heap_[pos-1]) {
        pos--;
      }

      // check repeat
      if (pos >= 1 && heap_[pos-1] == new_point) {
        return max_size_;
      }

      // resize array size+1
      if (size() < max_size_) {
        heap_.push_back(new_point);
      }

      // put new point at pos
      for (size_t i = size(); i > pos+1; i--) {
        heap_[i-1] = heap_[i-2];
      }
      heap_[pos] = new_point;
      return pos;
    }

    size_t insert_heap(const PointType& new_point) {
      size_t cur_size = size();
      if (cur_size < max_size_) {
        push(new_point);
        return 1;
      }
      else if (cur_size > 0) {
        const PointType& top_point = heap_.front();
        // max heap
        if (new_point < top_point) {
          // remove old top one
          pop();

          // add current new one
          push(new_point);
          return 1;
        }
      }
      return 0;
    }

    void push(const PointType& new_point) {
      heap_.push_back(new_point);
      std::push_heap(heap_.begin(), heap_.end());
    }

    void pop() {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.pop_back();
    }

  private:
    std::vector<PointType> heap_;
    size_t max_size_;
    Mutex lock_;
};

} // namespace util
} // namespace yannsa

#endif
