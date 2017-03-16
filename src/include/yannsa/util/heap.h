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
    typedef typename std::vector<PointType>::reverse_iterator reverse_iterator;

  public:
    Heap(int max_size = 0) {
      // 0 means no max size
      max_size_ = max_size;
      heap_.reserve(max_size_);
    }

    inline void clear() {
      heap_.clear();
    }

    inline size_t size() {
      return heap_.size();
    }

    inline void resize(int new_size) {
      while (size() > new_size) {
        pop();
      }
    }

    inline iterator begin() {
      return heap_.begin();
    }

    inline iterator end() {
      return heap_.end();
    }

    inline reverse_iterator rbegin() {
      return heap_.rbegin();
    }

    inline reverse_iterator rend() {
      return heap_.rend();
    }

    inline void sort() {
      std::sort_heap(heap_.begin(), heap_.end());
    }

    inline void reset() {
      std::make_heap(heap_.begin(), heap_.end());
    }

    int SafeUniqInsert(const PointType& new_point) {
      ScopedLock lock = ScopedLock(lock_);
      return UniqInsert(new_point);
    }

    int UniqInsert(const PointType& new_point) {
      if (find(heap_.begin(), heap_.end(), new_point) != heap_.end()) {
        return 0;
      }
      return insert(new_point);
    }

    int insert(const PointType& new_point) {
      size_t cur_size = size();
      if (max_size_ == 0 || cur_size < max_size_) {
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

    PointType GetMinValue() {
      return *std::min_element(heap_.begin(), heap_.end());
    }

    PointType GetMaxValue() {
      return heap_.front(); 
    }

    PointType& front() {
      return heap_.front(); 
    }

  private:
    std::vector<PointType> heap_;
    int max_size_;
    Mutex lock_;
};

} // namespace util
} // namespace yannsa

#endif
