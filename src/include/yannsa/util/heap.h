#ifndef YANNSA_HEAP_H
#define YANNSA_HEAP_H

#include <vector>
#include <algorithm>

namespace yannsa {
namespace util {

template <typename PointType>
class Heap {
  public:
    typedef typename std::vector<PointType>::iterator Iterator;

  public:
    Heap(int max_size = 0) {
      // 0 means no max size
      max_size_ = max_size;
      heap_.reserve(max_size_);
    }

    inline void Clear() {
      heap_.clear();
    }

    inline size_t Size() {
      return heap_.size();
    }

    inline Iterator Begin() {
      return heap_.begin();
    }

    inline Iterator End() {
      return heap_.end();
    }

    inline void Sort() {
      std::sort_heap(heap_.begin(), heap_.end());
    }

    inline void Reset() {
      std::make_heap(heap_.begin(), heap_.end());
    }

    int Insert(const PointType& new_point) {
      size_t cur_size = Size();
      if (max_size_ == 0 || cur_size < max_size_) {
        Push(new_point);
        return 1;
      }
      else if (cur_size > 0) {
        const PointType& top_point = heap_.front();
        // max heap
        if (new_point < top_point) {
          // remove old top one
          Pop();

          // add current new one
          Push(new_point);
          return 1;
        }
      }
      return 0;
    }

    void Push(const PointType& new_point) {
      heap_.push_back(new_point);
      std::push_heap(heap_.begin(), heap_.end());
    }

    void Pop() {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.pop_back();
    }

    PointType& Front() {
      return heap_.front(); 
    }
  private:
    std::vector<PointType> heap_;
    int max_size_;
};

} // namespace util
} // namespace yannsa

#endif
