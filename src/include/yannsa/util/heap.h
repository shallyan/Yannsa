#ifndef YANNSA_HEAP_H
#define YANNSA_HEAP_H

#include <vector>
#include <algorithm>

namespace yannsa {
namespace util {

template <typename PointType>
class Heap {
  public:
    Heap(int max_size) {
      max_size_ = max_size;
      heap_.reserve(max_size_);
    }

    inline size_t Size() {
      return heap_.size();
    }

    inline std::vector<PointType>& GetContent() {
      return this->heap_;
    }

    inline void Sort() {
      std::sort_heap(heap_.begin(), heap_.end());
    }

    inline void Reset() {
      std::make_heap(heap_.begin(), heap_.end());
    }

    void Insert(const PointType& new_point) {
      size_t cur_size = this->Size();
      if (cur_size < max_size_) {
        this->Push(new_point);
      }
      else if (cur_size > 0) {
        const PointType& top_point = heap_.front();
        // max heap
        if (new_point < top_point) {
          // remove old top one
          this->Pop();

          // add current new one
          this->Push(new_point);
        }
      }
    }

  private:
    void Push(const PointType& new_point) {
      heap_.push_back(new_point);
      std::push_heap(heap_.begin(), heap_.end());
    }

    void Pop() {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.pop_back();
    }

  private:
    std::vector<PointType> heap_;
    int max_size_;
};

} // namespace util
} // namespace yannsa

#endif
