#ifndef YANNSA_HEAP_H
#define YANNSA_HEAP_H

#include <vector>
#include <algorithm>

namespace yannsa {
namespace util {

template <typename T>
class Heap {
  public:
    Heap(int max_size) {
      max_size_ = max_size;
      heap_.reserve(max_size_);
    }

    inline size_t Size() {
      return heap_.size();
    }

    inline std::vector<T>& GetContent() {
      return this->heap_;
    }

    inline void Sort() {
      std::sort_heap(heap_.begin(), heap_.end());
    }

    inline void Reset() {
      std::make_heap(heap_.begin(), heap_.end());
    }

    void Insert(const T& new_item) {
      size_t cur_size = this->Size();
      if (cur_size < max_size_) {
        this->Push(new_item);
      }
      else if (cur_size > 0) {
        const T& top_item = heap_[0];
        // max heap
        if (new_item < top_item) {
          // remove old top one
          this->Pop();

          // add current new one
          this->Push(new_item);
        }
      }
    }

  private:
    void Push(const T& new_item) {
      heap_.push_back(new_item);
      std::push_heap(heap_.begin(), heap_.end());
    }

    void Pop() {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.pop_back();
    }

  private:
    std::vector<T> heap_;
    int max_size_;
};

} // namespace util
} // namespace yannsa

#endif
