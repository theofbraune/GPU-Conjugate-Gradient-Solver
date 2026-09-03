#include <GPUSolver/Permutation.h>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace gpuSolver {

Permutation::Permutation(std::size_t size, const int *oldToNew,
                         const int *newToOld) {
  if (size == 0) {
    throw std::runtime_error("Permutation: size must be nonzero.");
  }

  if (oldToNew == nullptr || newToOld == nullptr) {
    throw std::runtime_error("Permutation: input pointers must not be null.");
  }
  this->size_ = size;
  oldToNew_ = new int[size];
  newToOld_ = new int[size];

  std::memcpy(oldToNew_, oldToNew, size_ * sizeof(int));
  std::memcpy(newToOld_, newToOld, size_ * sizeof(int));

  for (std::size_t oldIndex = 0; oldIndex < size_; ++oldIndex) {
    const int newIndex = oldToNew_[oldIndex];

    if (newIndex < 0 || newIndex >= static_cast<int>(size_)) {
      throw std::runtime_error("Permutation: oldToNew contains invalid index.");
    }

    if (newToOld_[newIndex] != static_cast<int>(oldIndex)) {
      throw std::runtime_error("Permutation: mappings are not inverses.");
    }
  }
}

Permutation::~Permutation() {

  delete[] oldToNew_;
  delete[] newToOld_;
}

const int *Permutation::oldToNew() const { return this->oldToNew_; }

const int *Permutation::newToOld() const { return this->newToOld_; }

std::size_t Permutation::size() const { return size_; }

} // namespace gpuSolver
