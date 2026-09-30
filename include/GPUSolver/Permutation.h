#pragma once

#include <cstddef>

namespace gpuSolver {

class Permutation {

public:
  Permutation(std::size_t size, const int *oldToNew, const int *newToOld);

  ~Permutation();

  Permutation(const Permutation &) = delete;
  Permutation &operator=(const Permutation &) = delete;
  Permutation(Permutation &&other) noexcept;

  Permutation &operator=(Permutation &&other) noexcept;

  std::size_t size() const;

  const int *oldToNew() const;
  const int *newToOld() const;

private:
  std::size_t size_;

  int *oldToNew_;
  int *newToOld_;
};

} // namespace gpuSolver
