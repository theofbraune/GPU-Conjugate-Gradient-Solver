#pragma once
#include <cstddef>

namespace gpuSolver {

class ReorderingStrategy {

public:
  virtual ~ReorderingStrategy() = default;

  virtual void compute(std::size_t nRows, const int *rowPtr, const int *colPtr,
                       int *newToOld, int *oldToNew) const = 0;
};

} // namespace gpuSolver
