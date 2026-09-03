#pragma once

#include "ReorderingStrategy.h"

namespace gpuSolver {

class RCMReordering : public ReorderingStrategy {

public:
  void compute(std::size_t nRows, const int *rowPtr, const int *colPtr,
               int *oldToNew, int *newToOld) const override;
};

} // namespace gpuSolver
