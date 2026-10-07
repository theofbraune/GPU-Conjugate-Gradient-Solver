#pragma once

#include "MultigridSmoother.h"

namespace gpuSolver {

class MultigridHybridBlockSmoother
    : public MultigridSmoother {

public:
  MultigridHybridBlockSmoother(
      Backend &backend,
      const AMGHierarchy &hierarchy,
      const std::vector<Permutation> &permutations,
      std::size_t preSmoothingSteps,
      std::size_t postSmoothingSteps,
      std::size_t coarseSmoothingSteps,
      float omega)
      : MultigridSmoother(
            backend,
            hierarchy,
            permutations,
            SmootherType::BlockGaussSeidel,
            SmootherType::BlockJacobi,
            preSmoothingSteps,
            postSmoothingSteps,
            coarseSmoothingSteps,
            omega)
  {
  }
};

} // namespace gpuSolver
