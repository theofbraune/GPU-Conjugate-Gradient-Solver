#pragma once

#include "MultigridSmoother.h"

namespace gpuSolver {

class MultigridDampedJacobiSmoother
    : public MultigridSmoother {

public:
  MultigridDampedJacobiSmoother(
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
            SmootherType::DampedJacobi,
            preSmoothingSteps,
            postSmoothingSteps,
            coarseSmoothingSteps,
            omega)
  {
  }
};

} // namespace gpuSolver
