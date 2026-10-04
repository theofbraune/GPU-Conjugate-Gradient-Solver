#pragma once

#include "../Preconditioner.h"
#include "../Smoothers/MultigridDampedJacobiSmoother.h"

namespace gpuSolver {

class MultigridDampedJacobiPreconditioner : public Preconditioner {
private:
  MultigridDampedJacobiSmoother smoother_;

  std::size_t numberOfVCycles_;

public:
  MultigridDampedJacobiPreconditioner(
      Backend &backend,
      const AMGHierarchy &hierarchy,
      const std::vector<Permutation> &permutations,
      std::size_t preSmoothingSteps,
      std::size_t postSmoothingSteps,
      std::size_t numberOfVCycles,
      float omega
  );

  void initialize(
      Backend &backend,
      const HostCSRMatrix &hostMatrix,
      const DeviceCSRMatrix &deviceMatrix
  ) override;

  void apply(
      Backend &backend,
      BackendEncoder &encoder,
      const DeviceVector &residual,
      DeviceVector &z
  ) override;
};

} // namespace gpuSolver
