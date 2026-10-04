#pragma once

#include "../Preconditioner.h"
#include "../Smoothers/MultigridGaussSeidelSmoother.h"

namespace gpuSolver {

class MultigridGaussSeidelPreconditioner : public Preconditioner {
private:
  MultigridGaussSeidelSmoother smoother_;

  std::size_t numberOfVCycles_;

public:
  MultigridGaussSeidelPreconditioner(
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


