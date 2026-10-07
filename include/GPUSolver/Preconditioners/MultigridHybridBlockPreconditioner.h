
#pragma once

#include <GPUSolver/Preconditioner.h>
#include <GPUSolver/Smoothers/MultigridSmoother.h>

namespace gpuSolver {

class MultigridHybridBlockPreconditioner : public Preconditioner {
private:
  MultigridSmoother smoother_;
  std::size_t numberOfVCycles_;

public:
  MultigridHybridBlockPreconditioner(
      Backend &backend,
      const AMGHierarchy &hierarchy,
      const std::vector<Permutation> &permutations,
      std::size_t preSmoothingSteps,
      std::size_t postSmoothingSteps,
      std::size_t coarseSmoothingSteps,
      float omega,
      std::size_t numberOfVCycles = 1);

  ~MultigridHybridBlockPreconditioner() override = default;

  void initialize(
      Backend &backend,
      const HostCSRMatrix &hostMatrix,
      const DeviceCSRMatrix &deviceMatrix) override;

  void apply(
      Backend &backend,
      BackendEncoder &encoder,
      const DeviceVector &residual,
      DeviceVector &z) override;
};

} // namespace gpuSolver
