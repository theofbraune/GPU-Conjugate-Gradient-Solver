#pragma once

#include "../Preconditioner.h"
#include "SymmetricGaussSeidelPreconditioner.h"

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Permutation.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class Backend;
class BackendEncoder;
class DeviceCSRMatrix;
class DeviceVector;
class HostCSRMatrix;

class MultigridGaussSeidelPreconditioner : public Preconditioner {
private:

  struct Level {
    // Owned Galerkin matrix in the level's permuted ordering.
    DeviceCSRMatrix *matrix = nullptr;

    // Owned symmetric colored Gauss-Seidel smoother.
    SymmetricGaussSeidelPreconditioner *smoother = nullptr;

    // V-cycle scratch vectors.
    DeviceVector *rhs = nullptr;
    DeviceVector *residual = nullptr;
    DeviceVector *correction = nullptr;
    DeviceVector *Az = nullptr;
  };

  struct Transfer {
    DeviceCSRMatrix *restriction = nullptr;
    DeviceCSRMatrix *prolongation = nullptr;
  };

public:

  MultigridGaussSeidelPreconditioner(
      Backend &backend,
      const AMGHierarchy &hierarchy,
      const std::vector<Permutation> &permutations,
      std::size_t preSmoothingSteps,
      std::size_t postSmoothingSteps,
      std::size_t nbOfReps,
      float omega = 1.0f
  );

  ~MultigridGaussSeidelPreconditioner() override;

  MultigridGaussSeidelPreconditioner(
      const MultigridGaussSeidelPreconditioner &) = delete;

  MultigridGaussSeidelPreconditioner &operator=(
      const MultigridGaussSeidelPreconditioner &) = delete;

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

private:

  void vCycle(
      Backend &backend,
      BackendEncoder &encoder,
      std::size_t level,
      const DeviceVector &rhs,
      DeviceVector &solution
  );

  void applyVCycles(
      Backend &backend,
      BackendEncoder &encoder,
      const DeviceVector &rhs,
      DeviceVector &solution,
      std::size_t numberOfCycles
  );

private:

  std::vector<Level> levels_;
  std::vector<Transfer> transfers_;

  std::size_t preSmoothingSteps_;
  std::size_t postSmoothingSteps_;
  std::size_t nbOfReps_;

  float omega_;
};

} // namespace gpuSolver
