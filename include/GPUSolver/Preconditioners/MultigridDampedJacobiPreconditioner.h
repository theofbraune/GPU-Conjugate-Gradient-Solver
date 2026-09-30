#pragma once

#include "../Preconditioner.h"
#include "DampedJacobiPreconditioner.h"
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

class MultigridDampedJacobiPreconditioner : public Preconditioner {
private:
  struct Level {
    // Galerkin-projected operator A_k, in this level's permuted ordering.
    // Owned here: vCycle needs it directly for residual computation,
    // independent of what the smoother does internally.
    DeviceCSRMatrix *matrix;

    // Already-tested single-level smoother for this level. Owns its own
    // diagonal and its own internal scratch (Az_, correction_) — this
    // class does not duplicate either.
    DampedJacobiPreconditioner *smoother;

    // V-cycle's own scratch: residual restricted down to this level,
    // and the correction prolongated back up through it.
    DeviceVector *residual;
    DeviceVector *correction;
    DeviceVector *Az;
    DeviceVector *rhs;

    Level();
  };
  struct Transfer {
    DeviceCSRMatrix *restriction;
    DeviceCSRMatrix *prolongation;

    Transfer();
  };

public:
  MultigridDampedJacobiPreconditioner(
      Backend &backend, const AMGHierarchy &hierarchy,
      const std::vector<Permutation> &permutations,
      std::size_t preSmoothingSteps, std::size_t postSmoothingSteps,
      std::size_t nbOfReps, float omega);

  ~MultigridDampedJacobiPreconditioner() override;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &residual, DeviceVector &z) override;

private:
  void smooth(Backend &backend, BackendEncoder &encoder,
              const DeviceVector &rhs, DeviceVector &solution,
              std::size_t numberOfSteps);

  void vCycle(Backend &backend, BackendEncoder &encoder, std::size_t level,
              const DeviceVector &rhs, DeviceVector &solution);

private:
  std::vector<Level> levels_;
  std::vector<Transfer> transfers_;

  std::size_t preSmoothingSteps_;
  std::size_t postSmoothingSteps_;
  std::size_t nbOfReps_;
  float omega_;
};

} // namespace gpuSolver
