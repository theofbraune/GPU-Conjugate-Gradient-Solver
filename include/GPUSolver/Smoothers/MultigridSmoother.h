#pragma once

#include "../Smoother.h"

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Permutation.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class DeviceVector;
class DeviceCSRMatrix;

class MultigridSmoother : public Smoother {
public:
  enum class SmootherType { DampedJacobi, SymmetricGaussSeidel, BlockJacobi, BlockGaussSeidel };

private:
  struct Level {
    DeviceCSRMatrix *matrix = nullptr;

    Smoother *smoother = nullptr;

    DeviceVector *rhs = nullptr;
    DeviceVector *residual = nullptr;
    DeviceVector *correction = nullptr;
    DeviceVector *Az = nullptr;
  };

  struct Transfer {
    DeviceCSRMatrix *restriction = nullptr;
    DeviceCSRMatrix *prolongation = nullptr;
  };

  std::vector<Level> levels_;
  std::vector<Transfer> transfers_;

  std::size_t preSmoothingSteps_;
  std::size_t postSmoothingSteps_;
  std::size_t coarseSmoothingSteps_;

  void vCycle(Backend &backend, BackendEncoder &encoder, std::size_t level,
              const DeviceVector &rhs, DeviceVector &solution);

public:
  MultigridSmoother(Backend &backend, const AMGHierarchy &hierarchy,
                    const std::vector<Permutation> &permutations,
                    SmootherType smootherType, std::size_t preSmoothingSteps,
                    std::size_t postSmoothingSteps,
                    std::size_t coarseSmoothingSteps, float omega);

  ~MultigridSmoother() override;

  MultigridSmoother(const MultigridSmoother &) = delete;

  MultigridSmoother &operator=(const MultigridSmoother &) = delete;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  void smooth(Backend &backend, BackendEncoder &encoder,
              const DeviceVector &rhs, DeviceVector &solution,
              std::size_t iterations) override;
};

} // namespace gpuSolver
