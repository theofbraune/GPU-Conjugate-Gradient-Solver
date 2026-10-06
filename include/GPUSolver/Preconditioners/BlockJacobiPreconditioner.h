#pragma once

#include "../Preconditioner.h"
#include <GPUSolver/Smoothers/BlockJacobiSmoother.h>

#include <cstddef>

namespace gpuSolver {

class BlockJacobiPreconditioner : public Preconditioner {
private:
  BlockJacobiSmoother *smoother = nullptr;

  float omega_;
  std::size_t smoothingSteps_;

public:
  BlockJacobiPreconditioner() = delete;
  BlockJacobiPreconditioner(const std::size_t nSmoother, const float weight);

  ~BlockJacobiPreconditioner() override;

  BlockJacobiPreconditioner(const BlockJacobiPreconditioner &) = delete;
  BlockJacobiPreconditioner &operator=(const BlockJacobiPreconditioner &) = delete;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &residual, DeviceVector &z) override;
};

} // namespace gpuSolver
