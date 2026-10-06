#pragma once

#include "../Preconditioner.h"
#include <GPUSolver/Smoothers/BlockGaussSeidelSmoother.h>

#include <cstddef>

namespace gpuSolver {

class BlockGaussSeidelPreconditioner : public Preconditioner {
private:
  BlockGaussSeidelSmoother *smoother = nullptr;

  float omega_;
  std::size_t numberOfSweeps_;

public:
  explicit BlockGaussSeidelPreconditioner(
      float omega = 1.0f, std::size_t numberOfSweeps = 1);

  ~BlockGaussSeidelPreconditioner() override;

  BlockGaussSeidelPreconditioner(const BlockGaussSeidelPreconditioner &) = delete;
  BlockGaussSeidelPreconditioner &operator=(
      const BlockGaussSeidelPreconditioner &) = delete;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &input, DeviceVector &output) override;
};

} // namespace gpuSolver
