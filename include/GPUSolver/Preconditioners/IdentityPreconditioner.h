#pragma once

#include "../Preconditioner.h"
namespace gpuSolver {
class IdentityPreconditioner : public Preconditioner {

public:
  IdentityPreconditioner() = default;

  ~IdentityPreconditioner() override = default;
  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &input, DeviceVector &output) override;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;
};
} // namespace gpuSolver
