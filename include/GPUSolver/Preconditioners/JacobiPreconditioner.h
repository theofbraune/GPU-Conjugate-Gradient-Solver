#pragma once

#include "../Preconditioner.h"
#include "GPUSolver/DeviceVector.h"
#include <cstddef>
namespace gpuSolver {

class JacobiPreconditioner : public Preconditioner {

  // std::size_t sizeVec;
  std::size_t diagonalSize_;
  DeviceVector *diagonalValuesInvDevice;
  float clampingTol = 1e-8f;

public:
  JacobiPreconditioner();

  ~JacobiPreconditioner() override;

  JacobiPreconditioner(const JacobiPreconditioner &) = delete;

  JacobiPreconditioner &operator=(const JacobiPreconditioner &) = delete;

  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &input, DeviceVector &output) override;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;
};
} // namespace gpuSolver
