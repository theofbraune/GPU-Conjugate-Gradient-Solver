#pragma once

#include "../Preconditioner.h"
#include "GPUSolver/DeviceVector.h"
#include <cstddef>
namespace gpuSolver {

class MetalJacobiPreconditioner : public Preconditioner {

  // std::size_t sizeVec;
  std::size_t diagonalSize_;
  DeviceVector *diagonalValuesInvDevice;
  float clampingTol = 1e-8f;

public:
  MetalJacobiPreconditioner();

  ~MetalJacobiPreconditioner() override;

  MetalJacobiPreconditioner(const MetalJacobiPreconditioner &) = delete;

  MetalJacobiPreconditioner &operator=(const MetalJacobiPreconditioner &) = delete;

  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &input, DeviceVector &output) override;

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;
};
} // namespace gpuSolver
