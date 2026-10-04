#pragma once

#include "../Preconditioner.h"
#include "GPUSolver/DeviceSparseMatrix.h"
#include "GPUSolver/DeviceVector.h"
#include <GPUSolver/Smoothers/DampedJacobiSmoother.h>
#include <cstddef>
namespace gpuSolver {

class DampedJacobiPreconditioner : public Preconditioner {
private:
  DampedJacobiSmoother* smoother;

  float omega_;
  std::size_t smoothingSteps_;


public:
  void apply(Backend &backend, BackendEncoder &encoder,
             const DeviceVector &residual, DeviceVector &z) override;

  DampedJacobiPreconditioner() = delete;
  DampedJacobiPreconditioner(const std::size_t nSmoother, const float weight);
  
  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  ~DampedJacobiPreconditioner() override;
};

} // namespace gpuSolver
