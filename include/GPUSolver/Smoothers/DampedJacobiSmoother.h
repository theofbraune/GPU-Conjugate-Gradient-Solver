#pragma once 

#include <cstddef>
#include "../Smoother.h"

#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>

namespace gpuSolver {

class DampedJacobiSmoother : public Smoother {
private:
  DeviceVector *inverseDiagonal_ = nullptr;
  DeviceVector *Az_ = nullptr;
  DeviceVector *correction_ = nullptr;

  const DeviceCSRMatrix *matrix_ = nullptr;

  float omega_;

public:
  explicit DampedJacobiSmoother(float omega = 0.7f);

  ~DampedJacobiSmoother() override;

  DampedJacobiSmoother(const DampedJacobiSmoother &) = delete;

  DampedJacobiSmoother &operator=(
      const DampedJacobiSmoother &) = delete;

  void initialize(
      Backend &backend,
      const HostCSRMatrix &hostMatrix,
      const DeviceCSRMatrix &deviceMatrix
  ) override;

  void smooth(
      Backend &backend,
      BackendEncoder &encoder,
      const DeviceVector &rhs,
      DeviceVector &solution,
      std::size_t iterations
  ) override;
};

} // namespace gpuSolver
