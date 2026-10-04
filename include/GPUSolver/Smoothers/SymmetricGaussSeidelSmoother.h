
#pragma once

#include "../Smoother.h"

#include <GPUSolver/DeviceIndexVector.h>
#include <GPUSolver/DeviceSparseMatrix.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class SymmetricGaussSeidelSmoother : public Smoother {
private:
  const DeviceCSRMatrix *matrix_ = nullptr;

  DeviceIndexVector *colorVertices_ = nullptr;

  std::vector<int> colorOffsets_;

  std::size_t numberOfColors_ = 0;

  float omega_;

public:
  explicit SymmetricGaussSeidelSmoother(float omega = 1.0f);

  ~SymmetricGaussSeidelSmoother() override;

  SymmetricGaussSeidelSmoother(
      const SymmetricGaussSeidelSmoother &) = delete;

  SymmetricGaussSeidelSmoother &operator=(
      const SymmetricGaussSeidelSmoother &) = delete;

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
