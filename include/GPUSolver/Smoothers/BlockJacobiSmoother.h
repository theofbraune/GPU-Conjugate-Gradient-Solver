#pragma once
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Smoother.h>
#include <GPUSolver/Backend.h>

namespace gpuSolver {

class BlockJacobiSmoother : public Smoother {

  const DeviceCSRMatrix *matrix_;

  DeviceVector *inverseDiagonalBlocks_;

  DeviceVector *Az_;
  DeviceVector *residual_;
  DeviceVector *correction_;

  float omega_;

public:
  explicit BlockJacobiSmoother(float omega);

  void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) override;

  void smooth(Backend &backend, BackendEncoder &encoder,
              const DeviceVector &rhs, DeviceVector &solution,
              std::size_t iterations) override;

  ~BlockJacobiSmoother() override;
};

} // namespace gpuSolver
