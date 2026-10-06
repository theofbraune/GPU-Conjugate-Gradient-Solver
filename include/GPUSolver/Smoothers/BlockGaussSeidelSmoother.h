#pragma once
#include <GPUSolver/Smoother.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/DeviceIndexVector.h> 
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>

namespace gpuSolver{


class BlockGaussSeidelSmoother : public Smoother {
private:
  const DeviceCSRMatrix *matrix_ = nullptr;

  DeviceVector *inverseDiagonalBlocks_ = nullptr;

  DeviceIndexVector *colorBlocks_ = nullptr;

  std::vector<int> colorOffsets_;

  std::size_t numberOfColors_ = 0;

  float omega_;

public:
  explicit BlockGaussSeidelSmoother(
      float omega = 1.0f
  );

  ~BlockGaussSeidelSmoother() override;

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

}
