#include "CudaTestHelpers.h"

#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceIndexVector.h>

#include <Eigen/Sparse>

#include <gtest/gtest.h>

#include <memory>
#include <vector>


TEST(CudaBlockOps, ApplyTwoInverseBlocks)
{
  gpuSolver::CudaBackend backend;

  // Two row-major 3x3 blocks:
  //
  // D0^{-1} = diag(2,3,4)
  //
  // D1^{-1} =
  // [1 1 0]
  // [0 2 0]
  // [0 0 3]
  //
  const std::vector<float> inverseBlocks = {
      2.0f, 0.0f, 0.0f,
      0.0f, 3.0f, 0.0f,
      0.0f, 0.0f, 4.0f,

      1.0f, 1.0f, 0.0f,
      0.0f, 2.0f, 0.0f,
      0.0f, 0.0f, 3.0f
  };

  const std::vector<float> residual = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f
  };

  const std::vector<float> expected = {
      2.0f, 6.0f, 12.0f,
      9.0f, 10.0f, 18.0f
  };

  std::unique_ptr<gpuSolver::DeviceVector> deviceInverse(
      backend.createVector(
          inverseBlocks.size(),
          inverseBlocks.data()));

  std::unique_ptr<gpuSolver::DeviceVector> deviceResidual(
      backend.createVector(
          residual.size(),
          residual.data()));

  std::unique_ptr<gpuSolver::DeviceVector> correction(
      backend.createVector(
          residual.size()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeApplyBlockInverses3x3(
      *encoder,
      *deviceInverse,
      *deviceResidual,
      *correction);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(
          *correction),
      expected);
}
