#include "CudaTestHelpers.h"

#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceIndexVector.h>

#include <Eigen/Sparse>

#include <gtest/gtest.h>

#include <memory>
#include <vector>


TEST(CudaGaussSeidel, OneColor)
{
  using Matrix =
      Eigen::SparseMatrix<
          float,
          Eigen::RowMajor>;

  Matrix matrix(4, 4);

  std::vector<Eigen::Triplet<float>> triplets = {
      {0, 0, 4.0f},
      {0, 1, -1.0f},

      {1, 0, -1.0f},
      {1, 1, 4.0f},
      {1, 2, -1.0f},

      {2, 1, -1.0f},
      {2, 2, 4.0f},
      {2, 3, -1.0f},

      {3, 2, -1.0f},
      {3, 3, 4.0f}
  };

  matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());

  matrix.makeCompressed();

  gpuSolver::HostCSRMatrix hostMatrix(
      4,
      4,
      static_cast<std::size_t>(
          matrix.nonZeros()),
      matrix.outerIndexPtr(),
      matrix.innerIndexPtr(),
      matrix.valuePtr());

  gpuSolver::CudaBackend backend;

  std::unique_ptr<gpuSolver::DeviceCSRMatrix> deviceMatrix(
      backend.createCSRMatrix(
          hostMatrix));

  // color 0 first, then color 1
  const std::vector<int> colors =
      {0, 2, 1, 3};

  gpuSolver::CudaDeviceIndexVector colorVertices(
      colors);

  const std::vector<float> rhs =
      {4.0f, 8.0f, 12.0f, 16.0f};

  const std::vector<float> initial =
      {0.0f, 0.0f, 0.0f, 0.0f};

  std::unique_ptr<gpuSolver::DeviceVector> deviceRhs(
      backend.createVector(
          rhs.size(),
          rhs.data()));

  std::unique_ptr<gpuSolver::DeviceVector> solution(
      backend.createVector(
          initial.size(),
          initial.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  // Only color {0,2}.
  backend.encodeGaussSeidelColor(
      *encoder,
      *deviceMatrix,
      colorVertices,
      0,
      2,
      *deviceRhs,
      *solution,
      1.0f);

  backend.submitAndWait(
      *encoder);

  // Starting from zero:
  //
  // x0 = 4 / 4 = 1
  // x2 = 12 / 4 = 3
  //
  // Rows 1 and 3 are untouched.
  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(
          *solution),
      {1.0f, 0.0f, 3.0f, 0.0f});
}


TEST(CudaGaussSeidel, FullTwoColorSweep)
{
  using Matrix =
      Eigen::SparseMatrix<
          float,
          Eigen::RowMajor>;

  Matrix matrix(4, 4);

  std::vector<Eigen::Triplet<float>> triplets = {
      {0, 0, 4.0f},
      {0, 1, -1.0f},

      {1, 0, -1.0f},
      {1, 1, 4.0f},
      {1, 2, -1.0f},

      {2, 1, -1.0f},
      {2, 2, 4.0f},
      {2, 3, -1.0f},

      {3, 2, -1.0f},
      {3, 3, 4.0f}
  };

  matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());

  matrix.makeCompressed();

  gpuSolver::HostCSRMatrix hostMatrix(
      4,
      4,
      static_cast<std::size_t>(
          matrix.nonZeros()),
      matrix.outerIndexPtr(),
      matrix.innerIndexPtr(),
      matrix.valuePtr());

  gpuSolver::CudaBackend backend;

  std::unique_ptr<gpuSolver::DeviceCSRMatrix> deviceMatrix(
      backend.createCSRMatrix(
          hostMatrix));

  gpuSolver::CudaDeviceIndexVector colorVertices(
      std::vector<int>{
          0, 2,
          1, 3
      });

  const std::vector<float> rhs =
      {4.0f, 8.0f, 12.0f, 16.0f};

  std::unique_ptr<gpuSolver::DeviceVector> deviceRhs(
      backend.createVector(
          4,
          rhs.data()));

  std::unique_ptr<gpuSolver::DeviceVector> solution(
      backend.createVector(
          4));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSetZero(
      *encoder,
      *solution);

  backend.encodeGaussSeidelColor(
      *encoder,
      *deviceMatrix,
      colorVertices,
      0,
      2,
      *deviceRhs,
      *solution,
      1.0f);

  backend.encodeGaussSeidelColor(
      *encoder,
      *deviceMatrix,
      colorVertices,
      2,
      2,
      *deviceRhs,
      *solution,
      1.0f);

  backend.submitAndWait(
      *encoder);

  // After color 0:
  //
  // x0 = 1
  // x2 = 3
  //
  // Then:
  //
  // x1 = (8 + x0 + x2) / 4 = 3
  // x3 = (16 + x2) / 4 = 4.75
  //
  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(
          *solution),
      {1.0f, 3.0f, 3.0f, 4.75f});
}
