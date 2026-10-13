#include "CudaTestHelpers.h"

#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceScalar.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>


TEST(CudaDot, SmallKnownResult)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> xHost =
      {1.0f, 2.0f, 3.0f};

  const std::vector<float> yHost =
      {4.0f, 5.0f, 6.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          xHost.size(),
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          yHost.size(),
          yHost.data()));

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeDot(
      *encoder,
      *x,
      *y,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      32.0f,
      1.0e-5f);
}


TEST(CudaDot, LargeVectorAgainstCpu)
{
  gpuSolver::CudaBackend backend;

  constexpr std::size_t n =
      10000;

  std::vector<float> xHost(n);
  std::vector<float> yHost(n);

  double reference = 0.0;

  for (std::size_t i = 0;
       i < n;
       ++i) {

    const float index =
        static_cast<float>(i);

    xHost[i] =
        std::sin(
            0.01f * index);

    yHost[i] =
        std::cos(
            0.017f * index);

    reference +=
        static_cast<double>(xHost[i])
        * static_cast<double>(yHost[i]);
  }

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          n,
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          n,
          yHost.data()));

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeDot(
      *encoder,
      *x,
      *y,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      static_cast<float>(reference),
      1.0e-3f);
}


TEST(CudaSpmv, ThreeByThreeKnownMatrix)
{
  using Matrix =
      Eigen::SparseMatrix<
          float,
          Eigen::RowMajor>;

  Matrix matrix(3, 3);

  std::vector<Eigen::Triplet<float>> triplets = {
      {0, 0, 4.0f},
      {0, 1, 1.0f},

      {1, 0, 1.0f},
      {1, 1, 3.0f},
      {1, 2, 2.0f},

      {2, 1, 2.0f},
      {2, 2, 5.0f}
  };

  matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());

  matrix.makeCompressed();

  gpuSolver::HostCSRMatrix hostMatrix(
      3,
      3,
      static_cast<std::size_t>(
          matrix.nonZeros()),
      matrix.outerIndexPtr(),
      matrix.innerIndexPtr(),
      matrix.valuePtr());

  gpuSolver::CudaBackend backend;

  std::unique_ptr<gpuSolver::DeviceCSRMatrix> deviceMatrix(
      backend.createCSRMatrix(
          hostMatrix));

  const std::vector<float> xHost =
      {1.0f, 2.0f, 3.0f};

  const std::vector<float> expected =
      {6.0f, 13.0f, 19.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          xHost.size(),
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          expected.size()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSpmv(
      *encoder,
      *deviceMatrix,
      *x,
      *y);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*y),
      expected);
}

TEST(CudaSpmv, UpdatedMatrixValuesAreUsed)
{
  using Matrix =
      Eigen::SparseMatrix<
          float,
          Eigen::RowMajor>;

  Matrix matrix(2, 2);

  std::vector<Eigen::Triplet<float>> triplets = {
      {0, 0, 2.0f},
      {0, 1, 1.0f},
      {1, 0, 1.0f},
      {1, 1, 3.0f}
  };

  matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());

  matrix.makeCompressed();

  gpuSolver::HostCSRMatrix hostMatrix(
      2,
      2,
      static_cast<std::size_t>(
          matrix.nonZeros()),
      matrix.outerIndexPtr(),
      matrix.innerIndexPtr(),
      matrix.valuePtr());

  gpuSolver::CudaBackend backend;

  std::unique_ptr<gpuSolver::DeviceCSRMatrix> deviceMatrix(
      backend.createCSRMatrix(
          hostMatrix));

  const std::vector<float> xHost =
      {1.0f, 2.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          2,
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          2));

  // Same CSR structure:
  //
  // [4 2]
  // [2 6]
  //
  const std::vector<float> newValues =
      {4.0f, 2.0f, 2.0f, 6.0f};

  deviceMatrix->updateValues(
      newValues.data(),
      newValues.size());

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSpmv(
      *encoder,
      *deviceMatrix,
      *x,
      *y);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*y),
      {8.0f, 14.0f});
}

TEST(CudaSpmv, RandomSparseMatrixAgainstEigen)
{
  using Matrix =
      Eigen::SparseMatrix<
          float,
          Eigen::RowMajor>;

  constexpr int n = 300;

  Matrix matrix(n, n);

  std::vector<Eigen::Triplet<float>> triplets;

  for (int row = 0;
       row < n;
       ++row) {

    triplets.emplace_back(
        row,
        row,
        4.0f);

    if (row > 0) {
      triplets.emplace_back(
          row,
          row - 1,
          -1.0f);
    }

    if (row + 1 < n) {
      triplets.emplace_back(
          row,
          row + 1,
          -1.0f);
    }

    if (row + 7 < n) {
      triplets.emplace_back(
          row,
          row + 7,
          0.25f);
    }
  }

  matrix.setFromTriplets(
      triplets.begin(),
      triplets.end());

  matrix.makeCompressed();

  Eigen::VectorXf xEigen(n);

  for (int i = 0;
       i < n;
       ++i) {

    xEigen[i] =
        std::sin(
            0.1f
            * static_cast<float>(i));
  }

  const Eigen::VectorXf expectedEigen =
      matrix * xEigen;

  gpuSolver::HostCSRMatrix hostMatrix(
      n,
      n,
      static_cast<std::size_t>(
          matrix.nonZeros()),
      matrix.outerIndexPtr(),
      matrix.innerIndexPtr(),
      matrix.valuePtr());

  gpuSolver::CudaBackend backend;

  std::unique_ptr<gpuSolver::DeviceCSRMatrix> deviceMatrix(
      backend.createCSRMatrix(
          hostMatrix));

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          n,
          xEigen.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          n));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSpmv(
      *encoder,
      *deviceMatrix,
      *x,
      *y);

  backend.submitAndWait(
      *encoder);

  std::vector<float> expected(
      expectedEigen.data(),
      expectedEigen.data()
          + expectedEigen.size());

  const std::vector<float> actual =
      gpuSolverTest::downloadVector(
          *y);

  EXPECT_LT(
      gpuSolverTest::relativeError(
          actual,
          expected),
      1.0e-5f);
}
