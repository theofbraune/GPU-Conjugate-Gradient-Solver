#include "CudaTestHelpers.h"

#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/cuda/CudaBackend.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>


TEST(CudaLibigl, CotangentMatrixSpmvMatchesEigen)
{
  const std::string meshPath =
      std::string(PROJECT_SOURCE_DIR)
      + "/data/surfaceMeshes/cow30k.obj";

  Eigen::MatrixXf vertices;
  Eigen::MatrixXi faces;

  ASSERT_TRUE(
      igl::read_triangle_mesh(
          meshPath,
          vertices,
          faces));

  ASSERT_GT(
      vertices.rows(),
      0);

  ASSERT_GT(
      faces.rows(),
      0);

  Eigen::SparseMatrix<float> cotangentColumnMajor;

  igl::cotmatrix(
      vertices,
      faces,
      cotangentColumnMajor);

  Eigen::SparseMatrix<
      float,
      Eigen::RowMajor>
      matrix =
          cotangentColumnMajor;

  matrix.makeCompressed();

  ASSERT_GT(
      matrix.nonZeros(),
      0);

  const std::size_t n =
      static_cast<std::size_t>(
          matrix.rows());

  std::vector<float> xHost(n);

  for (std::size_t i = 0;
       i < n;
       ++i) {

    const float index =
        static_cast<float>(i);

    xHost[i] =
        std::sin(
            0.013f * index)
        + 0.2f
              * std::cos(
                  0.037f * index);
  }

  Eigen::Map<const Eigen::VectorXf> xEigen(
      xHost.data(),
      static_cast<Eigen::Index>(n));

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

  std::unique_ptr<gpuSolver::DeviceVector> deviceX(
      backend.createVector(
          n,
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> deviceY(
      backend.createVector(
          n));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSpmv(
      *encoder,
      *deviceMatrix,
      *deviceX,
      *deviceY);

  backend.submitAndWait(
      *encoder);

  const std::vector<float> actual =
      gpuSolverTest::downloadVector(
          *deviceY);

  const std::vector<float> expected(
      expectedEigen.data(),
      expectedEigen.data()
          + expectedEigen.size());

  const float error =
      gpuSolverTest::relativeError(
          actual,
          expected);

  EXPECT_LT(
      error,
      1.0e-5f);
}
