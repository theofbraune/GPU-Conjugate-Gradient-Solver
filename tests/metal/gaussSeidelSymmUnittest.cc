#include <gtest/gtest.h>

#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/GraphColoring.h>
#include <GPUSolver/HostSparseMatrix.h>

#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>

#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

// ============================================================
// Construct SPD five-point Poisson matrix
// ============================================================

Matrix buildPoissonMatrix(int nx, int ny) {
  const int n = nx * ny;

  std::vector<Eigen::Triplet<float>> triplets;

  triplets.reserve(5 * n);

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int row = y * nx + x;

      triplets.emplace_back(row, row, 4.0f);

      if (x > 0) {
        triplets.emplace_back(row, row - 1, -1.0f);
      }

      if (x + 1 < nx) {
        triplets.emplace_back(row, row + 1, -1.0f);
      }

      if (y > 0) {
        triplets.emplace_back(row, row - nx, -1.0f);
      }

      if (y + 1 < ny) {
        triplets.emplace_back(row, row + nx, -1.0f);
      }
    }
  }

  Matrix A(n, n);

  A.setFromTriplets(triplets.begin(), triplets.end());

  A.makeCompressed();

  return A;
}

// ============================================================
// CPU reference: update one color
//
// IMPORTANT:
// This must use precisely the same formula as the Metal kernel.
// ============================================================

void gaussSeidelColorCPU(const Matrix &A, const Eigen::VectorXf &rhs,
                         Eigen::VectorXf &solution,
                         const std::vector<int> &colorVertices,
                         std::size_t begin, std::size_t end, float omega) {
  const int *rowPtr = A.outerIndexPtr();
  const int *colPtr = A.innerIndexPtr();
  const float *values = A.valuePtr();

  for (std::size_t index = begin; index < end; ++index) {
    const int row = colorVertices[index];

    float diagonal = 0.0f;

    float sum = rhs[row];

    for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
      const int col = colPtr[entry];

      const float value = values[entry];

      if (col == row) {
        diagonal = value;
      } else {
        sum -= value * solution[col];
      }
    }

    const float oldValue = solution[row];

    const float newValue = sum / diagonal;

    solution[row] = (1.0f - omega) * oldValue + omega * newValue;
  }
}

// ============================================================
// CPU reference: symmetric multicolor Gauss-Seidel
// ============================================================

void symmetricGaussSeidelCPU(const Matrix &A, const Eigen::VectorXf &rhs,
                             Eigen::VectorXf &solution,
                             const gpuSolver::GraphColoring &coloring,
                             float omega, std::size_t iterations) {
  const std::vector<int> &colorVertices = coloring.colorVertices();

  const std::vector<int> &colorOffsets = coloring.colorOffsets();

  const std::size_t numberOfColors = coloring.numberOfColors();

  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    // ----------------------------------------------------
    // Forward sweep
    // ----------------------------------------------------

    for (std::size_t color = 0; color < numberOfColors; ++color) {
      const std::size_t begin = colorOffsets[color];

      const std::size_t end = colorOffsets[color + 1];

      gaussSeidelColorCPU(A, rhs, solution, colorVertices, begin, end, omega);
    }

    // ----------------------------------------------------
    // Backward sweep
    // ----------------------------------------------------

    for (std::size_t color = numberOfColors; color > 0; --color) {
      const std::size_t currentColor = color - 1;

      const std::size_t begin = colorOffsets[currentColor];

      const std::size_t end = colorOffsets[currentColor + 1];

      gaussSeidelColorCPU(A, rhs, solution, colorVertices, begin, end, omega);
    }
  }
}

} // namespace

// ============================================================
// Main correctness test
// ============================================================

TEST(SymmetricGaussSeidel, GPUAgreesWithCPU) {
  constexpr int nx = 8;
  constexpr int ny = 8;

  constexpr float omega = 1.0f;

  // --------------------------------------------------------
  // Construct matrix.
  // --------------------------------------------------------

  Matrix A = buildPoissonMatrix(nx, ny);

  const std::size_t n = static_cast<std::size_t>(A.rows());

  // --------------------------------------------------------
  // Deterministic RHS.
  // --------------------------------------------------------

  Eigen::VectorXf rhs(A.rows());

  for (Eigen::Index i = 0; i < rhs.size(); ++i) {
    rhs[i] = std::sin(0.17f * static_cast<float>(i)) + 0.5f;
  }

  // --------------------------------------------------------
  // Construct coloring for CPU reference.
  // --------------------------------------------------------

  gpuSolver::GraphColoring coloring;

  coloring.compute(n, A.outerIndexPtr(), A.innerIndexPtr());

  std::cout << "\nGraph coloring:\n"
            << "  DOFs   = " << n << "\n"
            << "  colors = " << coloring.numberOfColors() << "\n";

  // --------------------------------------------------------
  // CPU reference.
  // --------------------------------------------------------

  Eigen::VectorXf cpuSolution = Eigen::VectorXf::Zero(A.rows());

  symmetricGaussSeidelCPU(A, rhs, cpuSolution, coloring, omega, 1);

  // --------------------------------------------------------
  // Construct Metal backend.
  // --------------------------------------------------------

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  // --------------------------------------------------------
  // Construct HostCSRMatrix.
  // --------------------------------------------------------

  gpuSolver::HostCSRMatrix hostMatrix(
      n, n, static_cast<std::size_t>(A.nonZeros()), A.outerIndexPtr(),
      A.innerIndexPtr(), A.valuePtr());

  // --------------------------------------------------------
  // Upload matrix.
  // --------------------------------------------------------

  gpuSolver::DeviceCSRMatrix *deviceMatrix =
      backend.createCSRMatrix(hostMatrix);

  // --------------------------------------------------------
  // Construct symmetric GS preconditioner.
  // --------------------------------------------------------

  gpuSolver::SymmetricGaussSeidelPreconditioner preconditioner(omega);

  preconditioner.initialize(backend, hostMatrix, *deviceMatrix);

  // --------------------------------------------------------
  // Upload RHS.
  // --------------------------------------------------------

  gpuSolver::DeviceVector *rhsDevice = backend.createVector(n, rhs.data());

  gpuSolver::DeviceVector *solutionDevice = backend.createVector(n);

  // --------------------------------------------------------
  // Run one symmetric GS application on GPU.
  // --------------------------------------------------------

  gpuSolver::BackendEncoder *encoder = backend.createEncoder();

  preconditioner.apply(backend, *encoder, *rhsDevice, *solutionDevice);

  backend.submitAndWait(*encoder);

  delete encoder;

  // --------------------------------------------------------
  // Download solution.
  // --------------------------------------------------------

  const float *gpuData = solutionDevice->download();

  Eigen::VectorXf gpuSolution(A.rows());

  for (Eigen::Index i = 0; i < gpuSolution.size(); ++i) {
    gpuSolution[i] = gpuData[i];
  }

  // --------------------------------------------------------
  // Compare solutions.
  // --------------------------------------------------------

  const float relativeDifference =
      (gpuSolution - cpuSolution).norm() / std::max(1.0f, cpuSolution.norm());

  // --------------------------------------------------------
  // Check true residual.
  // --------------------------------------------------------

  const float initialResidual = rhs.norm();

  const float cpuResidual = (rhs - A * cpuSolution).norm();

  const float gpuResidual = (rhs - A * gpuSolution).norm();

  std::cout << "\nSymmetric Gauss-Seidel test:\n"
            << "  initial residual       = " << initialResidual << "\n"
            << "  CPU residual           = " << cpuResidual << "\n"
            << "  GPU residual           = " << gpuResidual << "\n"
            << "  relative difference    = " << relativeDifference << "\n";

  const float preconditionerEnergy = rhs.dot(gpuSolution);

  std::cout << "  r^T M^-1 r           = " << preconditionerEnergy << "\n";

  EXPECT_GT(preconditionerEnergy, 0.0f);
  EXPECT_TRUE(gpuSolution.allFinite());

  EXPECT_LT(relativeDifference, 1e-5f);

  EXPECT_LT(gpuResidual, initialResidual);

  // --------------------------------------------------------
  // Cleanup.
  //
  // Preconditioner owns colorVertices_ but does not own
  // deviceMatrix.
  // --------------------------------------------------------

  delete rhsDevice;
  delete solutionDevice;
  delete deviceMatrix;
}
