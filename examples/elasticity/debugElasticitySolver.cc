// Purpose:
//   Debug the elasticity solver stack in the simplest possible order.
//
//   1. Build the constrained linear-elasticity system K u = f.
//   2. Compute an Eigen Cholesky reference solution.
//   3. Apply a SINGLE-LEVEL preconditioner once, outside CG,
//      and inspect ||b - A z|| / ||b||.
//   4. Run CG with that preconditioner, WITHOUT any permutation.
//   5. Build the AMG hierarchy.
//   6. Construct/apply the MG preconditioner once and inspect the residual.
//   7. Run CG with the MG preconditioner, still WITHOUT any permutation.
//
// IMPORTANT:
//   This file is a correctness/debug harness, not a benchmark.
//   Keep the mesh modest and print lots of diagnostics.
//
//   Two API names may need tiny adaptation to your current tree:
//     - MGBuilder::buildAmgcl...ElasticityHierarchy(...)
//     - the exact constructors of the MG preconditioners.
//   Everything else follows the solver interfaces used in the project.
//
// Suggested first run:
//   - enable scalar symmetric GS only
//   - then block Jacobi only
//   - then MG + block Jacobi
//
// Do NOT add RCM/permutations until this file is clean.

#include "ElasticitySimulator.h"
#include "GPUSolver/Permutation.h"
#include "GPUSolver/Preconditioner.h"
#include "GPUSolver/Smoothers/BlockJacobiSmoother.h"
#include "GPUSolver/Smoothers/MultigridSmoother.h"
#include "LinearElasticity.h"

#include <Eigen/SparseCore>
#include <GPUSolver/BlockUtils.h>
#include <GPUSolver/CGSolver.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>

#include <GPUSolver/Preconditioners/BlockGaussSeidelPreconditioner.h>
#include <GPUSolver/Preconditioners/BlockJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/DampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>

// Adapt these include names if your files live elsewhere.
#include <GPUSolver/Preconditioners/MultigridBlockGaussSeidelPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridBlockJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridGaussSeidelPreconditioner.h>

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/ReorderingStrategies/IdentityReordering.h>
#include <GPUSolver/ReorderingStrategies/RCMReordering.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

// Adapt include to your current AMG helper path.
#include "AMGUtils/AMGHierarchyBuilder.h"
#include <AMGUtils/SmoothedAggregationCoarsener.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cstddef>
#include <igl/readMESH.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace {

using SpMatF = Eigen::SparseMatrix<float, Eigen::RowMajor>;
using SpMatD = Eigen::SparseMatrix<double, Eigen::RowMajor>;

struct ElasticityProblem {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXi T;

  SpMatD Kdouble;
  Eigen::VectorXd bDouble;

  SpMatF A;
  Eigen::VectorXf b;

  Eigen::VectorXd reference;
};

std::vector<gpuSolver::Permutation> buildIdentityPermutationsForHierarchy(
    const gpuSolver::AMGHierarchy &hierarchy) {
  std::vector<gpuSolver::Permutation> permutations;

  permutations.reserve(hierarchy.A.size());

  gpuSolver::IdentityReordering idReordering;

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    const gpuSolver::AMGHierarchy::Matrix &A = hierarchy.A[level];

    const std::size_t nRows = static_cast<std::size_t>(A.rows());

    if (A.rows() != A.cols()) {
      throw std::runtime_error("buildRCMPermutationsForHierarchy: "
                               "Galerkin matrix is not square.");
    }

    int *oldToNew = new int[nRows];

    int *newToOld = new int[nRows];

    idReordering.compute(nRows, A.outerIndexPtr(), A.innerIndexPtr(), oldToNew,
                         newToOld);

    permutations.emplace_back(nRows, oldToNew, newToOld);

    delete[] oldToNew;
    delete[] newToOld;
  }

  return permutations;
}

// -----------------------------------------------------------------------------
// Small host-side diagnostics.
// -----------------------------------------------------------------------------

double relativeResidual(const SpMatF &A, const Eigen::VectorXf &x,
                        const Eigen::VectorXf &b) {
  const float denominator = std::max(1.0e-30f, b.norm());
  return static_cast<double>((b - A * x).norm() / denominator);
}

double relativeSolutionError(const Eigen::VectorXf &x,
                             const Eigen::VectorXd &reference) {
  const double denominator = std::max(1.0e-30, reference.norm());
  return (x.cast<double>() - reference).norm() / denominator;
}

void printVectorSanity(const std::string &name, const Eigen::VectorXf &x) {
  std::cout << "  " << name << ": norm=" << x.norm() << ", min=" << x.minCoeff()
            << ", max=" << x.maxCoeff()
            << ", finite=" << (x.allFinite() ? "yes" : "NO") << "\n";
}

// -----------------------------------------------------------------------------
// Build the physical elasticity problem.
// -----------------------------------------------------------------------------

ElasticityProblem buildProblem(const std::string &meshPath) {
  ElasticityProblem problem;

  if (!igl::readMESH(meshPath, problem.V, problem.T, problem.F)) {
    throw std::runtime_error("Could not read tetrahedral mesh.");
  }

  std::cout << "Loaded elasticity mesh:\n"
            << "  vertices = " << problem.V.rows() << "\n"
            << "  tets     = " << problem.T.rows() << "\n";

  constexpr double youngModulus = 1.0e5;
  constexpr double poissonRatio = 0.29;

  const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));

  const double lambda = youngModulus * poissonRatio /
                        ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

  std::cout << "Material:\n"
            << "  E      = " << youngModulus << "\n"
            << "  nu     = " << poissonRatio << "\n"
            << "  lambda = " << lambda << "\n"
            << "  mu     = " << mu << "\n";

  LinearElasticitySimulator simulator(problem.V, problem.F, problem.T, lambda,
                                      mu);

  // Clamp the x-min face.
  const double xMin = problem.V.col(0).minCoeff();
  const double xMax = problem.V.col(0).maxCoeff();
  const double length = xMax - xMin;
  const double tolerance = 1.0e-4 * length;

  std::vector<int> pinnedVertices;

  for (Eigen::Index i = 0; i < problem.V.rows(); ++i) {
    if (problem.V(i, 0) <= xMin + tolerance) {
      pinnedVertices.push_back(static_cast<int>(i));
    }
  }

  if (pinnedVertices.empty()) {
    throw std::runtime_error("No vertices found on the clamped face.");
  }

  std::cout << "Pinned " << pinnedVertices.size() << " vertices.\n";

  simulator.constrainSimulation(pinnedVertices);

  problem.Kdouble = simulator.getHessian();

  const SpMatD M = simulator.getMassMatrixStackedConstrained();

  if (problem.Kdouble.rows() % 3 != 0) {
    throw std::runtime_error(
        "Constrained elasticity system does not preserve 3-DOF blocks.");
  }

  std::cout << "Reduced elasticity system:\n"
            << "  DOFs     = " << problem.Kdouble.rows() << "\n"
            << "  vertices = " << problem.Kdouble.rows() / 3 << "\n"
            << "  nnz      = " << problem.Kdouble.nonZeros() << "\n";

  // Gravity in -z.
  constexpr double density = 1.0;
  constexpr double gravity = -9.81;

  Eigen::VectorXd gravityVector = Eigen::VectorXd::Zero(problem.Kdouble.rows());

  const int nFreeVertices = static_cast<int>(problem.Kdouble.rows() / 3);

  for (int i = 0; i < nFreeVertices; ++i) {
    gravityVector[3 * i + 2] = gravity;
  }

  problem.bDouble = density * M * gravityVector;

  problem.A = problem.Kdouble.cast<float>();

  problem.A.makeCompressed();

  problem.b = problem.bDouble.cast<float>();

  std::cout << "RHS:\n"
            << "  ||b|| = " << problem.b.norm() << "\n"
            << "  finite = " << (problem.b.allFinite() ? "yes" : "NO") << "\n";

  // ---------------------------------------------------------------------------
  // Cholesky reference.
  // ---------------------------------------------------------------------------

  std::cout << "\n[REFERENCE] Eigen Cholesky\n";

  Eigen::SparseMatrix<double, Eigen::ColMajor> Kcol = problem.Kdouble;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double, Eigen::ColMajor>> cholesky;

  cholesky.compute(Kcol);

  if (cholesky.info() != Eigen::Success) {
    throw std::runtime_error("Reference Cholesky factorization failed.");
  }

  problem.reference = cholesky.solve(problem.bDouble);

  if (cholesky.info() != Eigen::Success) {
    throw std::runtime_error("Reference Cholesky solve failed.");
  }

  const double referenceResidual =
      (problem.bDouble - problem.Kdouble * problem.reference).norm() /
      std::max(1.0e-30, problem.bDouble.norm());

  std::cout << "  relative residual = " << std::scientific << referenceResidual
            << std::defaultfloat << "\n";

  return problem;
}

gpuSolver::Permutation
buildBlockRCMPermutation(const Eigen::SparseMatrix<float, Eigen::RowMajor> &A) {
  if (A.rows() != A.cols()) {
    throw std::runtime_error(
        "buildBlockRCMPermutation: matrix must be square.");
  }

  if (A.rows() % 3 != 0) {
    throw std::runtime_error("buildBlockRCMPermutation: elasticity matrix size "
                             "must be divisible by 3.");
  }

  const std::size_t numberOfNodes = static_cast<std::size_t>(A.rows()) / 3;

  gpuSolver::BlockGraph graph = gpuSolver::buildBlockGraph(A, 3);

  if (graph.rowPtr.size() != numberOfNodes + 1) {
    throw std::runtime_error("buildBlockRCMPermutation: invalid block graph.");
  }

  std::vector<int> oldToNewNode(numberOfNodes);
  std::vector<int> newToOldNode(numberOfNodes);

  gpuSolver::RCMReordering rcm;

  rcm.compute(numberOfNodes, graph.rowPtr.data(), graph.colPtr.data(),
              oldToNewNode.data(), newToOldNode.data());

  gpuSolver::Permutation nodePermutation(numberOfNodes, oldToNewNode.data(),
                                         newToOldNode.data());

  return gpuSolver::expandBlockPermutation(nodePermutation, numberOfNodes, 3);
}

std::vector<gpuSolver::Permutation> buildBlockRCMPermutationsForHierarchy(
    const gpuSolver::AMGHierarchy &hierarchy) {
  std::vector<gpuSolver::Permutation> permutations;

  permutations.reserve(hierarchy.A.size());

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {

    const gpuSolver::AMGHierarchy::Matrix &A = hierarchy.A[level];

    gpuSolver::Permutation permutation = buildBlockRCMPermutation(A);

    // checkBlockPermutation(permutation, static_cast<std::size_t>(A.rows() /
    // 3));

    const int* oldToNewPerm = permutation.oldToNew();
    const int* newToOldPerm = permutation.newToOld();
    size_t size = permutation.size();
    permutations.emplace_back(size,oldToNewPerm,newToOldPerm);
  }

  return permutations;
}

void checkBlockPermutation(const gpuSolver::Permutation &permutation,
                           std::size_t numberOfNodes) {
  const int *oldToNew = permutation.oldToNew();

  const int *newToOld = permutation.newToOld();

  for (std::size_t oldNode = 0; oldNode < numberOfNodes; ++oldNode) {

    const int new0 = oldToNew[3 * oldNode + 0];

    const int new1 = oldToNew[3 * oldNode + 1];

    const int new2 = oldToNew[3 * oldNode + 2];

    if (new1 != new0 + 1 || new2 != new0 + 2 || new0 % 3 != 0) {

      throw std::runtime_error(
          "Block permutation does not preserve 3-DOF blocks.");
    }
  }

  for (std::size_t newDof = 0; newDof < 3 * numberOfNodes; ++newDof) {

    const int oldDof = newToOld[newDof];

    if (oldToNew[oldDof] != static_cast<int>(newDof)) {

      throw std::runtime_error("Permutation inverse check failed.");
    }
  }

  std::cout << "Block permutation check: PASS\n";
}

Eigen::SparseMatrix<float, Eigen::RowMajor>
permuteEigenSparseMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor> &A,
                         const gpuSolver::Permutation &permutation) {
  if (A.rows() != A.cols()) {
    throw std::runtime_error(
        "permuteEigenSparseMatrix: matrix must be square.");
  }

  if (static_cast<std::size_t>(A.rows()) != permutation.size()) {

    throw std::runtime_error(
        "permuteEigenSparseMatrix: permutation size mismatch.");
  }

  const int *oldToNew = permutation.oldToNew();

  std::vector<Eigen::Triplet<float>> triplets;

  triplets.reserve(static_cast<std::size_t>(A.nonZeros()));

  for (int oldRow = 0; oldRow < A.outerSize(); ++oldRow) {

    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A,
                                                                       oldRow);
         it; ++it) {

      const int oldCol = it.col();

      const int newRow = oldToNew[oldRow];

      const int newCol = oldToNew[oldCol];

      triplets.emplace_back(newRow, newCol, it.value());
    }
  }

  Eigen::SparseMatrix<float, Eigen::RowMajor> APermuted(A.rows(), A.cols());

  APermuted.setFromTriplets(triplets.begin(), triplets.end());

  APermuted.makeCompressed();

  return APermuted;
}

Eigen::VectorXf
permuteVectorOldToNew(const Eigen::VectorXf &x,
                      const gpuSolver::Permutation &permutation) {
  Eigen::VectorXf xPermuted(x.size());

  const int *oldToNew = permutation.oldToNew();

  for (Eigen::Index oldIndex = 0; oldIndex < x.size(); ++oldIndex) {

    const int newIndex = oldToNew[oldIndex];

    xPermuted[newIndex] = x[oldIndex];
  }

  return xPermuted;
}

Eigen::VectorXd
permuteVectorOldToNew(const Eigen::VectorXd &x,
                      const gpuSolver::Permutation &permutation) {
  Eigen::VectorXd xPermuted(x.size());

  const int *oldToNew = permutation.oldToNew();

  for (Eigen::Index oldIndex = 0; oldIndex < x.size(); ++oldIndex) {

    const int newIndex = oldToNew[oldIndex];

    xPermuted[newIndex] = x[oldIndex];
  }

  return xPermuted;
}

//---------------------------
// AMG checker
//---------------------------
//

void inspectElasticityHierarchy(const gpuSolver::AMGHierarchy &hierarchy,
                                std::size_t blockSize) {
  std::cout
      << "\n============================================================\n"
      << "[ELASTICITY AMG HIERARCHY CHECK]\n"
      << "============================================================\n";

  if (hierarchy.A.empty()) {
    throw std::runtime_error("Hierarchy contains no levels.");
  }

  if (hierarchy.P.size() + 1 != hierarchy.A.size()) {
    throw std::runtime_error(
        "Hierarchy has inconsistent number of prolongation matrices.");
  }

  if (hierarchy.R.size() + 1 != hierarchy.A.size()) {
    throw std::runtime_error(
        "Hierarchy has inconsistent number of restriction matrices.");
  }

  std::cout << "number of levels = " << hierarchy.A.size() << "\n";

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {

    const auto &A = hierarchy.A[level];

    std::cout
        << "\n------------------------------------------------------------\n"
        << "level " << level << "\n"
        << "------------------------------------------------------------\n";

    std::cout << "A: " << A.rows() << " x " << A.cols()
              << ", nnz = " << A.nonZeros() << "\n";

    if (A.rows() != A.cols()) {
      throw std::runtime_error("Coarse matrix is not square.");
    }

    if (A.rows() % static_cast<int>(blockSize) != 0) {
      throw std::runtime_error("Coarse level is not divisible by block size.");
    }

    const std::size_t numberOfBlocks =
        static_cast<std::size_t>(A.rows()) / blockSize;

    std::cout << "block nodes = " << numberOfBlocks << "\n";

    // ----------------------------------------------------------
    // Check nullspace dimensions, if stored.
    // ----------------------------------------------------------

    if (level < hierarchy.nullspaces.size()) {

      const Eigen::MatrixXf &nullspace = hierarchy.nullspaces[level];

      std::cout << "near-nullspace: " << nullspace.rows() << " x "
                << nullspace.cols() << "\n";

      if (nullspace.rows() != A.rows()) {
        throw std::runtime_error(
            "Near-nullspace row count does not match level matrix.");
      }
    }

    // ----------------------------------------------------------
    // Check 3x3 diagonal blocks.
    // ----------------------------------------------------------

    Eigen::SparseMatrix<float, Eigen::RowMajor> matrix = A;

    matrix.makeCompressed();

    gpuSolver::HostCSRMatrix hostMatrix(
        static_cast<std::size_t>(matrix.rows()),
        static_cast<std::size_t>(matrix.cols()),
        static_cast<std::size_t>(matrix.nonZeros()), matrix.outerIndexPtr(),
        matrix.innerIndexPtr(), matrix.valuePtr());

    try {

      std::vector<float> inverseBlocks =
          gpuSolver::extractInverseDiagonalBlocks3x3(hostMatrix);

      std::cout << "3x3 block diagonal inversion: PASS"
                << " (" << inverseBlocks.size() / 9 << " blocks)\n";

    } catch (const std::exception &exception) {

      std::cout << "3x3 block diagonal inversion: FAIL\n"
                << "  " << exception.what() << "\n";

      throw;
    }

    // ----------------------------------------------------------
    // Check transfer operators.
    // ----------------------------------------------------------

    if (level + 1 < hierarchy.A.size()) {

      const auto &P = hierarchy.P[level];

      const auto &R = hierarchy.R[level];

      const auto &ACoarse = hierarchy.A[level + 1];

      std::cout << "P: " << P.rows() << " x " << P.cols()
                << ", nnz = " << P.nonZeros() << "\n";

      std::cout << "R: " << R.rows() << " x " << R.cols()
                << ", nnz = " << R.nonZeros() << "\n";

      if (P.rows() != A.rows()) {
        throw std::runtime_error("P rows do not match fine level.");
      }

      if (P.cols() != ACoarse.rows()) {
        throw std::runtime_error("P columns do not match coarse level.");
      }

      if (R.rows() != ACoarse.rows()) {
        throw std::runtime_error("R rows do not match coarse level.");
      }

      if (R.cols() != A.rows()) {
        throw std::runtime_error("R columns do not match fine level.");
      }

      if (P.rows() % static_cast<int>(blockSize) != 0 ||
          P.cols() % static_cast<int>(blockSize) != 0) {

        throw std::runtime_error("Prolongation dimensions are incompatible "
                                 "with block size.");
      }

      if (R.rows() % static_cast<int>(blockSize) != 0 ||
          R.cols() % static_cast<int>(blockSize) != 0) {

        throw std::runtime_error("Restriction dimensions are incompatible "
                                 "with block size.");
      }
    }
  }

  std::cout << "\nElasticity AMG hierarchy check: PASS\n";
}

void testSingleGaussSeidelColorWrite(
    gpuSolver::Backend &backend, const gpuSolver::HostCSRMatrix &hostA,
    gpuSolver::DeviceCSRMatrix &deviceA,
    const gpuSolver::DeviceIndexVector &colorVertices,
    const std::vector<int> &colorOffsets, std::size_t colorIndex) {
  const std::size_t n = hostA.rows();

  if (colorIndex + 1 >= colorOffsets.size()) {
    throw std::runtime_error(
        "testSingleGaussSeidelColorWrite: invalid color index.");
  }

  const std::size_t begin = static_cast<std::size_t>(colorOffsets[colorIndex]);

  const std::size_t end =
      static_cast<std::size_t>(colorOffsets[colorIndex + 1]);

  const std::size_t count = end - begin;

  std::cout
      << "\n============================================================\n"
      << "[DIRECT GS COLOR WRITE TEST]\n"
      << "============================================================\n"
      << "color       = " << colorIndex << "\n"
      << "begin       = " << begin << "\n"
      << "count       = " << count << "\n"
      << "matrix rows = " << n << "\n";

  // ------------------------------------------------------------
  // Dummy rhs.
  // The debug Metal kernel does not actually need it,
  // but encodeGaussSeidelColor expects one.
  // ------------------------------------------------------------

  std::vector<float> rhs(n, 1.0f);

  gpuSolver::DeviceVector *deviceRhs = backend.createVector(n, rhs.data());

  // ------------------------------------------------------------
  // Start solution at 7 everywhere.
  //
  // Our temporary Metal kernel should contain:
  //
  //     x[row] = 42.0f;
  //
  // ------------------------------------------------------------

  std::vector<float> initialSolution(n, 7.0f);

  gpuSolver::DeviceVector *deviceSolution =
      backend.createVector(n, initialSolution.data());

  // ------------------------------------------------------------
  // Dispatch EXACTLY ONE color.
  // No smoother.
  // No preconditioner.
  // No encodeSetZero.
  // ------------------------------------------------------------

  gpuSolver::BackendEncoder *encoder = backend.createEncoder();

  backend.encodeGaussSeidelColor(*encoder, deviceA, colorVertices, begin, count,
                                 *deviceRhs, *deviceSolution, 1.0f);

  backend.submitAndWait(*encoder);

  delete encoder;

  // ------------------------------------------------------------
  // Inspect result.
  // download() is borrowed/shared memory in your implementation.
  // ------------------------------------------------------------

  float *solutionData = deviceSolution->download();

  // Also inspect GPU color indices directly.
  const int *gpuColorVertices = colorVertices.download();

  // ------------------------------------------------------------
  // Mark which rows belong to the selected color.
  // ------------------------------------------------------------

  std::vector<char> isInColor(n, 0);

  for (std::size_t i = begin; i < end; ++i) {

    const int row = gpuColorVertices[i];

    if (row < 0 || row >= static_cast<int>(n)) {

      throw std::runtime_error("testSingleGaussSeidelColorWrite: "
                               "invalid row in colorVertices.");
    }

    isInColor[static_cast<std::size_t>(row)] = 1;
  }

  // ------------------------------------------------------------
  // Expected:
  //
  // rows in this color: 42
  // all other rows:      7
  // ------------------------------------------------------------

  std::size_t correctChanged = 0;
  std::size_t wrongChanged = 0;
  std::size_t expectedButUnchanged = 0;

  for (std::size_t row = 0; row < n; ++row) {

    const float value = solutionData[row];

    if (isInColor[row]) {

      if (std::abs(value - 42.0f) < 1.0e-6f) {

        ++correctChanged;

      } else {

        ++expectedButUnchanged;
      }

    } else {

      if (std::abs(value - 7.0f) > 1.0e-6f) {

        ++wrongChanged;
      }
    }
  }

  // ------------------------------------------------------------
  // Print a few rows from this color.
  // ------------------------------------------------------------

  std::cout << "\nFirst few rows in selected color:\n";

  const std::size_t numberToPrint = std::min<std::size_t>(count, 10);

  for (std::size_t i = 0; i < numberToPrint; ++i) {

    const int row = gpuColorVertices[begin + i];

    std::cout << "  row " << row << " -> " << solutionData[row] << "\n";
  }

  std::cout << "\nResult summary:\n"
            << "  expected changed rows = " << count << "\n"
            << "  correctly changed     = " << correctChanged << "\n"
            << "  expected but unchanged= " << expectedButUnchanged << "\n"
            << "  wrongly changed rows  = " << wrongChanged << "\n";

  if (correctChanged == count && expectedButUnchanged == 0 &&
      wrongChanged == 0) {

    std::cout << "\nDIRECT GS COLOR TEST: PASS\n";

  } else {

    std::cout << "\nDIRECT GS COLOR TEST: FAIL\n";
  }

  delete deviceRhs;
  delete deviceSolution;
}
// -----------------------------------------------------------------------------
// Apply a preconditioner ONCE, outside CG.
//
// This is deliberately the most important debug test in the file.
// We start from z = 0 and compute z = B b.
//
// For a smoother/preconditioner that behaves sensibly, z should:
//   - be finite,
//   - be nonzero,
//   - usually reduce ||b - A z|| relative to ||b||.
//
// It does NOT need to solve the system accurately in one application.
// -----------------------------------------------------------------------------

void testOnePreconditionerApplication(const std::string &name,
                                      gpuSolver::Backend &backend,
                                      gpuSolver::Preconditioner &preconditioner,
                                      const gpuSolver::AMGHierarchy &hierarchy,
                                      const SpMatF &A, const Eigen::VectorXf &b,
                                      const Eigen::VectorXd &reference) {

  std::cout
      << "\n============================================================\n"
      << "[APPLY TEST] " << name << "\n"
      << "============================================================\n";

  // gpuSolver::HostCSRMatrix hostA(A);

  gpuSolver::HostCSRMatrix hostA(
      static_cast<std::size_t>(A.rows()), static_cast<std::size_t>(A.cols()),
      static_cast<std::size_t>(A.nonZeros()), A.outerIndexPtr(),
      A.innerIndexPtr(), A.valuePtr());
  LOG(INFO) << " host matrix created";

  // No permutation. The active CSR is the original CSR.
  gpuSolver::DeviceCSRMatrix *deviceA = backend.createCSRMatrix(hostA);

  LOG(INFO) << " device matrix created";
  gpuSolver::DeviceVector *deviceB =
      backend.createVector(static_cast<std::size_t>(b.size()), b.data());

  LOG(INFO) << " device vector created";
  float *downloadedB = deviceB->download();

  Eigen::Map<Eigen::VectorXf> bDeviceMap(downloadedB, b.size());

  LOG(INFO) << "device b:"
            << " norm=" << bDeviceMap.norm()
            << ", min=" << bDeviceMap.minCoeff()
            << ", max=" << bDeviceMap.maxCoeff() << "\n";

  std::vector<gpuSolver::Permutation> idPerms =
      buildIdentityPermutationsForHierarchy(hierarchy);
  for (std::size_t steps : {1, 2, 4, 8, 16}) {

    LOG(INFO) << " steps are " << steps;
    gpuSolver::DeviceVector *deviceZ =
        backend.createVector(static_cast<std::size_t>(b.size()));

    LOG(INFO) << "device vector created";
    preconditioner.initialize(backend, hostA, *deviceA);

    LOG(INFO) << " preconditioner initialized";
    gpuSolver::BackendEncoder *encoder = backend.createEncoder();

    LOG(INFO) << " encoder memory allocated";
    backend.encodeCopy(*encoder, *deviceB, *deviceZ);
    backend.encodeSetZero(*encoder, *deviceZ);

    gpuSolver::MultigridSmoother smoother = gpuSolver::MultigridSmoother(
        backend, hierarchy, idPerms,
        gpuSolver::MultigridSmoother::SmootherType::BlockJacobi, steps, steps,
        steps, 0.4f);
    // gpuSolver::BlockJacobiSmoother smoother =
    //     gpuSolver::BlockJacobiSmoother(0.7f);
    // smoother.initialize(backend, hostA, *deviceA);
    // testSingleGaussSeidelColorWrite(backend,hostA,deviceA,)
    // apply() should set/reset z itself if that is part of your
    // Preconditioner contract.
    // backend.encodeSetZero(*encoder, *deviceZ);
    // &smoother.smooth(backend, *encoder, *deviceB, *deviceZ, steps);
    smoother.smooth(backend, *encoder, *deviceB, *deviceZ, steps);

    // preconditioner.apply(backend, *encoder, *deviceB, *deviceZ);
    LOG(INFO) << " preconditioner: check ";

    backend.submitAndWait(*encoder);

    float *downloaded = deviceZ->download();

    Eigen::Map<Eigen::VectorXf> zMap(downloaded, b.size());

    Eigen::VectorXf z = zMap;

    // If download() returns heap ownership in your implementation,
    // restore this delete[]. If DeviceVector owns/returns borrowed memory,
    // remove it.
    // delete[] downloaded;

    printVectorSanity("z = B b", z);

    const double residualAfter = relativeResidual(A, z, b);

    const double errorAfter = relativeSolutionError(z, reference);

    std::cout << std::scientific << "  residual before = " << 1.0 << "\n"
              << "  residual after  = " << residualAfter << "\n"
              << "  solution error  = " << errorAfter << "\n"
              << std::defaultfloat;

    if (!z.allFinite()) {
      throw std::runtime_error(name + ": preconditioner produced NaN/Inf.");
    }

    if (z.norm() == 0.0f) {
      throw std::runtime_error(name +
                               ": preconditioner returned the zero vector.");
    }

    delete deviceZ;
    delete encoder;
  }
  delete deviceB;
  delete deviceA;
}

// -----------------------------------------------------------------------------
// Hook the same preconditioner into CG, still WITHOUT permutation.
// -----------------------------------------------------------------------------

void testCG(const std::string &name, gpuSolver::Backend &backend,
            gpuSolver::Preconditioner &preconditioner, const SpMatF &A,
            const Eigen::VectorXf &b, const Eigen::VectorXd &reference,
            std::size_t maxIterations = 200) {
  std::cout
      << "\n============================================================\n"
      << "[CG TEST] " << name << "\n"
      << "============================================================\n";

  gpuSolver::CGSolver solver(backend, preconditioner,
                             A); // IMPORTANT: no permutation.

  solver.setTolerance(1.0e-4f);
  solver.setMaxIterations(maxIterations);

  Eigen::VectorXf x = Eigen::VectorXf::Zero(A.rows());

  solver.solve(b, x);

  printVectorSanity("x", x);

  std::cout << "  iterations = " << solver.getNbOfIterations() << "\n"
            << std::scientific
            << "  true residual = " << relativeResidual(A, x, b) << "\n"
            << "  solution error = " << relativeSolutionError(x, reference)
            << "\n"
            << std::defaultfloat;
}

void testCGWithPermutation(const std::string &name, gpuSolver::Backend &backend,
                           gpuSolver::Preconditioner &preconditioner,
                           const SpMatF &A, const Eigen::VectorXf &b,
                           const Eigen::VectorXd &reference,
                           const gpuSolver::Permutation &permutation,
                           std::size_t maxIterations = 200) {
  std::cout
      << "\n============================================================\n"
      << "[CG + PERMUTATION TEST] " << name << "\n"
      << "============================================================\n";

  gpuSolver::CGSolver solver(backend, preconditioner, A, permutation);

  solver.setTolerance(1.0e-4f);
  solver.setMaxIterations(maxIterations);

  Eigen::VectorXf x = Eigen::VectorXf::Zero(A.rows());

  solver.solve(b, x);

  printVectorSanity("x", x);

  std::cout << "  iterations = " << solver.getNbOfIterations() << "\n"
            << std::scientific
            << "  true residual = " << relativeResidual(A, x, b) << "\n"
            << "  solution error = " << relativeSolutionError(x, reference)
            << "\n"
            << std::defaultfloat;
}

void testElasticityAMGHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A) {
  std::cout
      << "\n============================================================\n"
      << "[ELASTICITY AMG HIERARCHY TEST]\n"
      << "============================================================\n";

  if (A.rows() != A.cols()) {
    throw std::runtime_error("Elasticity AMG test requires a square matrix.");
  }

  if (A.rows() % 3 != 0) {
    throw std::runtime_error(
        "Elasticity AMG test requires number of DOFs divisible by 3.");
  }

  // ------------------------------------------------------------
  // Build hierarchy.
  //
  // No near-nullspace here:
  // the pinned elasticity problem is SPD.
  // ------------------------------------------------------------

  gpuSolver::SmoothedAggregationCoarsener coarsener(3,     // block size
                                                    0.1f); // epsStrong

  gpuSolver::AMGHierarchyBuilder builder(coarsener,
                                         10,    // max levels
                                         1000); // minimum number of DOFs

  gpuSolver::AMGHierarchy hierarchy = builder.build(A);

  // ------------------------------------------------------------
  // Basic hierarchy consistency.
  // ------------------------------------------------------------

  if (hierarchy.A.empty()) {
    throw std::runtime_error("AMG hierarchy contains no levels.");
  }

  if (hierarchy.P.size() + 1 != hierarchy.A.size()) {
    throw std::runtime_error("Wrong number of prolongation matrices.");
  }

  if (hierarchy.R.size() + 1 != hierarchy.A.size()) {
    throw std::runtime_error("Wrong number of restriction matrices.");
  }

  std::cout << "number of levels = " << hierarchy.A.size() << "\n";

  // ------------------------------------------------------------
  // Inspect every level.
  // ------------------------------------------------------------

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {

    const gpuSolver::AMGHierarchy::Matrix &levelMatrix = hierarchy.A[level];

    std::cout
        << "\n------------------------------------------------------------\n"
        << "level " << level << "\n"
        << "------------------------------------------------------------\n";

    std::cout << "A = " << levelMatrix.rows() << " x " << levelMatrix.cols()
              << "\n";

    std::cout << "nnz = " << levelMatrix.nonZeros() << "\n";

    if (levelMatrix.rows() != levelMatrix.cols()) {
      throw std::runtime_error("AMG coarse matrix is not square.");
    }

    if (levelMatrix.rows() % 3 != 0) {
      throw std::runtime_error(
          "AMG coarse level does not preserve 3-DOF blocks.");
    }

    const std::size_t numberOfBlocks =
        static_cast<std::size_t>(levelMatrix.rows()) / 3;

    std::cout << "block nodes = " << numberOfBlocks << "\n";

    // ----------------------------------------------------------
    // Check that the 3x3 diagonal blocks can be extracted
    // and inverted.
    // ----------------------------------------------------------

    Eigen::SparseMatrix<float, Eigen::RowMajor> compressedMatrix = levelMatrix;

    compressedMatrix.makeCompressed();

    gpuSolver::HostCSRMatrix hostMatrix(
        static_cast<std::size_t>(compressedMatrix.rows()),
        static_cast<std::size_t>(compressedMatrix.cols()),
        static_cast<std::size_t>(compressedMatrix.nonZeros()),
        compressedMatrix.outerIndexPtr(), compressedMatrix.innerIndexPtr(),
        compressedMatrix.valuePtr());

    const std::vector<float> inverseBlocks =
        gpuSolver::extractInverseDiagonalBlocks3x3(hostMatrix);

    const std::size_t expectedInverseEntries = 9 * numberOfBlocks;

    if (inverseBlocks.size() != expectedInverseEntries) {

      throw std::runtime_error(
          "Wrong number of inverse diagonal block entries.");
    }

    std::cout << "3x3 diagonal blocks: PASS\n";

    // ----------------------------------------------------------
    // Inspect transfer operators to next level.
    // ----------------------------------------------------------

    if (level + 1 < hierarchy.A.size()) {

      const gpuSolver::AMGHierarchy::Matrix &P = hierarchy.P[level];

      const gpuSolver::AMGHierarchy::Matrix &R = hierarchy.R[level];

      const gpuSolver::AMGHierarchy::Matrix &coarseMatrix =
          hierarchy.A[level + 1];

      std::cout << "P = " << P.rows() << " x " << P.cols()
                << ", nnz = " << P.nonZeros() << "\n";

      std::cout << "R = " << R.rows() << " x " << R.cols()
                << ", nnz = " << R.nonZeros() << "\n";

      if (P.rows() != levelMatrix.rows()) {
        throw std::runtime_error("Prolongation rows do not match fine level.");
      }

      if (P.cols() != coarseMatrix.rows()) {
        throw std::runtime_error(
            "Prolongation columns do not match coarse level.");
      }

      if (R.rows() != coarseMatrix.rows()) {
        throw std::runtime_error("Restriction rows do not match coarse level.");
      }

      if (R.cols() != levelMatrix.rows()) {
        throw std::runtime_error(
            "Restriction columns do not match fine level.");
      }

      if (P.rows() % 3 != 0 || P.cols() % 3 != 0 || R.rows() % 3 != 0 ||
          R.cols() % 3 != 0) {

        throw std::runtime_error(
            "Transfer operators are incompatible with 3-DOF blocks.");
      }

      std::cout << "transfer dimensions: PASS\n";
    }
  }

  std::cout
      << "\n============================================================\n"
      << "ELASTICITY AMG HIERARCHY TEST: PASS\n"
      << "============================================================\n";
}

// -----------------------------------------------------------------------------
// Print hierarchy sanity only.
//
// For now we care much more about:
//   - can AMGCL build it?
//   - how many levels?
//   - are matrix dimensions sensible?
//   - for block smoothers: are all level sizes divisible by 3?
// -----------------------------------------------------------------------------

void testBlockRCMAndBlockJacobi(
    gpuSolver::Backend &backend,
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A,
    const Eigen::VectorXf &b, const Eigen::VectorXd &reference) {
  std::cout
      << "\n============================================================\n"
      << "[BLOCK RCM + BLOCK JACOBI TEST]\n"
      << "============================================================\n";

  if (A.rows() != A.cols()) {
    throw std::runtime_error("Block RCM test requires square matrix.");
  }

  if (A.rows() % 3 != 0) {
    throw std::runtime_error(
        "Block RCM test requires number of DOFs divisible by 3.");
  }

  const std::size_t numberOfNodes = static_cast<std::size_t>(A.rows()) / 3;

  // ============================================================
  // 1. Build the block graph.
  // ============================================================

  gpuSolver::BlockGraph blockGraph = gpuSolver::buildBlockGraph(A, 3);

  if (blockGraph.rowPtr.size() != numberOfNodes + 1) {

    throw std::runtime_error("Block graph has wrong number of rows.");
  }

  LOG(INFO) << "Block graph created:"
            << " nodes = " << numberOfNodes
            << ", edges entries = " << blockGraph.colPtr.size();

  // ============================================================
  // 2. Compute node-level RCM.
  // ============================================================

  std::vector<int> oldToNewNode(numberOfNodes);

  std::vector<int> newToOldNode(numberOfNodes);

  gpuSolver::RCMReordering rcm;

  rcm.compute(numberOfNodes, blockGraph.rowPtr.data(), blockGraph.colPtr.data(),
              oldToNewNode.data(), newToOldNode.data());

  gpuSolver::Permutation nodePermutation(numberOfNodes, oldToNewNode.data(),
                                         newToOldNode.data());

  LOG(INFO) << "Node-level RCM permutation created.";

  // ============================================================
  // 3. Expand node permutation to the 3N scalar DOFs.
  // ============================================================

  gpuSolver::Permutation blockPermutation =
      gpuSolver::expandBlockPermutation(nodePermutation, numberOfNodes, 3);

  const int *oldToNew = blockPermutation.oldToNew();

  const int *newToOld = blockPermutation.newToOld();

  // ============================================================
  // 4. Verify that triples stay together.
  // ============================================================

  for (std::size_t node = 0; node < numberOfNodes; ++node) {

    const int new0 = oldToNew[3 * node + 0];

    const int new1 = oldToNew[3 * node + 1];

    const int new2 = oldToNew[3 * node + 2];

    if (new0 % 3 != 0 || new1 != new0 + 1 || new2 != new0 + 2) {

      throw std::runtime_error("Expanded RCM permutation destroys 3x3 blocks.");
    }
  }

  // ============================================================
  // 5. Verify oldToNew/newToOld consistency.
  // ============================================================

  for (std::size_t oldDof = 0; oldDof < static_cast<std::size_t>(A.rows());
       ++oldDof) {

    const int newDof = oldToNew[oldDof];

    if (newToOld[newDof] != static_cast<int>(oldDof)) {

      throw std::runtime_error("Permutation inverse consistency check failed.");
    }
  }

  LOG(INFO) << "Block permutation structural checks passed.";

  // ============================================================
  // 6. Permute A.
  //
  // A' = P A P^T
  // ============================================================

  Eigen::SparseMatrix<float, Eigen::RowMajor> APermuted =
      permuteEigenSparseMatrix(A, blockPermutation);
  // gpuSolver::permuteMatrix(A, oldToNew, oldToNew);

  APermuted.makeCompressed();

  LOG(INFO) << "Permuted matrix created:"
            << " rows = " << APermuted.rows()
            << ", nnz = " << APermuted.nonZeros();

  // ============================================================
  // 7. Permute rhs and reference solution.
  // ============================================================

  const Eigen::VectorXf bPermuted = permuteVectorOldToNew(b, blockPermutation);

  const Eigen::VectorXd referencePermuted =
      permuteVectorOldToNew(reference, blockPermutation);

  // ============================================================
  // 8. Check matrix/permutation consistency.
  //
  // If
  //
  //     x' = P x
  //
  // then
  //
  //     A' x' = P (A x).
  // ============================================================

  Eigen::VectorXf x = Eigen::VectorXf::Random(A.rows());

  Eigen::VectorXf xPermuted = permuteVectorOldToNew(x, blockPermutation);

  Eigen::VectorXf y = A * x;

  Eigen::VectorXf yExpected = permuteVectorOldToNew(y, blockPermutation);

  Eigen::VectorXf yPermuted = APermuted * xPermuted;

  const float permutationError =
      (yPermuted - yExpected).norm() / std::max(1.0e-30f, yExpected.norm());

  LOG(INFO) << "Permutation matrix consistency error = " << permutationError;

  if (permutationError > 1.0e-5f) {
    throw std::runtime_error("Block permutation matrix consistency failed.");
  }

  // ============================================================
  // 9. Verify that the permuted matrix still has valid
  //    3x3 diagonal blocks.
  // ============================================================

  gpuSolver::HostCSRMatrix hostPermuted(
      static_cast<std::size_t>(APermuted.rows()),
      static_cast<std::size_t>(APermuted.cols()),
      static_cast<std::size_t>(APermuted.nonZeros()), APermuted.outerIndexPtr(),
      APermuted.innerIndexPtr(), APermuted.valuePtr());

  const std::vector<float> inverseBlocks =
      gpuSolver::extractInverseDiagonalBlocks3x3(hostPermuted);

  if (inverseBlocks.size() != 9 * numberOfNodes) {

    throw std::runtime_error("Wrong number of inverse 3x3 blocks.");
  }

  LOG(INFO) << "All permuted 3x3 diagonal blocks "
            << "could be extracted/inverted.";

  // ============================================================
  // 10. Repeated smoother/preconditioner application test.
  //
  // Same test we used before, but now on the block-RCM system.
  // ============================================================

  const std::vector<std::size_t> smoothingSteps = {1, 2, 4, 8, 16};

  constexpr float omega = 0.7f;

  LOG(INFO) << "Testing Block Jacobi after block RCM";

  // Adapt only this constructor line if your current
  // argument order differs.
  gpuSolver::BlockJacobiPreconditioner blockJacobi(1, 0.7f);

  // testOnePreconditionerApplication("3x3 block Jacobi + block RCM", backend,
  //                                  blockJacobi, APermuted, bPermuted,
  //                                  referencePermuted);

  std::cout << "\nBlock RCM + Block Jacobi test completed.\n";
}

void printHierarchy(const gpuSolver::AMGHierarchy &hierarchy) {
  std::cout
      << "\n============================================================\n"
      << "[AMG HIERARCHY]\n"
      << "============================================================\n";

  std::cout << "levels = " << hierarchy.A.size() << "\n";

  if (hierarchy.A.empty()) {
    throw std::runtime_error("AMG hierarchy contains no levels.");
  }

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {

    const auto &A = hierarchy.A[level];

    std::cout << "  level " << level << ": " << A.rows() << " x " << A.cols()
              << ", nnz = " << A.nonZeros();

    if (A.rows() % 3 == 0) {
      std::cout << ", blocks = " << A.rows() / 3;
    } else {
      std::cout << ", *** NOT DIVISIBLE BY 3 ***";
    }

    std::cout << "\n";

    if (level + 1 < hierarchy.A.size()) {
      std::cout << "    P: " << hierarchy.P[level].rows() << " x "
                << hierarchy.P[level].cols() << "\n"
                << "    R: " << hierarchy.R[level].rows() << " x "
                << hierarchy.R[level].cols() << "\n";
    }
  }
}

} // namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
  try {

    if (argc != 2) {
      std::cerr << "Usage:\n"
                << "  elasticitySolverDebug path/to/beam.mesh\n";

      return 1;
    }

    // Ensure the logs directory exists
    bool logForFile = false;
    if (logForFile) {

      FLAGS_logtostderr = false;
      FLAGS_alsologtostderr = true; // terminal + files
      FLAGS_log_prefix = true;
      FLAGS_logbufsecs = 0; // flush immediately
    } else {
      FLAGS_logtostderr = 1;
      FLAGS_alsologtostderr = 1;
    }
    google::InitGoogleLogging(argv[0]);
    const ElasticityProblem problem = buildProblem(argv[1]);

    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    // =========================================================================
    // STAGE 1:
    // Test ONE single-level preconditioner application.
    //
    // Start with scalar SGS because it is already established code.
    // Then uncomment Block Jacobi.
    // =========================================================================

    /*
    {
      gpuSolver::SymmetricGaussSeidelPreconditioner scalarGS(1.0f);

      LOG(INFO) << " Preconditioner initialized";
      testOnePreconditionerApplication("scalar symmetric Gauss-Seidel", backend,
                                       scalarGS, problem.A, problem.b,
                                       problem.reference);
    }
    */

    // -------------------------------------------------------------------------
    // Uncomment once scalar GS behaves sensibly.
    // -------------------------------------------------------------------------

    /*
    {
      // Adapt constructor if yours is (steps, omega) rather than (omega,steps).
      gpuSolver::BlockJacobiPreconditioner blockJacobi(
          2,     // fixedsmoothing steps
          0.7f); // damping

      testOnePreconditionerApplication("3x3 block Jacobi", backend, blockJacobi,
                                       problem.A, problem.b, problem.reference);

      testBlockRCMAndBlockJacobi(backend, problem.A, problem.b,
                                 problem.reference);
    }
    */

    // =========================================================================
    // STAGE 2:
    // Put the same preconditioner into CG.
    //
    // Keep maxIterations small initially. We are looking for:
    //   - finite numbers
    //   - residual going down
    //   - no immediate stagnation/breakdown
    // =========================================================================

    /*
    {
      gpuSolver::SymmetricGaussSeidelPreconditioner scalarGS(
          1.0f);

      testCG(
          "CG + scalar symmetric Gauss-Seidel",
          backend,
          scalarGS,
          problem.A,
          problem.b,
          problem.reference,
          100);
    }
    */

    /*
    {
      gpuSolver::BlockJacobiPreconditioner blockJacobi(2, 0.7f);

      testCG("CG + 3x3 block Jacobi", backend, blockJacobi, problem.A,
             problem.b, problem.reference, 1000);
    }
    */

    // =========================================================================
    // STAGE 3:
    // Build AMG hierarchy ONLY.
    //
    // IMPORTANT:
    // For block Jacobi on every MG level, the hierarchy must preserve a
    // meaningful 3-component block structure on every level.
    //
    // Replace the following builder call with the exact elasticity-aware AMGCL
    // helper currently in your project.
    // =========================================================================

    std::cout << "\nBuilding AMGCL elasticity hierarchy...\n";

    gpuSolver::SmoothedAggregationCoarsener coarsener(3, 0.1f);

    gpuSolver::AMGHierarchyBuilder builder(coarsener, 10, 1000);

    gpuSolver::AMGHierarchy hierarchy = builder.build(problem.A);
    // gpuSolver::AMGHierarchy hierarchy =
    //     MGBuilder::buildAmgclElasticityHierarchy(problem.A,
    //                                              10,    // max levels
    //                                              5000); // coarse threshold
    // testElasticityAMGHierarchy(problem.A);
    // printHierarchy(hierarchy);
    /**/

    // =========================================================================
    // STAGE 4:
    // Construct/apply MG preconditioner once.
    //
    // IMPORTANT:
    // We intentionally DO NOT use RCM/per-level permutations here.
    // If your current MG constructor *requires* a permutation vector, pass an
    // identity permutation per level rather than introducing RCM.
    // =========================================================================

    std::vector<gpuSolver::Permutation> permutationsLevel =
        buildIdentityPermutationsForHierarchy(hierarchy);

    {
      gpuSolver::MultigridBlockJacobiPreconditioner mgBlockJacobi(
          backend, hierarchy, permutationsLevel,
          2,     // pre smoothing steps
          2,     // post smoothing steps
          4,     // coarse smoothing steps
          0.7f); // omega

      gpuSolver::MultigridDampedJacobiPreconditioner mgJacobi(
          backend, hierarchy, permutationsLevel,
          1,     // pre smoothing steps
          1,     // post smoothing steps
          1,     // coarse smoothing steps
          0.7f); // omega

      gpuSolver::MultigridBlockGaussSeidelPreconditioner mgBlockGS(
          backend, hierarchy, permutationsLevel,
          1,     // pre smoothing
          1,     // post smoothing
          1,     // coarse smoothing
          1.0f); // omega
      // testOnePreconditionerApplication("MG + 3x3 block Jacobi", backend,
      //                                  mgBlockJacobi, hierarchy, problem.A,
      //                                  problem.b, problem.reference);
      LOG(INFO) << " do the CG test with the proper preconditioner! ";

      LOG(INFO) << " start CG";
      testCG("mg 3x3 block jacobi", backend, mgBlockJacobi, problem.A,
             problem.b, problem.reference, 5000);
      LOG(INFO) << " endCG";
      LOG(INFO) << "--------------------------------";

      LOG(INFO) << " start CG";
      testCG("mg jacobi", backend, mgJacobi, problem.A, problem.b,
             problem.reference, 1000);
      LOG(INFO) << " endCG";
      LOG(INFO) << "--------------------------------";

      LOG(INFO) << " start CG";
      testCG("mg 3x3 block gauss seidel - identity", backend, mgBlockGS,
             problem.A, problem.b, problem.reference, 1000);
      LOG(INFO) << " end CG";
      LOG(INFO) << "--------------------------------";
      LOG(INFO) << " start CG";
      std::vector<gpuSolver::Permutation> blockRCM =
          buildBlockRCMPermutationsForHierarchy(hierarchy);

      gpuSolver::MultigridBlockGaussSeidelPreconditioner mgBlockGSRCM(
          backend, hierarchy, blockRCM, 2, 2, 1, 1.0f);

      testCGWithPermutation("mg 3x3 block GS - block RCM", backend,
                            mgBlockGSRCM, problem.A, problem.b,
                            problem.reference, blockRCM[0], 1000);
      LOG(INFO) << " end CG";
      LOG(INFO) << "--------------------------------";
    }
    /**/

    // =========================================================================
    // STAGE 5:
    // CG + MG Block Jacobi.
    // =========================================================================

    /*
    {
      gpuSolver::MultigridBlockJacobiPreconditioner mgBlockJacobi(
          backend,
          hierarchy,
           identity permutations if required ,
          2,
          2,
          4,
          0.7f);

      testCG(
          "CG + MG + 3x3 block Jacobi",
          backend,
          mgBlockJacobi,
          problem.A,
          problem.b,
          problem.reference,
          100);
    }
    */

    std::cout << "\nDebug harness completed.\n";

    return 0;

  } catch (const std::exception &exception) {

    std::cerr << "\nFATAL: " << exception.what() << "\n";

    return 1;
  }
  google::ShutdownGoogleLogging();
}
