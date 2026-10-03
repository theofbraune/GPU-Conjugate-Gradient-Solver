
#include "AMGUtils/AMGHierarchyBuilder.h"
#include "GPUSolver/AMGHierarchy.h"
#include <GPUSolver/CGSolver.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/Preconditioners/DampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MetalJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridGaussSeidelPreconditioner.h>
#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>

#include <GPUSolver/ReorderingStrategies/RCMReordering.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <igl/cotmatrix.h>
#include <igl/grad.h>
#include <igl/read_triangle_mesh.h>

#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<gpuSolver::Permutation>
buildRCMPermutationsForHierarchy(const gpuSolver::AMGHierarchy &hierarchy) {
  std::vector<gpuSolver::Permutation> permutations;

  permutations.reserve(hierarchy.A.size());

  gpuSolver::RCMReordering rcmReordering;

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    const gpuSolver::AMGHierarchy::Matrix &A = hierarchy.A[level];

    const std::size_t nRows = static_cast<std::size_t>(A.rows());

    if (A.rows() != A.cols()) {
      throw std::runtime_error("buildRCMPermutationsForHierarchy: "
                               "Galerkin matrix is not square.");
    }

    int *oldToNew = new int[nRows];

    int *newToOld = new int[nRows];

    rcmReordering.compute(nRows, A.outerIndexPtr(), A.innerIndexPtr(), oldToNew,
                          newToOld);

    permutations.emplace_back(nRows, oldToNew, newToOld);

    delete[] oldToNew;
    delete[] newToOld;
  }

  return permutations;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: surfacePoissonTest mesh.obj\n";

    return 1;
  }

  // --------------------------------------------------------
  // Load triangle mesh.
  // --------------------------------------------------------

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  if (!igl::read_triangle_mesh(argv[1], V, F)) {
    throw std::runtime_error("Could not load triangle mesh.");
  }

  const Eigen::Index n = V.rows();

  if (n < 4) {
    throw std::runtime_error("Mesh is too small.");
  }

  std::cout << "Loaded mesh with " << V.rows() << " vertices and " << F.rows()
            << " triangles.\n";

  // --------------------------------------------------------
  // Construct cotangent Laplacian.
  //
  // libigl's cotmatrix uses the convention that L is
  // negative semidefinite.
  //
  // Therefore
  //
  //     K = -L
  //
  // is positive semidefinite.
  // --------------------------------------------------------

  Eigen::SparseMatrix<double> LDouble;

  igl::cotmatrix(V, F, LDouble);

  Eigen::SparseMatrix<float, Eigen::RowMajor> K = (-LDouble).cast<float>();

  K.makeCompressed();

  // --------------------------------------------------------
  // Pick three distinct random vertices carrying charges.
  // --------------------------------------------------------

  std::mt19937 generator(42);

  std::uniform_int_distribution<Eigen::Index> distribution(0, n - 1);

  Eigen::Index chargeVertex0 = distribution(generator);

  Eigen::Index chargeVertex1 = distribution(generator);

  while (chargeVertex1 == chargeVertex0) {
    chargeVertex1 = distribution(generator);
  }

  Eigen::Index chargeVertex2 = distribution(generator);

  while (chargeVertex2 == chargeVertex0 || chargeVertex2 == chargeVertex1) {
    chargeVertex2 = distribution(generator);
  }

  // --------------------------------------------------------
  // Build the point-charge RHS.
  //
  // Let's deliberately choose charges which do not sum to
  // zero first, and then explicitly project onto the image
  // of the Laplacian.
  // --------------------------------------------------------

  Eigen::VectorXf b = Eigen::VectorXf::Zero(n);

  b[chargeVertex0] = 1.0f;
  b[chargeVertex1] = 0.7f;
  b[chargeVertex2] = -0.4f;

  std::cout << "\nCharge vertices:\n"
            << "  v0 = " << chargeVertex0 << "\n"
            << "  v1 = " << chargeVertex1 << "\n"
            << "  v2 = " << chargeVertex2 << "\n";

  std::cout << "\nCharge sum before projection = " << b.sum() << "\n";

  // --------------------------------------------------------
  // Project RHS onto im(K).
  //
  // For a connected closed surface
  //
  //     ker(K) = span{1}
  //
  // so we remove the constant component:
  //
  //     b <- b - mean(b) * 1.
  // --------------------------------------------------------

  const float meanCharge = b.mean();

  b.array() -= meanCharge;

  std::cout << "Charge sum after projection  = " << b.sum() << "\n";

  // --------------------------------------------------------
  // K is still singular because constants are in its kernel.
  //
  // To give CG a genuinely SPD matrix, fix one vertex:
  //
  //     u[pinnedVertex] = 0.
  //
  // We construct the reduced (n-1)x(n-1) system.
  // --------------------------------------------------------

  const Eigen::Index pinnedVertex = 0;

  std::vector<Eigen::Index> oldToReduced(static_cast<std::size_t>(n), -1);

  Eigen::Index reducedIndex = 0;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    oldToReduced[static_cast<std::size_t>(i)] = reducedIndex;

    ++reducedIndex;
  }

  // --------------------------------------------------------
  // Build reduced sparse matrix.
  // --------------------------------------------------------

  using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

  std::vector<Eigen::Triplet<float>> triplets;

  triplets.reserve(static_cast<std::size_t>(K.nonZeros()));

  for (Eigen::Index row = 0; row < K.outerSize(); ++row) {
    for (Matrix::InnerIterator it(K, row); it; ++it) {
      const Eigen::Index i = it.row();

      const Eigen::Index j = it.col();

      if (i == pinnedVertex || j == pinnedVertex) {
        continue;
      }

      triplets.emplace_back(oldToReduced[static_cast<std::size_t>(i)],
                            oldToReduced[static_cast<std::size_t>(j)],
                            it.value());
    }
  }

  Matrix A(n - 1, n - 1);

  A.setFromTriplets(triplets.begin(), triplets.end());

  A.makeCompressed();

  // compute the permutation for A
  std::size_t nRowsA = n - 1;

  int *oldToNew = new int[nRowsA];
  int *newToOld = new int[nRowsA];
  const int *rowPtrA = A.outerIndexPtr();
  const int *colPtrA = A.innerIndexPtr();

  gpuSolver::RCMReordering rcmReordering;

  rcmReordering.compute(nRowsA, rowPtrA, colPtrA, oldToNew, newToOld);

  gpuSolver::Permutation permutation(nRowsA, oldToNew, newToOld);

  delete[] oldToNew;
  delete[] newToOld;

  // --------------------------------------------------------
  // Reduced RHS.
  //
  // Since the pinned value is zero, no boundary contribution
  // needs to be added.
  // --------------------------------------------------------

  Eigen::VectorXf bReduced(n - 1);

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    bReduced[oldToReduced[static_cast<std::size_t>(i)]] = b[i];
  }

  // --------------------------------------------------------
  // Initial guess.
  // --------------------------------------------------------

  Eigen::VectorXf xReduced = Eigen::VectorXf::Zero(n - 1);

  // --------------------------------------------------------
  // Set up Metal backend.
  // --------------------------------------------------------

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  // --------------------------------------------------------
  // Jacobi preconditioner.
  //
  // Adapt this one line to the exact constructor/API you
  // implemented.
  // --------------------------------------------------------

  std::size_t nRows = std::size_t(V.rows());

  float *diagonalValues = new float[nRows - 1];
  for (int i = 0; i < nRows - 1; i++) {
    diagonalValues[i] = A.coeff(i, i);
  }

  // gpuSolver::IdentityPreconditioner preconditioner;
  // gpuSolver::DampedJacobiPreconditioner preconditioner(4, 0.9f);

  // --------------------------------------------------------
  // GPU PCG.
  // --------------------------------------------------------

  constexpr float residualTolerance = 1e-4f;
  // gpuSolver::CGSolver solver(backend, preconditioner, A); //, permutation);

  std::cout << "\nStarting GPU PCG...\n";

  const Clock::time_point jacobiSetupStart = Clock::now();

  gpuSolver::DampedJacobiPreconditioner preconditioner(4, 0.9f);

  gpuSolver::CGSolver solver(backend, preconditioner, A, permutation);
  solver.setMaxIterations(2000);
  solver.setTolerance(residualTolerance);

  const double jacobiSetupTime = elapsedMs(jacobiSetupStart, Clock::now());
  const Clock::time_point jacobiSolveStart = Clock::now();

  solver.solve(bReduced, xReduced);

  const double jacobiSolveTime = elapsedMs(jacobiSolveStart, Clock::now());

  std::cout << "GPU PCG finished.\n";
  std::size_t nIter = solver.getNbOfIterations();
  std::cout << "finished in " << nIter << " iterations" << std::endl;

  // --------------------------------------------------------
  // Reconstruct full potential.
  // --------------------------------------------------------

  Eigen::VectorXf potential = Eigen::VectorXf::Zero(n);

  potential[pinnedVertex] = 0.0f;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    potential[i] = xReduced[oldToReduced[static_cast<std::size_t>(i)]];
  }

  // --------------------------------------------------------
  // Check residual against the ORIGINAL Laplacian equation.
  //
  //     r = K*u - b
  // --------------------------------------------------------

  const Eigen::VectorXf residual = K * potential - b;

  const float absoluteResidual = residual.norm();

  const float relativeResidual = absoluteResidual / b.norm();

  const float gpuTrueResidual =
      (A * xReduced - bReduced).norm() / bReduced.norm();

  std::cout << "\nSolver comparison:\n"
            << "  GPU iterations with jacobi Preconditioner             = "
            << solver.getNbOfIterations() << "\n"
            << "  GPU true relative residual with Jacobi Preconditioner  = "
            << gpuTrueResidual << "\n"
            << "\n";
  // constexpr float residualSanityTolerance = 5e-4f;

  constexpr float solutionComparisonTolerance = 5e-4f;

  bool passed = true;

  if (gpuTrueResidual > residualTolerance) {
    // std::cerr << "[FAIL] GPU true residual is too large: " << gpuTrueResidual
    //           << "\n";

    passed = false;
  }

  // if (relativeSolutionDifference > solutionComparisonTolerance) {
  //   std::cerr << "[FAIL] GPU solution differs too much from Eigen: "
  //             << relativeSolutionDifference << "\n";
  //
  //   passed = false;
  // }

  if (passed) {
    std::cout << "\n[PASS] Surface Poisson sanity check with Jacobi "
                 "Preconditioner.\n";
  } else {
    std::cerr << "\n[FAIL] Surface Poisson sanity check with Jacobi "
                 "Precondirtioner.\n";
    //
    // return 1;
  }

  // now build the AMG hierarchy for the preconditioner
  //
  // gpuSolver::AMGHierarchy hierarchyForA =
  // MGBuilder::buildAmgclSmoothedAggregationHierarchy(A, const
  // Eigen::SparseMatrix<float, Eigen::RowMajor> &M0, const Eigen::MatrixXf &V)

  // ========================================================
  // Multigrid damped-Jacobi preconditioner
  // ========================================================

  std::cout << "\nBuilding AMG hierarchy...\n";

  // --------------------------------------------------------
  // Build AMGCL smoothed-aggregation hierarchy.
  //
  // This is the scalar version:
  //     block_size = 1
  //     near-nullspace = constant vector
  // --------------------------------------------------------

  const Clock::time_point hierarchyStart = Clock::now();

  gpuSolver::AMGHierarchy hierarchyForA =
      MGBuilder::buildAmgclScalarSmoothedAggregationHierarchy(A, 10, 5000);

  const double hierarchyTime = elapsedMs(hierarchyStart, Clock::now());

  std::cout << "AMG hierarchy construction: " << hierarchyTime << " ms\n";
  std::cout << "AMG hierarchy contains " << hierarchyForA.A.size()
            << " levels.\n";

  for (std::size_t level = 0; level < hierarchyForA.A.size(); ++level) {
    std::cout << "  level " << level << ": " << hierarchyForA.A[level].rows()
              << " DOFs, " << hierarchyForA.A[level].nonZeros()
              << " nonzeros\n";
  }
  // std::vector<gpuSolver::Permutation> mgPermutations =
  //     gpuSolver::buildHierarchicalPermutations(hierarchyForA);
  const Clock::time_point permutationStart = Clock::now();

  std::vector<gpuSolver::Permutation> mgPermutations =
      buildRCMPermutationsForHierarchy(hierarchyForA);

  const double permutationTime = elapsedMs(permutationStart, Clock::now());

  std::cout << "AMG permutations: " << permutationTime << " ms\n";

  std::cout << " there are " << mgPermutations.size() << " permutations "
            << std::endl;

  std::cout << "  requested CG tolerance      = " << solver.tolerance() << "\n";

  const Clock::time_point mgJacobiSetupStart = Clock::now();

  gpuSolver::MultigridDampedJacobiPreconditioner prec(
      backend, hierarchyForA, mgPermutations, 2, 2, 1, 0.7f);

  gpuSolver::CGSolver solverMG(backend, prec, A, mgPermutations[0]);

  const double mgJacobiSetupTime = elapsedMs(mgJacobiSetupStart, Clock::now());

  solverMG.setMaxIterations(5000);

  solverMG.setTolerance(residualTolerance);

  std::cout << "\nStarting GPU PCG...\n";
  Eigen::VectorXf xReducedMG = Eigen::VectorXf::Zero(n - 1);

  const Clock::time_point mgJacobiSolveStart = Clock::now();

  solverMG.solve(bReduced, xReducedMG);

  const double mgJacobiSolveTime = elapsedMs(mgJacobiSolveStart, Clock::now());

  std::cout << "GPU PCG finished.\n";
  nIter = solverMG.getNbOfIterations();
  std::cout << "finished in " << nIter << " iterations" << std::endl;

  // --------------------------------------------------------
  // Reconstruct full potential.
  // --------------------------------------------------------

  potential = Eigen::VectorXf::Zero(n);

  potential[pinnedVertex] = 0.0f;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    potential[i] = xReduced[oldToReduced[static_cast<std::size_t>(i)]];
  }

  const Eigen::VectorXf residualMG = K * potential - b;

  const float absoluteResidualMG = residualMG.norm();

  const float relativeResidualMG = absoluteResidualMG / b.norm();

  const float gpuTrueResidualMG =
      (A * xReduced - bReduced).norm() / bReduced.norm();

  std::cout
      << "\nSolver comparison:\n"
      << "  GPU iterations with damped jacobi MG Preconditioner             = "
      << solverMG.getNbOfIterations() << "\n"
      << "  GPU true relative residual with damped Jacobi MG Preconditioner  = "
      << gpuTrueResidualMG << "\n"
      << "\n";
  // constexpr float residualSanityTolerance = 5e-4f;

  passed = true;

  if (gpuTrueResidualMG > residualTolerance) {
    // std::cerr << "[FAIL] GPU true residual is too large: " << gpuTrueResidual
    //           << "\n";

    passed = false;
  }

  // if (relativeSolutionDifference > solutionComparisonTolerance) {
  //   std::cerr << "[FAIL] GPU solution differs too much from Eigen: "
  //             << relativeSolutionDifference << "\n";
  //
  //   passed = false;
  // }

  if (passed) {
    std::cout << "\n[PASS] Surface Poisson sanity check with Multigrid Jacobi "
                 "Preconditioner.\n";
  } else {
    std::cerr << "\n[FAIL] Surface Poisson sanity check with Multigrid Jacobi "
                 "Precondirtioner.\n";
    //
    // return 1;
  }

  // ========================================================
  // Single-level symmetric colored Gauss-Seidel
  // ========================================================

  std::cout << "\nStarting single-level symmetric Gauss-Seidel PCG...\n";

  // --------------------------------------------------------
  // Preconditioner.
  //
  // Each application performs:
  //   1. forward color sweep
  //   2. backward color sweep
  //
  // omega = 1 gives ordinary symmetric Gauss-Seidel.
  // --------------------------------------------------------
  const Clock::time_point gsSetupStart = Clock::now();

  gpuSolver::SymmetricGaussSeidelPreconditioner gsPreconditioner(1.0f);

  gpuSolver::CGSolver solverGS(backend, gsPreconditioner, A, permutation);

  const double gsSetupTime = elapsedMs(gsSetupStart, Clock::now());

  solverGS.setMaxIterations(2000);
  solverGS.setTolerance(residualTolerance);

  // --------------------------------------------------------
  // Solve from zero.
  // --------------------------------------------------------
  Eigen::VectorXf xReducedGS = Eigen::VectorXf::Zero(n - 1);

  const Clock::time_point gsSolveStart = Clock::now();

  solverGS.solve(bReduced, xReducedGS);

  const double gsSolveTime = elapsedMs(gsSolveStart, Clock::now());
  std::cout << "GPU PCG with symmetric Gauss-Seidel finished.\n";

  // --------------------------------------------------------
  // Reconstruct potential on the original surface.
  // --------------------------------------------------------

  Eigen::VectorXf potentialGS = Eigen::VectorXf::Zero(n);

  potentialGS[pinnedVertex] = 0.0f;

  for (Eigen::Index i = 0; i < n; ++i) {

    if (i == pinnedVertex) {
      continue;
    }

    potentialGS[i] = xReducedGS[oldToReduced[static_cast<std::size_t>(i)]];
  }

  // --------------------------------------------------------
  // Check residuals.
  // --------------------------------------------------------

  const Eigen::VectorXf residualGS = K * potentialGS - b;

  const float absoluteResidualGS = residualGS.norm();

  const float relativeResidualGS = absoluteResidualGS / b.norm();

  const float gpuTrueResidualGS =
      (A * xReducedGS - bReduced).norm() / bReduced.norm();

  // --------------------------------------------------------
  // Print results.
  // --------------------------------------------------------

  std::cout
      << "\nSolver comparison:\n"
      << "  GPU iterations with symmetric GS Preconditioner             = "
      << solverGS.getNbOfIterations() << "\n"
      << "  GPU true relative residual with symmetric GS Preconditioner = "
      << gpuTrueResidualGS << "\n"
      << "  Full Laplacian relative residual                            = "
      << relativeResidualGS << "\n"
      << "\n";

  // --------------------------------------------------------
  // Sanity check.
  // --------------------------------------------------------

  const float residualSanityTolerance = 5e-4f;

  bool passedGS = std::isfinite(gpuTrueResidualGS) &&
                  gpuTrueResidualGS < residualSanityTolerance;

  if (passedGS) {
    std::cout << "\n[PASS] Surface Poisson sanity check with "
              << "symmetric Gauss-Seidel Preconditioner.\n";
  } else {
    std::cerr << "\n[FAIL] Surface Poisson sanity check with "
              << "symmetric Gauss-Seidel Preconditioner.\n";
  }

  // ========================================================
  // Multigrid symmetric colored Gauss-Seidel
  // ========================================================

  std::cout << "\nStarting multigrid symmetric Gauss-Seidel PCG...\n";

  // --------------------------------------------------------
  // Multigrid parameters.
  //
  // For the first comparison:
  //   1 symmetric GS pre-smoothing step
  //   1 symmetric GS post-smoothing step
  //   1 V-cycle per preconditioner application
  //
  // Each symmetric GS step already performs a complete
  // forward and backward color sweep.
  // --------------------------------------------------------

  constexpr std::size_t mgGSPreSmoothingSteps = 1;
  constexpr std::size_t mgGSPostSmoothingSteps = 1;
  constexpr std::size_t mgGSVcycles = 1;

  constexpr float mgGSOmega = 1.0f;

  // --------------------------------------------------------
  // Construct MG-GS preconditioner.
  // --------------------------------------------------------

  const Clock::time_point mgGSSetupStart = Clock::now();

  gpuSolver::MultigridGaussSeidelPreconditioner mgGSPreconditioner(
      backend, hierarchyForA, mgPermutations, 1, 1, 1, 1.0f);

  gpuSolver::CGSolver solverMGGS(backend, mgGSPreconditioner, A,
                                 mgPermutations[0]);

  const double mgGSSetupTime = elapsedMs(mgGSSetupStart, Clock::now());
  // --------------------------------------------------------
  // Construct CG with the same level-0 permutation as MG.
  // --------------------------------------------------------

  solverMGGS.setMaxIterations(500);
  solverMGGS.setTolerance(residualTolerance);

  // --------------------------------------------------------
  // Solve from zero.
  // --------------------------------------------------------
  Eigen::VectorXf xReducedMGGS = Eigen::VectorXf::Zero(n - 1);

  const Clock::time_point mgGSSolveStart = Clock::now();

  solverMGGS.solve(bReduced, xReducedMGGS);

  const double mgGSSolveTime = elapsedMs(mgGSSolveStart, Clock::now());

  std::cout << "GPU PCG with multigrid symmetric Gauss-Seidel finished.\n";

  // --------------------------------------------------------
  // Reconstruct potential.
  // --------------------------------------------------------

  Eigen::VectorXf potentialMGGS = Eigen::VectorXf::Zero(n);

  potentialMGGS[pinnedVertex] = 0.0f;

  for (Eigen::Index i = 0; i < n; ++i) {

    if (i == pinnedVertex) {
      continue;
    }

    potentialMGGS[i] = xReducedMGGS[oldToReduced[static_cast<std::size_t>(i)]];
  }

  // --------------------------------------------------------
  // Check residuals.
  // --------------------------------------------------------

  const Eigen::VectorXf residualMGGS = K * potentialMGGS - b;

  const float absoluteResidualMGGS = residualMGGS.norm();

  const float relativeResidualMGGS = absoluteResidualMGGS / b.norm();

  const float gpuTrueResidualMGGS =
      (A * xReducedMGGS - bReduced).norm() / bReduced.norm();

  // --------------------------------------------------------
  // Print results.
  // --------------------------------------------------------

  std::cout
      << "\nSolver comparison:\n"
      << "  GPU iterations with MG symmetric GS Preconditioner             = "
      << solverMGGS.getNbOfIterations() << "\n"
      << "  GPU true relative residual with MG symmetric GS Preconditioner = "
      << gpuTrueResidualMGGS << "\n"
      << "  Full Laplacian relative residual                               = "
      << relativeResidualMGGS << "\n"
      << "\n";

  // --------------------------------------------------------
  // Sanity check.
  // --------------------------------------------------------

  bool passedMGGS = std::isfinite(gpuTrueResidualMGGS) &&
                    gpuTrueResidualMGGS < residualSanityTolerance;

  if (passedMGGS) {
    std::cout << "\n[PASS] Surface Poisson sanity check with "
              << "multigrid symmetric Gauss-Seidel Preconditioner.\n";
  } else {
    std::cerr << "\n[FAIL] Surface Poisson sanity check with "
              << "multigrid symmetric Gauss-Seidel Preconditioner.\n";
  }
  // ========================================================
  // Final comparison
  // ========================================================

  std::cout
      << "\n==================================================================="
         "=============================================\n"
      << "                                       PRECONDITIONER COMPARISON\n"
      << "====================================================================="
         "===========================================\n";

  std::cout << std::left << std::setw(28) << "Preconditioner" << std::right
            << std::setw(12) << "Iterations" << std::setw(18) << "Setup (ms)"
            << std::setw(18) << "Solve (ms)" << std::setw(18) << "Total (ms)"
            << std::setw(20) << "True residual"
            << "\n";

  std::cout << "---------------------------------------------------------------"
               "-------------------------------------------------\n";

  std::cout << std::fixed << std::setprecision(3);

  std::cout << std::left << std::setw(28) << "Damped Jacobi" << std::right
            << std::setw(12) << solver.getNbOfIterations() << std::setw(18)
            << jacobiSetupTime << std::setw(18) << jacobiSolveTime
            << std::setw(18) << jacobiSetupTime + jacobiSolveTime
            << std::setw(20) << gpuTrueResidual << "\n";

  std::cout << std::left << std::setw(28) << "Symmetric Gauss-Seidel"
            << std::right << std::setw(12) << solverGS.getNbOfIterations()
            << std::setw(18) << gsSetupTime << std::setw(18) << gsSolveTime
            << std::setw(18) << gsSetupTime + gsSolveTime << std::setw(20)
            << gpuTrueResidualGS << "\n";

  std::cout << std::left << std::setw(28) << "MG + damped Jacobi" << std::right
            << std::setw(12) << solverMG.getNbOfIterations() << std::setw(18)
            << mgJacobiSetupTime << std::setw(18) << mgJacobiSolveTime
            << std::setw(18) << mgJacobiSetupTime + mgJacobiSolveTime
            << std::setw(20) << gpuTrueResidualMG << "\n";

  std::cout << std::left << std::setw(28) << "MG + symmetric GS" << std::right
            << std::setw(12) << solverMGGS.getNbOfIterations() << std::setw(18)
            << mgGSSetupTime << std::setw(18) << mgGSSolveTime << std::setw(18)
            << mgGSSetupTime + mgGSSolveTime << std::setw(20)
            << gpuTrueResidualMGGS << "\n";

  std::cout << "==============================================================="
               "=================================================\n";

  std::cout << "\nShared AMG preprocessing:\n"
            << "  AMGCL hierarchy construction = " << hierarchyTime << " ms\n"
            << "  RCM permutations             = " << permutationTime
            << " ms\n";

  // --------------------------------------------------------
  // Compute electric field
  //
  //     E = -grad(u)
  //
  // For piecewise-linear vertex potentials, grad(u) is
  // piecewise constant on each triangle.
  // --------------------------------------------------------

  Eigen::SparseMatrix<double> gradientOperator;

  igl::grad(V, F, gradientOperator);

  const Eigen::VectorXd potentialDouble = potential.cast<double>();

  const Eigen::VectorXd gradientFlat = gradientOperator * potentialDouble;

  // libigl stores the gradient components in blocks:
  //
  //     [ gx ]
  //     [ gy ]
  //     [ gz ]
  //
  // each block has #F entries.
  //
  // Convert this to an #F x 3 matrix for Polyscope.
  // --------------------------------------------------------

  Eigen::MatrixXd electricField(F.rows(), 3);

  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    electricField(f, 0) = -gradientFlat[f];

    electricField(f, 1) = -gradientFlat[f + F.rows()];

    electricField(f, 2) = -gradientFlat[f + 2 * F.rows()];
  }

  // --------------------------------------------------------
  // Positions of the three original point charges.
  // --------------------------------------------------------

  Eigen::MatrixXd chargePositions(3, 3);

  chargePositions.row(0) = V.row(chargeVertex0);

  chargePositions.row(1) = V.row(chargeVertex1);

  chargePositions.row(2) = V.row(chargeVertex2);

  Eigen::VectorXd chargeValues(3);

  chargeValues << 1.0, 0.7, -0.4;

  // ========================================================
  // Polyscope visualization
  // ========================================================

  polyscope::init();

  // --------------------------------------------------------
  // Surface mesh.
  // --------------------------------------------------------

  polyscope::SurfaceMesh *psMesh =
      polyscope::registerSurfaceMesh("surface", V, F);

  // --------------------------------------------------------
  // Potential u.
  // --------------------------------------------------------

  polyscope::SurfaceVertexScalarQuantity *potentialQuantity =
      psMesh->addVertexScalarQuantity("potential", potential);

  potentialQuantity->setEnabled(true);

  // --------------------------------------------------------
  // Projected charge density.
  //
  // This is the actual RHS which entered the Poisson solve.
  // --------------------------------------------------------

  polyscope::SurfaceVertexScalarQuantity *chargeDensityQuantity =
      psMesh->addVertexScalarQuantity("projected charge density", b);

  chargeDensityQuantity->setEnabled(false);

  // --------------------------------------------------------
  // Electric field.
  //
  //     E = -grad(u)
  //
  // This is naturally a per-face vector field.
  // --------------------------------------------------------

  polyscope::SurfaceFaceVectorQuantity *electricFieldQuantity =
      psMesh->addFaceVectorQuantity("electric field", electricField);

  electricFieldQuantity->setEnabled(true);

  electricFieldQuantity->setVectorLengthScale(0.03);

  // --------------------------------------------------------
  // Original point charges.
  //
  // Show the three selected vertices as a separate point
  // cloud so they remain clearly visible on top of the mesh.
  // --------------------------------------------------------

  polyscope::PointCloud *chargeCloud =
      polyscope::registerPointCloud("point charges", chargePositions);

  chargeCloud->setPointRadius(0.015, true);

  // Store the signed charge values on the point cloud as
  // another scalar quantity.
  polyscope::PointCloudScalarQuantity *pointChargeQuantity =
      chargeCloud->addScalarQuantity("charge", chargeValues);

  pointChargeQuantity->setEnabled(true);

  // --------------------------------------------------------
  // Launch viewer.
  // --------------------------------------------------------

  polyscope::show();

  delete[] diagonalValues;
  return 0;
}
