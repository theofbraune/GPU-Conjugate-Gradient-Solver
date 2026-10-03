#include <Eigen/SparseCore>
#include <cstddef>
#include <iostream>

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

#include <igl/boundary_facets.h>
#include <igl/cotmatrix.h>
#include <igl/grad.h>
#include <igl/readMESH.h>

#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>

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

struct BenchmarkResult {
  std::string name;

  std::size_t iterations = 0;

  double setupMs = 0.0;
  double solveMs = 0.0;

  float trueRelativeResidual = 0.0f;

  // Full temperature, including prescribed boundary values.
  Eigen::VectorXf temperature;
};

BenchmarkResult
benchmarkSolver(const std::string &name, gpuSolver::CGSolver &solver,
                const Eigen::SparseMatrix<float, Eigen::RowMajor> &A,
                const Eigen::VectorXf &bReduced,
                const Eigen::VectorXf &boundaryTemperature,
                const std::vector<int> &indexReduced, double setupMs) {
  BenchmarkResult result;

  result.name = name;
  result.setupMs = setupMs;

  // --------------------------------------------------------
  // Start every solver from the same initial guess.
  // --------------------------------------------------------

  Eigen::VectorXf xReduced = Eigen::VectorXf::Zero(A.rows());

  // --------------------------------------------------------
  // Solve and measure elapsed wall-clock time.
  // --------------------------------------------------------

  std::cout << "\nStarting " << name << "...\n";

  const Clock::time_point solveStart = Clock::now();

  solver.solve(bReduced, xReduced);

  result.solveMs = elapsedMs(solveStart, Clock::now());

  result.iterations = solver.getNbOfIterations();

  std::cout << name << " finished.\n";

  // --------------------------------------------------------
  // Compute the true reduced-system residual.
  // --------------------------------------------------------

  const Eigen::VectorXf residual = A * xReduced - bReduced;

  const float denominator = bReduced.norm();

  result.trueRelativeResidual =
      denominator > 0.0f ? residual.norm() / denominator : residual.norm();

  // --------------------------------------------------------
  // Reconstruct the full temperature field.
  // --------------------------------------------------------

  result.temperature = boundaryTemperature;

  for (Eigen::Index i = 0; i < boundaryTemperature.size(); ++i) {

    const int reducedIndex = indexReduced[static_cast<std::size_t>(i)];

    if (reducedIndex < 0) {
      continue;
    }

    result.temperature[i] = xReduced[reducedIndex];
  }

  // --------------------------------------------------------
  // Print individual results.
  // --------------------------------------------------------

  std::cout << "  Iterations     : " << result.iterations << "\n"
            << "  Setup time     : " << result.setupMs << " ms\n"
            << "  Solve time     : " << result.solveMs << " ms\n"
            << "  True residual  : " << std::scientific
            << result.trueRelativeResidual << std::defaultfloat << "\n";

  if (!xReduced.allFinite()) {
    std::cerr << "[FAIL] Solution contains NaN or Inf.\n";
  }

  return result;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: tetrahedralPoissonTest geometry.mesh\n";

    return 1;
  }

  // --------------------------------------------------------
  // Load tet mesh.
  // --------------------------------------------------------

  Eigen::MatrixXd V;
  Eigen::MatrixXi F, T;

  if (!igl::readMESH(argv[1], V, T, F)) {
    throw std::runtime_error("Could not load triangle mesh.");
  }

  const Eigen::Index n = V.rows();

  if (n < 4) {
    throw std::runtime_error("Mesh is too small.");
  }

  std::cout << "Loaded mesh with " << V.rows() << " vertices and " << T.rows()
            << "tetrahedra.\n";

  // create the laplacian and the mass matrix for the problem
  Eigen::SparseMatrix<float> L;
  igl::cotmatrix(V, T, L);

  Eigen::SparseMatrix<float, Eigen::RowMajor> K = -L;

  K.makeCompressed();

  Eigen::MatrixXi boundaryFacets;
  igl::boundary_facets(T, boundaryFacets);

  std::cout << " tjere are " << boundaryFacets.rows() << " many boundary faces"
            << std::endl;
  std::vector<bool> isVertexBoundary(static_cast<std::size_t>(V.rows()), false);
  for (std::size_t idxFace = 0; idxFace < boundaryFacets.rows(); idxFace++) {

    std::size_t i0 = static_cast<std::size_t>(boundaryFacets(idxFace, 0));
    std::size_t i1 = static_cast<std::size_t>(boundaryFacets(idxFace, 1));
    std::size_t i2 = static_cast<std::size_t>(boundaryFacets(idxFace, 2));
    isVertexBoundary[i0] = true;
    isVertexBoundary[i1] = true;
    isVertexBoundary[i2] = true;
  }

  // --------------------------------------------------------
  // Compute the bounding-box diagonal.
  // --------------------------------------------------------

  const Eigen::Vector3d bboxMin = V.colwise().minCoeff().transpose();

  const Eigen::Vector3d bboxMax = V.colwise().maxCoeff().transpose();

  const double meshSize = (bboxMax - bboxMin).norm();

  std::cout << "Mesh bounding-box diagonal: " << meshSize << std::endl;

  // --------------------------------------------------------
  // Random generator.
  //
  // Fixed seed for reproducible benchmarks.
  // --------------------------------------------------------

  std::mt19937 rng(40);

  // Random peak temperature.
  std::uniform_real_distribution<float> peakDist(0.2, 0.8);

  // Random Gaussian width, relative to mesh size.
  std::uniform_real_distribution<float> sigmaDist(0.08 * meshSize,
                                                  0.25 * meshSize);

  std::vector<int> boundaryVertices;

  for (int i = 0; i < V.rows(); ++i) {
    if (isVertexBoundary[i]) {
      boundaryVertices.push_back(i);
    }
  }

  if (boundaryVertices.empty()) {
    throw std::runtime_error("Tet mesh has no boundary vertices.");
  }

  // --------------------------------------------------------
  // Generate three Gaussian heat sources.
  // --------------------------------------------------------
  std::uniform_int_distribution<std::size_t> vertexDist(
      0, boundaryVertices.size() - 1);

  struct HeatSource {
    Eigen::Vector3f center;
    float peak;
    float sigma;
  };

  std::vector<HeatSource> heatSources;

  for (int k = 0; k < 3; ++k) {
    const int vertex = boundaryVertices[vertexDist(rng)];
    HeatSource source;

    // source.center = static_cast<Eigen::Vector3f>(V.row(vertex).transpose());
    source.center = Eigen::Vector3f(float(V(vertex, 0)), float(V(vertex, 1)),float(V(vertex, 2)));
    source.peak = peakDist(rng);
    source.sigma = sigmaDist(rng);

    heatSources.push_back(source);

    std::cout << "Heat source " << k << ":\n"
              << "  vertex = " << vertex << "\n"
              << "  peak   = " << source.peak << "\n"
              << "  sigma  = " << source.sigma << "\n";
  }

  // --------------------------------------------------------
  // Evaluate the boundary temperature distribution.
  // --------------------------------------------------------

  Eigen::VectorXf boundaryTemperature = Eigen::VectorXf::Zero(V.rows());

  for (Eigen::Index i = 0; i < V.rows(); ++i) {

    if (!isVertexBoundary[static_cast<std::size_t>(i)]) {
      continue;
    }

    const Eigen::Vector3f position = Eigen::Vector3f(float(V(i, 0)), float(V(i, 1)),float(V(i, 2)));

      V.row(i).transpose();

    float temperature = 0.0;

    for (const HeatSource &source : heatSources) {

      const float distSquared = (position - source.center).squaredNorm();

      const double variance = source.sigma * source.sigma;

      temperature += source.peak * std::exp(-distSquared / (2.0 * variance));
    }

    boundaryTemperature[i] = temperature;
  }

  std::cout << "Boundary temperature range: [" << boundaryTemperature.minCoeff()
            << ", " << boundaryTemperature.maxCoeff() << "]\n";

  Eigen::VectorXf rhsBoundary = -K * boundaryTemperature;

  // now build the reduced matrix for the reduced problem
  std::vector<Eigen::Triplet<float>> tripletsForInterior;
  std::vector<float> entriesForRHS;
  int counterInterior = 0;
  std::vector<int> indexReduced(V.rows(), -1);
  for (int i = 0; i < V.rows(); i++) {
    if (!isVertexBoundary[i]) {
      indexReduced[i] = counterInterior;
      entriesForRHS.push_back(rhsBoundary(i));
      counterInterior++;
    }
  }

  for (int outer = 0; outer < K.outerSize(); outer++) {
    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(K,
                                                                       outer);
         it; ++it) {
      const int i = it.row();
      const int j = it.col();
      const float val = it.value();

      const int reducedI = indexReduced[i];
      const int reducedJ = indexReduced[j];

      if (reducedI < 0 || reducedJ < 0) {
        continue;
      }
      tripletsForInterior.emplace_back(reducedI, reducedJ, val);
    }
  }

  int nVertexInterior = counterInterior;

  Eigen::VectorXf bReduced(nVertexInterior);
  for (int i = 0; i < V.rows(); i++) {
    const int reducedI = indexReduced[static_cast<size_t>(i)];
    if (reducedI < 0) {
      continue;
    }
    bReduced[reducedI] = rhsBoundary[i];
  }

  Eigen::SparseMatrix<float, Eigen::RowMajor> K_reduced(nVertexInterior,
                                                        nVertexInterior);

  K_reduced.setFromTriplets(tripletsForInterior.begin(),
                            tripletsForInterior.end());

  K_reduced.makeCompressed();
  std::cout << "\nReduced Dirichlet system:\n"
            << "  Total vertices    = " << V.rows() << "\n"
            << "  Boundary vertices = " << V.rows() - nVertexInterior << "\n"
            << "  Interior vertices = " << nVertexInterior << "\n"
            << "  Matrix nonzeros   = " << K_reduced.nonZeros() << "\n"
            << "  RHS norm          = " << rhsBoundary.norm() << "\n";

  // prepare tje solver
  //
  // ========================================================
  // Set up Metal backend
  // ========================================================

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  // ========================================================
  // Common solver parameters
  // ========================================================

  constexpr float residualTolerance = 1e-4f;

  constexpr std::size_t maxCGIterations = 2000;

  // ========================================================
  // Construct AMG hierarchy
  // ========================================================

  std::cout << "\nBuilding AMG hierarchy...\n";

  const Clock::time_point hierarchyStart = Clock::now();

  gpuSolver::AMGHierarchy hierarchyForA =
      MGBuilder::buildAmgclScalarSmoothedAggregationHierarchy(
          K_reduced,
          10,  // Maximum number of levels
          5000 // Coarsest-level DOF threshold
      );

  const double hierarchyTime = elapsedMs(hierarchyStart, Clock::now());

  std::cout << "AMG hierarchy construction: " << hierarchyTime << " ms\n";

  for (std::size_t level = 0; level < hierarchyForA.A.size(); ++level) {

    std::cout << "  Level " << level << ": " << hierarchyForA.A[level].rows()
              << " DOFs, " << hierarchyForA.A[level].nonZeros()
              << " nonzeros\n";
  }

  // ========================================================
  // Construct one RCM permutation per AMG level
  // ========================================================

  const Clock::time_point permutationStart = Clock::now();

  std::vector<gpuSolver::Permutation> mgPermutations =
      buildRCMPermutationsForHierarchy(hierarchyForA);

  const double permutationTime = elapsedMs(permutationStart, Clock::now());

  std::cout << "RCM permutation time: " << permutationTime << " ms\n";

  // ========================================================
  // 1. Single-level damped Jacobi
  // ========================================================

  BenchmarkResult jacobiResult;

  {
    const Clock::time_point setupStart = Clock::now();

    gpuSolver::DampedJacobiPreconditioner preconditioner(4, 0.9f);

    gpuSolver::CGSolver solver(backend, preconditioner, K_reduced,
                               mgPermutations[0]);

    solver.setMaxIterations(maxCGIterations);
    solver.setTolerance(residualTolerance);

    const double setupTime = elapsedMs(setupStart, Clock::now());

    jacobiResult =
        benchmarkSolver("Damped Jacobi", solver, K_reduced, bReduced,
                        boundaryTemperature, indexReduced, setupTime);
  }

  // ========================================================
  // 2. Single-level symmetric Gauss-Seidel
  // ========================================================

  BenchmarkResult gsResult;

  {
    const Clock::time_point setupStart = Clock::now();

    gpuSolver::SymmetricGaussSeidelPreconditioner preconditioner(1.0f);

    gpuSolver::CGSolver solver(backend, preconditioner, K_reduced,
                               mgPermutations[0]);

    solver.setMaxIterations(maxCGIterations);
    solver.setTolerance(residualTolerance);

    const double setupTime = elapsedMs(setupStart, Clock::now());

    gsResult =
        benchmarkSolver("Symmetric Gauss-Seidel", solver, K_reduced, bReduced,
                        boundaryTemperature, indexReduced, setupTime);
  }

  // ========================================================
  // 3. Multigrid damped Jacobi
  // ========================================================

  BenchmarkResult mgJacobiResult;

  {
    const Clock::time_point setupStart = Clock::now();

    gpuSolver::MultigridDampedJacobiPreconditioner preconditioner(
        backend, hierarchyForA, mgPermutations,
        2,   // Pre-smoothing steps
        2,   // Post-smoothing steps
        1,   // Number of V-cycles
        0.7f // Jacobi weight
    );

    gpuSolver::CGSolver solver(backend, preconditioner, K_reduced,
                               mgPermutations[0]);

    solver.setMaxIterations(maxCGIterations);
    solver.setTolerance(residualTolerance);

    const double setupTime = elapsedMs(setupStart, Clock::now());

    mgJacobiResult =
        benchmarkSolver("MG + damped Jacobi", solver, K_reduced, bReduced,
                        boundaryTemperature, indexReduced, setupTime);
  }

  // ========================================================
  // 4. Multigrid symmetric Gauss-Seidel
  // ========================================================

  BenchmarkResult mgGSResult;

  {
    const Clock::time_point setupStart = Clock::now();

    gpuSolver::MultigridGaussSeidelPreconditioner preconditioner(
        backend, hierarchyForA, mgPermutations,
        1,   // Pre-smoothing steps
        1,   // Post-smoothing steps
        1,   // Number of V-cycles
        1.0f // GS relaxation
    );

    gpuSolver::CGSolver solver(backend, preconditioner, K_reduced,
                               mgPermutations[0]);

    solver.setMaxIterations(maxCGIterations);
    solver.setTolerance(residualTolerance);

    const double setupTime = elapsedMs(setupStart, Clock::now());

    mgGSResult =
        benchmarkSolver("MG + symmetric GS", solver, K_reduced, bReduced,
                        boundaryTemperature, indexReduced, setupTime);
  }

  // ========================================================
  // Final performance comparison
  // ========================================================

  std::cout << "\n"
            << "==============================================================="
               "=============================================\n"
            << "                              TETRAHEDRAL POISSON BENCHMARK\n"
            << "==============================================================="
               "=============================================\n";

  std::cout << std::left << std::setw(28) << "Preconditioner" << std::right
            << std::setw(12) << "Iterations" << std::setw(16) << "Setup (ms)"
            << std::setw(16) << "Solve (ms)" << std::setw(16) << "Total (ms)"
            << std::setw(20) << "True residual"
            << "\n";

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------\n";

  const std::vector<BenchmarkResult> results = {jacobiResult, gsResult,
                                                mgJacobiResult, mgGSResult};

  for (const BenchmarkResult &result : results) {

    std::cout << std::left << std::setw(28) << result.name << std::right
              << std::setw(12) << result.iterations << std::fixed
              << std::setprecision(3) << std::setw(16) << result.setupMs
              << std::setw(16) << result.solveMs << std::setw(16)
              << result.setupMs + result.solveMs << std::scientific
              << std::setprecision(6) << std::setw(20)
              << result.trueRelativeResidual << std::defaultfloat << "\n";
  }

  std::cout << "==============================================================="
               "=============================================\n";

  std::cout << "\nShared AMG preprocessing:\n"
            << "  AMGCL hierarchy construction = " << hierarchyTime << " ms\n"
            << "  RCM permutations             = " << permutationTime
            << " ms\n";

  // ========================================================
  // Polyscope visualization
  // ========================================================

  polyscope::init();

  // --------------------------------------------------------
  // tet mesh.
  // --------------------------------------------------------

  polyscope::VolumeMesh *psMesh = polyscope::registerTetMesh("surface", V, T);

  psMesh->addVertexScalarQuantity("boundary heat", boundaryTemperature);

  // --------------------------------------------------------
  // Four reconstructed harmonic extensions.
  // --------------------------------------------------------

  psMesh->addVertexScalarQuantity("Damped Jacobi", jacobiResult.temperature);

  psMesh->addVertexScalarQuantity("Symmetric Gauss-Seidel",
                                  gsResult.temperature);

  psMesh
      ->addVertexScalarQuantity("MG + damped Jacobi",
                                mgJacobiResult.temperature)
      ->setEnabled(true);

  psMesh->addVertexScalarQuantity("MG + symmetric GS", mgGSResult.temperature);
  // --------------------------------------------------------
  // Launch viewer.
  // --------------------------------------------------------

  polyscope::show();

  return 0;
}
