#include "ElasticitySimulator.h"
#include "LinearElasticity.h"

#include "AMGUtils/AMGHierarchyBuilder.h"
#include <AMGUtils/SmoothedAggregationCoarsener.h>

#include <Eigen/SparseCholesky>
#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/BlockUtils.h>
#include <GPUSolver/CGSolver.h>
#include <GPUSolver/Permutation.h>

#include <GPUSolver/Preconditioners/DampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>

#include <GPUSolver/Preconditioners/BlockGaussSeidelPreconditioner.h>
#include <GPUSolver/Preconditioners/BlockJacobiPreconditioner.h>

#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/MultigridGaussSeidelPreconditioner.h>

// Adapt these two names to whatever you choose.
// #include
// <GPUSolver/Preconditioners/MultigridBlockGaussSeidelPreconditioner.h>
// #include <GPUSolver/Preconditioners/MultigridBlockJacobiPreconditioner.h>

#include <GPUSolver/ReorderingStrategies/RCMReordering.h>

#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <igl/readMESH.h>

#include <polyscope/polyscope.h>
#include <polyscope/volume_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchmarkResult {
  std::string name;

  std::size_t iterations = 0;

  double setupMs = 0.0;
  double solveMs = 0.0;

  float trueResidual = 0.0f;

  Eigen::VectorXf solution;
};

BenchmarkResult
runBenchmark(const std::string &name, gpuSolver::CGSolver &solver,
             const Eigen::SparseMatrix<float, Eigen::RowMajor> &A,
             const Eigen::VectorXf &b, double setupMs) {
  BenchmarkResult result;

  result.name = name;
  result.setupMs = setupMs;

  result.solution = Eigen::VectorXf::Zero(A.rows());

  std::cout << "\nStarting " << name << "...\n";

  const Clock::time_point start = Clock::now();

  solver.solve(b, result.solution);

  result.solveMs = elapsedMs(start, Clock::now());

  result.iterations = solver.getNbOfIterations();

  const Eigen::VectorXf trueResidual = A * result.solution - b;

  result.trueResidual = trueResidual.norm() / std::max(1.0e-20f, b.norm());

  std::cout << name << " finished.\n"
            << "  iterations    = " << result.iterations << "\n"
            << "  setup         = " << result.setupMs << " ms\n"
            << "  solve         = " << result.solveMs << " ms\n"
            << "  true residual = " << std::scientific << result.trueResidual
            << std::defaultfloat << "\n";

  return result;
}

std::vector<gpuSolver::Permutation>
buildBlockRCMPermutationsForHierarchy(const gpuSolver::AMGHierarchy &hierarchy,
                                      std::size_t blockSize) {
  std::vector<gpuSolver::Permutation> permutations;

  permutations.reserve(hierarchy.A.size());

  gpuSolver::RCMReordering rcm;

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {

    const auto &A = hierarchy.A[level];

    if (A.rows() % static_cast<int>(blockSize) != 0) {

      throw std::runtime_error("AMG level does not preserve "
                               "block structure.");
    }

    gpuSolver::BlockGraph graph = gpuSolver::buildBlockGraph(A, blockSize);

    const std::size_t numberOfBlocks = graph.rowPtr.size() - 1;

    int *oldToNew = new int[numberOfBlocks];

    int *newToOld = new int[numberOfBlocks];

    rcm.compute(numberOfBlocks, graph.rowPtr.data(), graph.colPtr.data(),
                oldToNew, newToOld);

    gpuSolver::Permutation nodePermutation(numberOfBlocks, oldToNew, newToOld);

    delete[] oldToNew;
    delete[] newToOld;

    permutations.push_back(gpuSolver::expandBlockPermutation(
        nodePermutation, numberOfBlocks, blockSize));
  }

  return permutations;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: cantileverSolverBenchmark beam.mesh\n";

    return 1;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi T;
  Eigen::MatrixXi F;

  if (!igl::readMESH(argv[1], V, T, F)) {

    throw std::runtime_error("Could not read tetrahedral mesh.");
  }

  std::cout << "Loaded elasticity mesh:\n"
            << "  vertices = " << V.rows() << "\n"
            << "  tets     = " << T.rows() << "\n";
  constexpr double youngModulus = 1.0e5;

  constexpr double poissonRatio = 0.40;

  const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));

  const double lambda = youngModulus * poissonRatio /
                        ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

  std::cout << "Material:\n"
            << "  E      = " << youngModulus << "\n"
            << "  nu     = " << poissonRatio << "\n"
            << "  lambda = " << lambda << "\n"
            << "  mu     = " << mu << "\n";

  LinearElasticitySimulator simulator(V, F, T, lambda, mu);

  const double xMin = V.col(0).minCoeff();

  const double xMax = V.col(0).maxCoeff();

  const double beamLength = xMax - xMin;

  const double clampTolerance = 1.0e-4 * beamLength;

  std::vector<int> pinnedVertices;

  for (Eigen::Index i = 0; i < V.rows(); ++i) {

    if (V(i, 0) <= xMin + clampTolerance) {

      pinnedVertices.push_back(static_cast<int>(i));
    }
  }

  std::cout << "Pinned " << pinnedVertices.size() << " vertices.\n";

  if (pinnedVertices.empty()) {
    throw std::runtime_error("No vertices found on clamped side.");
  }

  simulator.constrainSimulation(pinnedVertices);

  const Eigen::SparseMatrix<double, Eigen::RowMajor> Kdouble =
      simulator.getHessian();

  const Eigen::SparseMatrix<double, Eigen::RowMajor> Mdouble =
      simulator.getMassMatrixStackedConstrained();

  if (Kdouble.rows() % 3 != 0) {
    throw std::runtime_error("Constrained elasticity system "
                             "is not divisible into 3x3 blocks.");
  }

  std::cout << "Reduced elasticity system:\n"
            << "  DOFs     = " << Kdouble.rows() << "\n"
            << "  vertices = " << Kdouble.rows() / 3 << "\n"
            << "  nnz      = " << Kdouble.nonZeros() << "\n";

  constexpr double density = 1.0;

  constexpr double gravity = -9.81;

  Eigen::VectorXd gravityVector = Eigen::VectorXd::Zero(Kdouble.rows());

  const int numberOfFreeVertices = Kdouble.rows() / 3;

  for (int i = 0; i < numberOfFreeVertices; ++i) {

    gravityVector[3 * i + 2] = gravity;
  }

  Eigen::VectorXd forceDouble = density * Mdouble * gravityVector;

  Eigen::SparseMatrix<float, Eigen::RowMajor> A = Kdouble.cast<float>();

  A.makeCompressed();

  Eigen::VectorXf b = forceDouble.cast<float>();

  constexpr std::size_t blockSize = 3;

  gpuSolver::BlockGraph blockGraph = gpuSolver::buildBlockGraph(A, blockSize);

  const std::size_t numberOfBlocks = blockGraph.rowPtr.size() - 1;

  int *nodeOldToNew = new int[numberOfBlocks];

  int *nodeNewToOld = new int[numberOfBlocks];

  gpuSolver::RCMReordering rcm;

  rcm.compute(numberOfBlocks, blockGraph.rowPtr.data(),
              blockGraph.colPtr.data(), nodeOldToNew, nodeNewToOld);

  gpuSolver::Permutation nodePermutation(numberOfBlocks, nodeOldToNew,
                                         nodeNewToOld);

  delete[] nodeOldToNew;
  delete[] nodeNewToOld;

  gpuSolver::Permutation dofPermutation = gpuSolver::expandBlockPermutation(
      nodePermutation, numberOfBlocks, blockSize);

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  constexpr float tolerance = 1.0e-4f;

  constexpr std::size_t maxIterations = 1000;

  gpuSolver::SmoothedAggregationCoarsener coarsener(3, 0.1f);

  gpuSolver::AMGHierarchyBuilder builder(coarsener, 10, 1000);

  gpuSolver::AMGHierarchy hierarchy = builder.build(A);

  // gpuSolver::AMGHierarchy hierarchy =
  //     MGBuilder::buildAmgclBlockSmoothedAggregationHierarchy(A, 3);
  for (auto A : hierarchy.A) {
    std::cout << " the size of the matrix is " << A.rows() << " x " << A.cols()
              << std::endl;
  }

  std::vector<gpuSolver::Permutation> mgPermutations =
      buildBlockRCMPermutationsForHierarchy(hierarchy, 3);
  BenchmarkResult scalarJacobi;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::DampedJacobiPreconditioner preconditioner(4, 0.9f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, dofPermutation);

    solver.setTolerance(tolerance);

    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    scalarJacobi = runBenchmark("Scalar Jacobi", solver, A, b, setupMs);
  }

  BenchmarkResult scalarGS;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::SymmetricGaussSeidelPreconditioner preconditioner(1.0f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, dofPermutation);

    solver.setTolerance(tolerance);

    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    scalarGS = runBenchmark("Scalar symmetric GS", solver, A, b, setupMs);
  }
  /*
  BenchmarkResult blockJacobi;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::BlockJacobiPreconditioner preconditioner(4, 0.9f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, dofPermutation);

    solver.setTolerance(tolerance);

    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    blockJacobi = runBenchmark("3x3 block Jacobi", solver, A, b, setupMs);
  }

  BenchmarkResult blockGS;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::BlockGaussSeidelPreconditioner preconditioner(1.0f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, dofPermutation);

    solver.setTolerance(tolerance);

    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    blockGS = runBenchmark("3x3 block symmetric GS", solver, A, b, setupMs);
  }

  */
  const Clock::time_point hierarchyStart = Clock::now();

  const double hierarchyMs = elapsedMs(hierarchyStart, Clock::now());

  BenchmarkResult mgScalarJacobi;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::MultigridDampedJacobiPreconditioner preconditioner(
        backend, hierarchy, mgPermutations, 2, 2, 1, 0.7f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, mgPermutations[0]);

    solver.setTolerance(tolerance);
    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    mgScalarJacobi = runBenchmark("MG + scalar Jacobi", solver, A, b, setupMs);
  }

  /*
  BenchmarkResult mgBlockJacobi;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::MultigridBlockJacobiPreconditioner preconditioner(
        backend, hierarchy, mgPermutations, 2, 2, 1, 0.7f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, mgPermutations[0]);

    solver.setTolerance(tolerance);
    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    mgBlockJacobi =
        runBenchmark("MG + 3x3 block Jacobi", solver, A, b, setupMs);
  }
  */

  /*
  BenchmarkResult mgSGS;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::MultigridGaussSeidelPreconditioner preconditioner(
        backend, hierarchy, mgPermutations, 2, 2, 1, 0.7f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, mgPermutations[0]);

    solver.setTolerance(tolerance);
    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    mgSGS = runBenchmark("MG + gauss seidel", solver, A, b, setupMs);
  }



  BenchmarkResult mgBlockSGS;

  {
    const Clock::time_point start = Clock::now();

    gpuSolver::MultigridBlockGaussSeidelPreconditioner preconditioner(
        backend, hierarchy, mgPermutations, 2, 2, 1, 0.7f);

    gpuSolver::CGSolver solver(backend, preconditioner, A, mgPermutations[0]);

    solver.setTolerance(tolerance);
    solver.setMaxIterations(maxIterations);

    const double setupMs = elapsedMs(start, Clock::now());

    mgBlockSGS = runBenchmark("MG + gauss seidel", solver, A, b, setupMs);
  }


  std::vector<BenchmarkResult> results = {
      scalarJacobi,   scalarGS,   blockJacobi,   blockGS,
      mgScalarJacobi, mgSGS};

  std::cout << "\n"
            << "==============================================================="
               "================================\n"
            << "                         LINEAR ELASTICITY SOLVER BENCHMARK\n"
            << "==============================================================="
               "================================\n";

  std::cout << std::left << std::setw(30) << "Preconditioner" << std::right
            << std::setw(12) << "Iterations" << std::setw(15) << "Setup (ms)"
            << std::setw(15) << "Solve (ms)" << std::setw(15) << "Total (ms)"
            << std::setw(18) << "True residual"
            << "\n";

  for (const BenchmarkResult &result : results) {

    std::cout << std::left << std::setw(30) << result.name

              << std::right << std::setw(12) << result.iterations

              << std::fixed << std::setprecision(3)

              << std::setw(15) << result.setupMs

              << std::setw(15) << result.solveMs

              << std::setw(15) << result.setupMs + result.solveMs

              << std::scientific << std::setprecision(5)

              << std::setw(18) << result.trueResidual

              << std::defaultfloat << "\n";
  }

  std::cout << "==============================================================="
               "================================\n";

  std::cout << "AMG hierarchy construction = " << hierarchyMs << " ms\n";
  */
  Eigen::VectorXd constrainedSolution = mgScalarJacobi.solution.cast<double>();
  // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>>
  // lltA(Kdouble);

  // Eigen::VectorXd constrainedSolution = lltA.solve(forceDouble);

  Eigen::VectorXd fullSolution =
      simulator.liftConstrainedToFull(constrainedSolution);

  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(V.rows(), 3);

  for (Eigen::Index i = 0; i < V.rows(); ++i) {

    U.row(i) = fullSolution.segment<3>(3 * i).transpose();
  }

  const double maxDisp = U.rowwise().norm().maxCoeff();

  const double bboxDiagonal =
      (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();

  double visualizationScale = 1.0;

  if (maxDisp > 0.0) {
    visualizationScale = 0.15 * bboxDiagonal / maxDisp;
  }

  Eigen::MatrixXd Vdeformed = V + visualizationScale * U;
  polyscope::init();

  polyscope::registerTetMesh("Rest shape", V, T);

  polyscope::VolumeMesh *deformed =
      polyscope::registerTetMesh("Deformed shape", Vdeformed, T);

  Eigen::VectorXd displacementMagnitude = U.rowwise().norm();

  deformed
      ->addVertexScalarQuantity("Displacement magnitude", displacementMagnitude)
      ->setEnabled(true);

  polyscope::show();

  return 0;
}
