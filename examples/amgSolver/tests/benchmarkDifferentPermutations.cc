#include <AMGUtils/AMGHierarchyBuilder.h>
#include <AMGUtils/SmoothedAggregationCoarsener.h>

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>

#include <GPUSolver/ReorderingStrategies/HierarchicalReordering.h>
#include <GPUSolver/ReorderingStrategies/RCMReordering.h>

#include <GPUSolver/SparseMatrixUtils.h>

#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

// ============================================================
// RCM permutation for one matrix
// ============================================================

gpuSolver::Permutation computeRCMPermutation(const Matrix &A) {
  const std::size_t n = static_cast<std::size_t>(A.rows());

  int *oldToNew = new int[n];

  int *newToOld = new int[n];

  gpuSolver::RCMReordering rcm;

  rcm.compute(n, A.outerIndexPtr(), A.innerIndexPtr(), oldToNew, newToOld);

  gpuSolver::Permutation permutation(n, oldToNew, newToOld);

  delete[] oldToNew;
  delete[] newToOld;

  return permutation;
}

// ============================================================
// Permute vector:
//
//     xNew = P xOld
// ============================================================

Eigen::VectorXf permuteVector(const Eigen::VectorXf &x,
                              const gpuSolver::Permutation &permutation) {
  Eigen::VectorXf result(x.size());

  const int *oldToNew = permutation.oldToNew();

  for (Eigen::Index oldIndex = 0; oldIndex < x.size(); ++oldIndex) {
    const int newIndex = oldToNew[oldIndex];

    result[newIndex] = x[oldIndex];
  }

  return result;
}

// ============================================================
// Build DeviceCSRMatrix
// ============================================================

gpuSolver::DeviceCSRMatrix *createDeviceMatrix(gpuSolver::Backend &backend,
                                               const Matrix &A) {
  gpuSolver::HostCSRMatrix hostMatrix(
      static_cast<std::size_t>(A.rows()), static_cast<std::size_t>(A.cols()),
      static_cast<std::size_t>(A.nonZeros()), A.outerIndexPtr(),
      A.innerIndexPtr(), A.valuePtr());

  return backend.createCSRMatrix(hostMatrix);
}

// ============================================================
// Result of one benchmark
// ============================================================

struct BenchmarkResult {
  double millisecondsPerSpmv;
  float relativeError;
};

// ============================================================
// Benchmark many SpMVs inside one command submission.
//
// This is deliberate:
// we want primarily kernel throughput rather than command-buffer
// submission latency.
// ============================================================

BenchmarkResult benchmarkSpmv(gpuSolver::Backend &backend, const Matrix &A,
                              const Eigen::VectorXf &x, std::size_t repetitions,
                              std::size_t warmupRepetitions) {
  gpuSolver::DeviceCSRMatrix *deviceA = createDeviceMatrix(backend, A);

  gpuSolver::DeviceVector *deviceX =
      backend.createVector(static_cast<std::size_t>(x.size()), x.data());

  gpuSolver::DeviceVector *deviceY =
      backend.createVector(static_cast<std::size_t>(x.size()));

  // --------------------------------------------------------
  // Warmup
  // --------------------------------------------------------

  {
    gpuSolver::BackendEncoder *encoder = backend.createEncoder();

    for (std::size_t i = 0; i < warmupRepetitions; ++i) {
      backend.encodeSpmv(*encoder, *deviceA, *deviceX, *deviceY);
    }

    backend.submitAndWait(*encoder);

    delete encoder;
  }

  // --------------------------------------------------------
  // Timed section
  // --------------------------------------------------------

  gpuSolver::BackendEncoder *encoder = backend.createEncoder();

  const auto start = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < repetitions; ++i) {
    backend.encodeSpmv(*encoder, *deviceA, *deviceX, *deviceY);
  }

  backend.submitAndWait(*encoder);

  const auto end = std::chrono::high_resolution_clock::now();

  delete encoder;

  const double elapsedMilliseconds =
      std::chrono::duration<double, std::milli>(end - start).count();

  const double millisecondsPerSpmv =
      elapsedMilliseconds / static_cast<double>(repetitions);

  // --------------------------------------------------------
  // Correctness
  // --------------------------------------------------------

  const float *yData = deviceY->download();

  Eigen::VectorXf gpuY(x.size());

  for (Eigen::Index i = 0; i < gpuY.size(); ++i) {
    gpuY[i] = yData[i];
  }

  const Eigen::VectorXf cpuY = A * x;

  const float relativeError =
      (gpuY - cpuY).norm() / std::max(1.0f, cpuY.norm());

  delete deviceA;
  delete deviceX;
  delete deviceY;

  return {millisecondsPerSpmv, relativeError};
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: hierarchicalPermutationBenchmark mesh.obj\n";

    return 1;
  }

  // --------------------------------------------------------
  // Load mesh
  // --------------------------------------------------------

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  if (!igl::read_triangle_mesh(argv[1], V, F)) {
    throw std::runtime_error("Could not load triangle mesh.");
  }

  std::cout << "Loaded mesh with " << V.rows() << " vertices and " << F.rows()
            << " triangles.\n";

  // --------------------------------------------------------
  // Cotangent stiffness matrix
  // --------------------------------------------------------

  Eigen::SparseMatrix<double> LDouble;

  igl::cotmatrix(V, F, LDouble);

  Matrix K = (-LDouble).cast<float>();

  K.makeCompressed();

  // --------------------------------------------------------
  // Pin vertex 0, as in your Poisson example.
  // --------------------------------------------------------

  const Eigen::Index n = K.rows();

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

  std::cout << "\nFine matrix:\n"
            << "  DOFs = " << A.rows() << "\n"
            << "  nnz  = " << A.nonZeros() << "\n";

  // ========================================================
  // Build AMGCL hierarchy
  // ========================================================

  std::cout << "\nBuilding AMG hierarchy...\n";
  gpuSolver::SmoothedAggregationCoarsener coarsener(1, 0.1f);

  gpuSolver::AMGHierarchyBuilder builder(coarsener, 10, 1000);

  gpuSolver::AMGHierarchy hierarchy = builder.build(A);

  std::cout << "Hierarchy contains " << hierarchy.A.size() << " levels.\n";

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    std::cout << "  level " << level << ": " << hierarchy.A[level].rows()
              << " DOFs, " << hierarchy.A[level].nonZeros() << " nnz\n";
  }

  // ========================================================
  // Plain RCM permutation
  // ========================================================

  std::cout << "\nComputing finest-level RCM...\n";

  const gpuSolver::Permutation rcmPermutation = computeRCMPermutation(A);

  Matrix ARCM = gpuSolver::permuteMatrix(A, rcmPermutation.oldToNew(),
                                         rcmPermutation.oldToNew());

  // ========================================================
  // Nested hierarchy-aware permutation
  // ========================================================

  std::cout << "Computing hierarchical MG permutation...\n";

  gpuSolver::HierarchicalReordering hierarchicalReordering;

  const std::vector<gpuSolver::Permutation> hierarchicalPermutations =
      hierarchicalReordering.compute(hierarchy);

  if (hierarchicalPermutations.size() != hierarchy.A.size()) {
    throw std::runtime_error("Hierarchical permutation count does not "
                             "match hierarchy level count.");
  }

  const gpuSolver::Permutation &finestHierarchicalPermutation =
      hierarchicalPermutations[0];

  Matrix AMG =
      gpuSolver::permuteMatrix(A, finestHierarchicalPermutation.oldToNew(),
                               finestHierarchicalPermutation.oldToNew());

  // ========================================================
  // Same mathematical test vector in all three orderings
  // ========================================================

  Eigen::VectorXf x(A.rows());

  std::mt19937 generator(42);

  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (Eigen::Index i = 0; i < x.size(); ++i) {
    x[i] = distribution(generator);
  }

  const Eigen::VectorXf xRCM = permuteVector(x, rcmPermutation);

  const Eigen::VectorXf xMG = permuteVector(x, finestHierarchicalPermutation);

  // ========================================================
  // Backend
  // ========================================================

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  constexpr std::size_t warmup = 20;

  constexpr std::size_t repetitions = 200;

  // ========================================================
  // Benchmarks
  // ========================================================

  std::cout << "\nRunning SpMV benchmarks...\n";

  const BenchmarkResult originalResult =
      benchmarkSpmv(backend, A, x, repetitions, warmup);

  const BenchmarkResult rcmResult =
      benchmarkSpmv(backend, ARCM, xRCM, repetitions, warmup);

  const BenchmarkResult mgResult =
      benchmarkSpmv(backend, AMG, xMG, repetitions, warmup);

  // ========================================================
  // Results
  // ========================================================

  std::cout << "\n============================================\n"
            << "CSR SpMV permutation benchmark\n"
            << "============================================\n";

  std::cout << "\nOriginal:\n"
            << "  ms / SpMV       = " << originalResult.millisecondsPerSpmv
            << "\n"
            << "  relative error  = " << originalResult.relativeError << "\n";

  std::cout << "\nRCM:\n"
            << "  ms / SpMV       = " << rcmResult.millisecondsPerSpmv << "\n"
            << "  relative error  = " << rcmResult.relativeError << "\n";

  std::cout << "\nNested MG:\n"
            << "  ms / SpMV       = " << mgResult.millisecondsPerSpmv << "\n"
            << "  relative error  = " << mgResult.relativeError << "\n";

  std::cout << "\nRelative performance:\n"
            << "  RCM / original       = "
            << rcmResult.millisecondsPerSpmv /
                   originalResult.millisecondsPerSpmv
            << "\n"
            << "  Nested MG / original = "
            << mgResult.millisecondsPerSpmv / originalResult.millisecondsPerSpmv
            << "\n"
            << "  Nested MG / RCM      = "
            << mgResult.millisecondsPerSpmv / rcmResult.millisecondsPerSpmv
            << "\n";

  // --------------------------------------------------------
  // Minimal correctness criterion
  // --------------------------------------------------------

  constexpr float errorTolerance = 1e-5f;

  if (originalResult.relativeError > errorTolerance ||
      rcmResult.relativeError > errorTolerance ||
      mgResult.relativeError > errorTolerance) {
    std::cerr << "\n[FAIL] SpMV correctness check failed.\n";

    return 1;
  }

  std::cout << "\n[PASS] All three SpMV layouts are correct.\n";

  return 0;
}
