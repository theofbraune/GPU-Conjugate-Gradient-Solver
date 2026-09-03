#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/ReorderingStrategies/IdentityReordering.h>
#include <GPUSolver/ReorderingStrategies/RCMReordering.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/readMESH.h>
#include <igl/read_triangle_mesh.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

using gpuSolver::HostCSRMatrix;

// ------------------------------------------------------------
// Simple multithreaded CPU CSR SpMV.
// ------------------------------------------------------------

void cpuParallelSpMV(const HostCSRMatrix &A, const std::vector<float> &x,
                     std::vector<float> &y, std::size_t numberOfThreads) {
  const std::size_t numberOfRows = A.rows();

  const std::size_t rowsPerThread =
      (numberOfRows + numberOfThreads - 1) / numberOfThreads;

  const int *rowPtrA = A.activeRowPtr();

  const int *colPtrA = A.activeColPtr();

  const float *valPtrA = A.activeValPtr();

  std::vector<std::thread> threads;
  threads.reserve(numberOfThreads);

  for (std::size_t threadId = 0; threadId < numberOfThreads; ++threadId) {
    const std::size_t firstRow = threadId * rowsPerThread;

    const std::size_t lastRow =
        std::min(firstRow + rowsPerThread, numberOfRows);

    if (firstRow >= lastRow) {
      break;
    }

    threads.emplace_back(
        [&x, &y, rowPtrA, colPtrA, valPtrA, firstRow, lastRow]() {
          for (std::size_t row = firstRow; row < lastRow; ++row) {
            float sum = 0.0f;

            const int firstEntry = rowPtrA[row];

            const int lastEntry = rowPtrA[row + 1];

            for (int k = firstEntry; k < lastEntry; ++k) {
              sum += valPtrA[k] * x[colPtrA[k]];
            }

            y[row] = sum;
          }
        });
  }

  for (std::thread &thread : threads) {
    thread.join();
  }
}

// ------------------------------------------------------------

double elapsedMilliseconds(const Clock::time_point &start,
                           const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ------------------------------------------------------------
// Give the CPU/GPU a short idle period between benchmark phases.
// This reduces carry-over from the previous benchmark (temperature,
// power state, memory pressure), but does not replace repeated trials.
// ------------------------------------------------------------

void coolDown(std::chrono::milliseconds duration) {
  std::this_thread::sleep_for(duration);
}

// ------------------------------------------------------------

int main(int argc, char *argv[]) {
  constexpr int repetitions = 100;
  constexpr int warmupIterations = 5;
  constexpr std::chrono::milliseconds cooldownDuration(1000);

  // --------------------------------------------------------
  // 1. Build/load your tet mesh Laplacian here.
  // --------------------------------------------------------

  std::string path_to_source = std::string(PROJECT_SOURCE_DIR);
  Eigen::SparseMatrix<float, Eigen::RowMajor> A;

  //
  // YOUR EXISTING CODE HERE
  //
  // Example idea:
  //
  // Eigen::MatrixXd V;
  // Eigen::MatrixXi T;
  // load tet mesh...
  // build Laplacian...
  // A = ...
  //
  std::string mesh_path;

  // check if a mesh was passed as argument
  if (argc > 1) {
    mesh_path = std::string(argv[1]);
  } else {
    mesh_path = path_to_source + "/data/tetmeshes/box.mesh";
  }

  Eigen::MatrixXf V;
  Eigen::MatrixXi F, T;

  igl::read_triangle_mesh(mesh_path, V, F);

  std::cout << " rows of V are " << V.rows() << " x " << V.cols() << std::endl;
  std::cout << " rows of F are " << F.rows() << " x " << F.cols() << std::endl;
  Eigen::SparseMatrix<float> ACM;
  igl::cotmatrix(V, F, ACM);

  A = ACM.eval();
  A.makeCompressed();

  if (A.rows() == 0) {
    throw std::runtime_error("Benchmark matrix has not been initialized.");
  }
  // --------------------------------------------------------
  // 2. Convert once to CSR.
  // --------------------------------------------------------

  // HostCSRMatrix csr = makeCSR(A);
  const std::size_t nRows = static_cast<std::size_t>(A.rows());

  const std::size_t nCols = static_cast<std::size_t>(A.cols());

  const std::size_t nnz = static_cast<std::size_t>(A.nonZeros());

  int *rowPtrA = A.outerIndexPtr();

  int *colPtrA = A.innerIndexPtr();

  float *valPtrA = A.valuePtr();
  std::cout << "Matrix statistics\n"
            << "-----------------\n"
            << "Rows:       " << nRows << '\n'
            << "Cols:       " << nCols << '\n'
            << "NNZ:        " << nnz << '\n'
            << "NNZ / row:  "
            << static_cast<double>(nnz) / static_cast<double>(nRows) << "\n\n";

  // --------------------------------------------------------
  // Build permutation.
  //
  // --------------------------------------------------------

  int *oldToNew = new int[nRows];

  int *newToOld = new int[nRows];

  // gpuSolver::IdentityReordering identityReordering;
  gpuSolver::RCMReordering rcmReordering;

  // identityReordering.compute(nRows, rowPtrA, colPtrA, oldToNew, newToOld);
  rcmReordering.compute(nRows, rowPtrA, colPtrA, oldToNew, newToOld);

  gpuSolver::Permutation permutation(nRows, oldToNew, newToOld);

  delete[] oldToNew;
  delete[] newToOld;

  // --------------------------------------------------------
  // HostCSRMatrix copies both the original CSR data and the
  // permutation. Its active CSR representation is therefore
  // the permuted one.
  //
  // For the identity permutation, this must be equivalent to A.
  // --------------------------------------------------------

  gpuSolver::HostCSRMatrix csr(nRows, nCols, nnz, rowPtrA, colPtrA, valPtrA,
                               permutation);

  std::cout << "Matrix statistics\n"
            << "-----------------\n"
            << "Rows:       " << csr.rows() << '\n'
            << "Cols:       " << csr.cols() << '\n'
            << "NNZ:        " << csr.nnz() << '\n'
            << "NNZ / row:  "
            << static_cast<double>(csr.nnz()) / static_cast<double>(csr.rows())
            << "\n\n";

  // --------------------------------------------------------
  // 3. Construct deterministic input vector.
  // --------------------------------------------------------

  std::vector<float> x(csr.cols());

  for (std::size_t i = 0; i < x.size(); ++i) {
    const float index = static_cast<float>(i);

    x[i] = std::sin(0.001f * index) + 0.1f * std::cos(0.013f * index);
  }
  std::vector<float> xPermuted(nCols);

  const int *oldToNewPtr = permutation.oldToNew();

  for (std::size_t oldIndex = 0; oldIndex < nCols; ++oldIndex) {
    const int newIndex = oldToNewPtr[oldIndex];

    xPermuted[newIndex] = x[oldIndex];
  }
  // --------------------------------------------------------
  // 4. Eigen reference vectors.
  // --------------------------------------------------------

  Eigen::Map<const Eigen::VectorXf> eigenX(x.data(),
                                           static_cast<Eigen::Index>(x.size()));

  Eigen::VectorXf eigenY(A.rows());

  // --------------------------------------------------------
  // Warm up Eigen.
  // --------------------------------------------------------

  coolDown(cooldownDuration);

  for (int iteration = 0; iteration < warmupIterations; ++iteration) {
    eigenY.noalias() = A * eigenX;
  }

  // --------------------------------------------------------
  // Benchmark Eigen.
  // --------------------------------------------------------

  Clock::time_point eigenStart = Clock::now();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    eigenY.noalias() = A * eigenX;
  }

  Clock::time_point eigenEnd = Clock::now();

  const double eigenMilliseconds = elapsedMilliseconds(eigenStart, eigenEnd);

  // --------------------------------------------------------
  // The reordered matrix computes
  //
  //   y' = (P A P^T) (P x) = P (A x).
  //
  // Therefore CPU and Metal results are in the permuted ordering.
  // Build the matching reference y' = P y once.
  // --------------------------------------------------------

  std::vector<float> eigenYPermuted(nRows);

  for (std::size_t oldIndex = 0; oldIndex < nRows; ++oldIndex) {
    const int newIndex = oldToNewPtr[oldIndex];

    eigenYPermuted[newIndex] =
        eigenY[static_cast<Eigen::Index>(oldIndex)];
  }

  // --------------------------------------------------------
  // 5. CPU multithreaded implementation.
  //
  // cpuParallelSpMV should use:
  //     A.activeRowPtr()
  //     A.activeColPtr()
  //     A.activeValPtr()
  //
  // Thus it uses exactly the same reordered representation
  // that will later be uploaded to Metal.
  // --------------------------------------------------------

  std::vector<float> cpuY(csr.rows(), 0.0f);

  std::size_t numberOfThreads = std::thread::hardware_concurrency();

  if (numberOfThreads == 0) {
    numberOfThreads = 1;
  }

  std::cout << "CPU threads: " << numberOfThreads << "\n\n";

  coolDown(cooldownDuration);

  for (int iteration = 0; iteration < warmupIterations; ++iteration) {
    cpuParallelSpMV(csr, xPermuted, cpuY, numberOfThreads);
  }

  Clock::time_point cpuStart = Clock::now();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    cpuParallelSpMV(csr, xPermuted, cpuY, numberOfThreads);
  }

  Clock::time_point cpuEnd = Clock::now();

  const double cpuMilliseconds = elapsedMilliseconds(cpuStart, cpuEnd);

  // --------------------------------------------------------
  // 6. Metal setup.
  //
  // DeviceCSRMatrix only receives the active CSR representation.
  // It knows nothing about permutations.
  // --------------------------------------------------------

  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceCSRMatrix deviceA(context, csr);

  gpuSolver::DeviceVector deviceX(context, xPermuted);

  gpuSolver::DeviceVector deviceY(context, csr.rows());

  // --------------------------------------------------------
  // Warm up Metal.
  // --------------------------------------------------------

  coolDown(cooldownDuration);

  for (int iteration = 0; iteration < warmupIterations; ++iteration) {
    backend.spmv(deviceA, deviceX, deviceY);
  }

  // --------------------------------------------------------
  // Benchmark synchronized Metal SpMV.
  // --------------------------------------------------------

  Clock::time_point metalStart = Clock::now();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    backend.spmv(deviceA, deviceX, deviceY);
  }

  Clock::time_point metalEnd = Clock::now();

  const double metalMilliseconds = elapsedMilliseconds(metalStart, metalEnd);

  // --------------------------------------------------------
  // Benchmark repeated Metal SpMV with one synchronization.
  // --------------------------------------------------------

  coolDown(cooldownDuration);

  // Warm up the batched command-buffer path independently.
  backend.spmvRepeated(deviceA, deviceX, deviceY, warmupIterations);

  Clock::time_point metalBatchStart = Clock::now();

  backend.spmvRepeated(deviceA, deviceX, deviceY, repetitions);

  Clock::time_point metalBatchEnd = Clock::now();

  const double metalBatchMilliseconds =
      elapsedMilliseconds(metalBatchStart, metalBatchEnd);

  // --------------------------------------------------------
  // 7. Correctness check.
  // --------------------------------------------------------

  float *metalY = deviceY.download();

  float maxMetalError = 0.0f;
  float maxCpuError = 0.0f;

  for (std::size_t i = 0; i < csr.rows(); ++i) {
    const float reference = eigenYPermuted[i];

    const float metalError = std::abs(metalY[i] - reference);

    const float cpuError = std::abs(cpuY[i] - reference);

    if (metalError > maxMetalError) {
      maxMetalError = metalError;
    }

    if (cpuError > maxCpuError) {
      maxCpuError = cpuError;
    }
  }

  // --------------------------------------------------------
  // 8. Results.
  // --------------------------------------------------------

  std::cout << "Correctness\n"
            << "-----------\n"
            << "Max |Eigen - CPU MT|: " << maxCpuError << '\n'
            << "Max |Eigen - Metal|:  " << maxMetalError << "\n\n";

  std::cout << "Timing over " << repetitions << " SpMVs\n"
            << "---------------------------\n";

  std::cout << "Eigen total:              " << eigenMilliseconds << " ms\n";

  std::cout << "Eigen / SpMV:             " << eigenMilliseconds / repetitions
            << " ms\n\n";

  std::cout << "CPU MT total:             " << cpuMilliseconds << " ms\n";

  std::cout << "CPU MT / SpMV:            " << cpuMilliseconds / repetitions
            << " ms\n\n";

  std::cout << "Metal synchronized total: " << metalMilliseconds << " ms\n";

  std::cout << "Metal synchronized / SpMV:" << metalMilliseconds / repetitions
            << " ms\n\n";

  std::cout << "Metal batched total:      " << metalBatchMilliseconds
            << " ms\n";

  std::cout << "Metal batched / SpMV:     "
            << metalBatchMilliseconds / repetitions << " ms\n\n";

  std::cout << "Metal speedup vs Eigen:   "
            << eigenMilliseconds / metalMilliseconds << "x\n";

  std::cout << "Metal speedup vs CPU MT:  "
            << cpuMilliseconds / metalMilliseconds << "x\n";

  std::cout << "Metal batched speedup vs CPU MT: "
            << cpuMilliseconds / metalBatchMilliseconds << "x\n";
}


