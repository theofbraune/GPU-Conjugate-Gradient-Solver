#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

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

// ------------------------------------------------------------
// Convert an Eigen sparse matrix to row-major CSR.
// ------------------------------------------------------------

struct HostCSRMatrix {
  std::size_t rows;
  std::size_t cols;

  std::vector<int> rowPtr;
  std::vector<int> colIdx;
  std::vector<float> values;
};

HostCSRMatrix makeCSR(const Eigen::SparseMatrix<float, Eigen::RowMajor> &A) {
  HostCSRMatrix csr;

  csr.rows = static_cast<std::size_t>(A.rows());
  csr.cols = static_cast<std::size_t>(A.cols());

  csr.rowPtr.resize(csr.rows + 1);
  csr.colIdx.resize(static_cast<std::size_t>(A.nonZeros()));
  csr.values.resize(static_cast<std::size_t>(A.nonZeros()));

  for (std::size_t i = 0; i < csr.rowPtr.size(); ++i) {
    csr.rowPtr[i] = A.outerIndexPtr()[i];
  }

  for (std::size_t i = 0; i < csr.colIdx.size(); ++i) {
    csr.colIdx[i] = A.innerIndexPtr()[i];
    csr.values[i] = A.valuePtr()[i];
  }

  return csr;
}

// ------------------------------------------------------------
// Simple multithreaded CPU CSR SpMV.
// ------------------------------------------------------------

void cpuParallelSpMV(const HostCSRMatrix &A, const std::vector<float> &x,
                     std::vector<float> &y, std::size_t numberOfThreads) {
  const std::size_t numberOfRows = A.rows;

  const std::size_t rowsPerThread =
      (numberOfRows + numberOfThreads - 1) / numberOfThreads;

  std::vector<std::thread> threads;
  threads.reserve(numberOfThreads);

  for (std::size_t threadId = 0; threadId < numberOfThreads; ++threadId) {
    const std::size_t firstRow = threadId * rowsPerThread;

    const std::size_t lastRow =
        std::min(firstRow + rowsPerThread, numberOfRows);

    if (firstRow >= lastRow) {
      break;
    }

    threads.emplace_back([&A, &x, &y, firstRow, lastRow]() {
      for (std::size_t row = firstRow; row < lastRow; ++row) {
        float sum = 0.0f;

        const int firstEntry = A.rowPtr[row];

        const int lastEntry = A.rowPtr[row + 1];

        for (int k = firstEntry; k < lastEntry; ++k) {
          sum += A.values[k] * x[A.colIdx[k]];
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

int main(int argc, char *argv[]) {
  constexpr int repetitions = 100;
  constexpr int warmupIterations = 5;

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

  if (A.rows() == 0) {
    throw std::runtime_error("Benchmark matrix has not been initialized.");
  }
  // --------------------------------------------------------
  // 2. Convert once to CSR.
  // --------------------------------------------------------

  HostCSRMatrix csr = makeCSR(A);

  std::cout << "Matrix statistics\n"
            << "-----------------\n"
            << "Rows:       " << csr.rows << '\n'
            << "Cols:       " << csr.cols << '\n'
            << "NNZ:        " << csr.values.size() << '\n'
            << "NNZ / row:  "
            << static_cast<double>(csr.values.size()) /
                   static_cast<double>(csr.rows)
            << "\n\n";

  // --------------------------------------------------------
  // 3. Construct deterministic input vector.
  // --------------------------------------------------------

  std::vector<float> x(csr.cols);

  for (std::size_t i = 0; i < x.size(); ++i) {
    const float index = static_cast<float>(i);

    x[i] = std::sin(0.001f * index) + 0.1f * std::cos(0.013f * index);
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
  // 5. CPU multithreaded implementation.
  // --------------------------------------------------------

  std::vector<float> cpuY(csr.rows, 0.0f);

  std::size_t numberOfThreads = std::thread::hardware_concurrency();

  if (numberOfThreads == 0) {
    numberOfThreads = 1;
  }

  std::cout << "CPU threads: " << numberOfThreads << "\n\n";

  for (int iteration = 0; iteration < warmupIterations; ++iteration) {
    cpuParallelSpMV(csr, x, cpuY, numberOfThreads);
  }

  Clock::time_point cpuStart = Clock::now();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    cpuParallelSpMV(csr, x, cpuY, numberOfThreads);
  }

  Clock::time_point cpuEnd = Clock::now();

  const double cpuMilliseconds = elapsedMilliseconds(cpuStart, cpuEnd);

  // --------------------------------------------------------
  // 6. Metal setup.
  //    IMPORTANT: all allocation/upload happens BEFORE timing.
  // --------------------------------------------------------

  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceCSRMatrix deviceA(context, csr.rows, csr.cols,
                                     csr.values.size(), csr.rowPtr.data(),
                                     csr.colIdx.data(), csr.values.data());

  gpuSolver::DeviceVector deviceX(context, x);

  gpuSolver::DeviceVector deviceY(context, csr.rows);

  // --------------------------------------------------------
  // Warm up Metal.
  // --------------------------------------------------------

  for (int iteration = 0; iteration < warmupIterations; ++iteration) {
    backend.spmv(deviceA, deviceX, deviceY);
  }

  // --------------------------------------------------------
  // Benchmark Metal.
  // --------------------------------------------------------

  Clock::time_point metalStart = Clock::now();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    backend.spmv(deviceA, deviceX, deviceY);
  }

  Clock::time_point metalEnd = Clock::now();

  const double metalMilliseconds = elapsedMilliseconds(metalStart, metalEnd);

  Clock::time_point metalSyncStart = Clock::now();

  backend.spmvRepeated(deviceA, deviceX, deviceY, repetitions);
  Clock::time_point metalSyncEnd = Clock::now();

  const double metalSyncMilliseconds =
      elapsedMilliseconds(metalSyncStart, metalSyncEnd);

  // --------------------------------------------------------
  // 7. Correctness check.
  //    Download only AFTER timing.
  // --------------------------------------------------------

  float *metalY = deviceY.download();

  float maxMetalError = 0.0f;
  float maxCpuError = 0.0f;

  for (std::size_t i = 0; i < csr.rows; ++i) {
    const float reference = eigenY[static_cast<Eigen::Index>(i)];

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

  std::cout << "Eigen total:       " << eigenMilliseconds << " ms\n";

  std::cout << "Eigen / SpMV:      " << eigenMilliseconds / repetitions
            << " ms\n\n";

  std::cout << "CPU MT total:      " << cpuMilliseconds << " ms\n";

  std::cout << "CPU MT / SpMV:     " << cpuMilliseconds / repetitions
            << " ms\n\n";

  std::cout << "Metal total:       " << metalMilliseconds << " ms\n";

  std::cout << "Metal / SpMV:      " << metalMilliseconds / repetitions
            << " ms\n\n";


  std::cout << "Metal total without sync:       " << metalSyncMilliseconds << " ms\n";

  std::cout << "MetalSync / SpMV:      " << metalSyncMilliseconds/ repetitions
            << " ms\n\n";

  std::cout << "Metal speedup vs Eigen:   "
            << eigenMilliseconds / metalMilliseconds << "x\n";

  std::cout << "Metal speedup vs CPU MT:  "
            << cpuMilliseconds / metalMilliseconds << "x\n";

  std::cout << "Metal without sync speedup vs CPU MT:  "
            << cpuMilliseconds / metalSyncMilliseconds << "x\n";

  return 0;
}
