#include <GPUSolver/DeviceELLMatrix.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostELLMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/ReorderingStrategies/RCMReordering.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/cotmatrix.h>
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

void cpuParallelSpMV(const HostCSRMatrix& A,
                     const std::vector<float>& x,
                     std::vector<float>& y,
                     std::size_t numberOfThreads)
{
    const std::size_t numberOfRows = A.rows();
    const std::size_t rowsPerThread =
        (numberOfRows + numberOfThreads - 1) / numberOfThreads;

    const int* rowPtrA = A.activeRowPtr();
    const int* colPtrA = A.activeColPtr();
    const float* valPtrA = A.activeValPtr();

    std::vector<std::thread> threads;
    threads.reserve(numberOfThreads);

    for (std::size_t threadId = 0; threadId < numberOfThreads; ++threadId)
    {
        const std::size_t firstRow = threadId * rowsPerThread;
        const std::size_t lastRow =
            std::min(firstRow + rowsPerThread, numberOfRows);

        if (firstRow >= lastRow)
        {
            break;
        }

        threads.emplace_back(
            [&x, &y, rowPtrA, colPtrA, valPtrA, firstRow, lastRow]()
            {
                for (std::size_t row = firstRow; row < lastRow; ++row)
                {
                    float sum = 0.0f;

                    const int firstEntry = rowPtrA[row];
                    const int lastEntry = rowPtrA[row + 1];

                    for (int k = firstEntry; k < lastEntry; ++k)
                    {
                        sum += valPtrA[k] * x[colPtrA[k]];
                    }

                    y[row] = sum;
                }
            });
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

double elapsedMilliseconds(const Clock::time_point& start,
                           const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void coolDown(std::chrono::milliseconds duration)
{
    std::this_thread::sleep_for(duration);
}

int main(int argc, char* argv[])
{
    constexpr int repetitions = 100;
    constexpr int warmupIterations = 5;
    constexpr std::size_t ellWidth = 7;
    constexpr std::chrono::milliseconds cooldownDuration(1000);

    const std::string pathToSource = std::string(PROJECT_SOURCE_DIR);

    std::string meshPath;

    if (argc > 1)
    {
        meshPath = std::string(argv[1]);
    }
    else
    {
        meshPath = pathToSource + "/data/tetmeshes/box.mesh";
    }

    Eigen::MatrixXf V;
    Eigen::MatrixXi F;

    igl::read_triangle_mesh(meshPath, V, F);

    std::cout << "Rows of V: " << V.rows() << " x " << V.cols() << '\n';
    std::cout << "Rows of F: " << F.rows() << " x " << F.cols() << "\n\n";

    Eigen::SparseMatrix<float> ACM;
    igl::cotmatrix(V, F, ACM);

    Eigen::SparseMatrix<float, Eigen::RowMajor> A;
    A = ACM.eval();
    A.makeCompressed();

    if (A.rows() == 0)
    {
        throw std::runtime_error(
            "Benchmark matrix has not been initialized.");
    }

    const std::size_t nRows = static_cast<std::size_t>(A.rows());
    const std::size_t nCols = static_cast<std::size_t>(A.cols());
    const std::size_t nnz = static_cast<std::size_t>(A.nonZeros());

    const int* rowPtrA = A.outerIndexPtr();
    const int* colPtrA = A.innerIndexPtr();
    const float* valPtrA = A.valuePtr();

    std::cout << "Matrix statistics\n"
              << "-----------------\n"
              << "Rows:       " << nRows << '\n'
              << "Cols:       " << nCols << '\n'
              << "NNZ:        " << nnz << '\n'
              << "NNZ / row:  "
              << static_cast<double>(nnz) / static_cast<double>(nRows)
              << "\n\n";

    int* oldToNew = new int[nRows];
    int* newToOld = new int[nRows];

    gpuSolver::RCMReordering rcmReordering;

    rcmReordering.compute(
        nRows,
        rowPtrA,
        colPtrA,
        oldToNew,
        newToOld);

    gpuSolver::Permutation permutation(
        nRows,
        oldToNew,
        newToOld);

    delete[] oldToNew;
    delete[] newToOld;

    gpuSolver::HostCSRMatrix csr(
        nRows,
        nCols,
        nnz,
        rowPtrA,
        colPtrA,
        valPtrA,
        permutation);

    gpuSolver::HostELLMatrix ell(
        csr,
        ellWidth);

    const std::size_t ellEntries =
        ell.rows() * ell.ellWidth();

    const double overflowPercentage =
        100.0
        * static_cast<double>(ell.overflowNnz())
        / static_cast<double>(ell.nnz());

    const double entryStorageRatio =
        static_cast<double>(ellEntries + ell.overflowNnz())
        / static_cast<double>(ell.nnz());

    std::cout << "ELL statistics\n"
              << "--------------\n"
              << "ELL width:           " << ell.ellWidth() << '\n'
              << "ELL entries:         " << ellEntries << '\n'
              << "Overflow NNZ:        " << ell.overflowNnz() << '\n'
              << "Overflow percentage: " << overflowPercentage << "%\n"
              << "Entry storage ratio: " << entryStorageRatio << "x\n\n";

    std::vector<float> x(nCols);

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const float index = static_cast<float>(i);

        x[i] =
            std::sin(0.001f * index)
            + 0.1f * std::cos(0.013f * index);
    }

    std::vector<float> xPermuted(nCols);
    const int* oldToNewPtr = permutation.oldToNew();

    for (std::size_t oldIndex = 0; oldIndex < nCols; ++oldIndex)
    {
        const int newIndex = oldToNewPtr[oldIndex];
        xPermuted[newIndex] = x[oldIndex];
    }

    Eigen::Map<const Eigen::VectorXf> eigenX(
        x.data(),
        static_cast<Eigen::Index>(x.size()));

    Eigen::VectorXf eigenY(A.rows());

    coolDown(cooldownDuration);

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        eigenY.noalias() = A * eigenX;
    }

    Clock::time_point eigenStart = Clock::now();

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration)
    {
        eigenY.noalias() = A * eigenX;
    }

    Clock::time_point eigenEnd = Clock::now();

    const double eigenMilliseconds =
        elapsedMilliseconds(eigenStart, eigenEnd);

    std::vector<float> eigenYPermuted(nRows);

    for (std::size_t oldIndex = 0; oldIndex < nRows; ++oldIndex)
    {
        const int newIndex = oldToNewPtr[oldIndex];

        eigenYPermuted[newIndex] =
            eigenY[static_cast<Eigen::Index>(oldIndex)];
    }

    std::vector<float> cpuY(nRows, 0.0f);

    std::size_t numberOfThreads =
        std::thread::hardware_concurrency();

    if (numberOfThreads == 0)
    {
        numberOfThreads = 1;
    }

    std::cout << "CPU threads: "
              << numberOfThreads
              << "\n\n";

    coolDown(cooldownDuration);

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        cpuParallelSpMV(
            csr,
            xPermuted,
            cpuY,
            numberOfThreads);
    }

    Clock::time_point cpuStart = Clock::now();

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration)
    {
        cpuParallelSpMV(
            csr,
            xPermuted,
            cpuY,
            numberOfThreads);
    }

    Clock::time_point cpuEnd = Clock::now();

    const double cpuMilliseconds =
        elapsedMilliseconds(cpuStart, cpuEnd);

    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceCSRMatrix deviceCSR(
        context,
        csr);

    gpuSolver::DeviceELLMatrix deviceELL(
        context,
        ell);

    gpuSolver::DeviceVector deviceX(
        context,
        xPermuted);

    gpuSolver::DeviceVector deviceYCSR(
        context,
        nRows);

    gpuSolver::DeviceVector deviceYELL(
        context,
        nRows);

    coolDown(cooldownDuration);

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        backend.spmv(
            deviceCSR,
            deviceX,
            deviceYCSR);
    }

    Clock::time_point csrStart = Clock::now();

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration)
    {
        backend.spmv(
            deviceCSR,
            deviceX,
            deviceYCSR);
    }

    Clock::time_point csrEnd = Clock::now();

    const double csrMilliseconds =
        elapsedMilliseconds(csrStart, csrEnd);

    coolDown(cooldownDuration);

    backend.spmvRepeated(
        deviceCSR,
        deviceX,
        deviceYCSR,
        warmupIterations);

    Clock::time_point csrBatchStart = Clock::now();

    backend.spmvRepeated(
        deviceCSR,
        deviceX,
        deviceYCSR,
        repetitions);

    Clock::time_point csrBatchEnd = Clock::now();

    const double csrBatchMilliseconds =
        elapsedMilliseconds(
            csrBatchStart,
            csrBatchEnd);

    coolDown(cooldownDuration);

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        backend.spmv(
            deviceELL,
            deviceX,
            deviceYELL);
    }

    Clock::time_point ellStart = Clock::now();

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration)
    {
        backend.spmv(
            deviceELL,
            deviceX,
            deviceYELL);
    }

    Clock::time_point ellEnd = Clock::now();

    const double ellMilliseconds =
        elapsedMilliseconds(ellStart, ellEnd);

    coolDown(cooldownDuration);

    backend.spmvRepeated(
        deviceELL,
        deviceX,
        deviceYELL,
        warmupIterations);

    Clock::time_point ellBatchStart = Clock::now();

    backend.spmvRepeated(
        deviceELL,
        deviceX,
        deviceYELL,
        repetitions);

    Clock::time_point ellBatchEnd = Clock::now();

    const double ellBatchMilliseconds =
        elapsedMilliseconds(
            ellBatchStart,
            ellBatchEnd);

    float* csrY = deviceYCSR.download();
    float* ellY = deviceYELL.download();

    float maxCpuError = 0.0f;
    float maxCSRError = 0.0f;
    float maxELLError = 0.0f;
    float maxCSRvsELL = 0.0f;

    for (std::size_t i = 0; i < nRows; ++i)
    {
        const float reference =
            eigenYPermuted[i];

        const float cpuError =
            std::abs(cpuY[i] - reference);

        const float csrError =
            std::abs(csrY[i] - reference);

        const float ellError =
            std::abs(ellY[i] - reference);

        const float csrVsELL =
            std::abs(csrY[i] - ellY[i]);

        maxCpuError =
            std::max(maxCpuError, cpuError);

        maxCSRError =
            std::max(maxCSRError, csrError);

        maxELLError =
            std::max(maxELLError, ellError);

        maxCSRvsELL =
            std::max(maxCSRvsELL, csrVsELL);
    }

    std::cout << "Correctness\n"
              << "-----------\n"
              << "Max |Eigen - CPU MT|: "
              << maxCpuError << '\n'
              << "Max |Eigen - CSR|:    "
              << maxCSRError << '\n'
              << "Max |Eigen - ELL|:    "
              << maxELLError << '\n'
              << "Max |CSR - ELL|:      "
              << maxCSRvsELL
              << "\n\n";

    std::cout << "Timing over "
              << repetitions
              << " SpMVs\n"
              << "---------------------------\n";

    std::cout << "Eigen / SpMV:             "
              << eigenMilliseconds / repetitions
              << " ms\n";

    std::cout << "CPU MT / SpMV:            "
              << cpuMilliseconds / repetitions
              << " ms\n\n";

    std::cout << "CSR synchronized / SpMV:  "
              << csrMilliseconds / repetitions
              << " ms\n";

    std::cout << "CSR batched / SpMV:       "
              << csrBatchMilliseconds / repetitions
              << " ms\n\n";

    std::cout << "ELL synchronized / SpMV:  "
              << ellMilliseconds / repetitions
              << " ms\n";

    std::cout << "ELL batched / SpMV:       "
              << ellBatchMilliseconds / repetitions
              << " ms\n\n";

    std::cout << "CSR synchronized speedup vs CPU MT: "
              << cpuMilliseconds / csrMilliseconds
              << "x\n";

    std::cout << "CSR batched speedup vs CPU MT:      "
              << cpuMilliseconds / csrBatchMilliseconds
              << "x\n";

    std::cout << "ELL synchronized speedup vs CPU MT: "
              << cpuMilliseconds / ellMilliseconds
              << "x\n";

    std::cout << "ELL batched speedup vs CPU MT:      "
              << cpuMilliseconds / ellBatchMilliseconds
              << "x\n\n";

    std::cout << "ELL speedup over CSR synchronized:   "
              << csrMilliseconds / ellMilliseconds
              << "x\n";

    std::cout << "ELL speedup over CSR batched:        "
              << csrBatchMilliseconds / ellBatchMilliseconds
              << "x\n";

    return 0;
}
