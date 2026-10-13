#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
  try {

    using Matrix =
        Eigen::SparseMatrix<float, Eigen::RowMajor>;

    std::vector<Eigen::Triplet<float>> triplets;

    triplets.emplace_back(0, 0, 4.0f);
    triplets.emplace_back(0, 1, 1.0f);

    triplets.emplace_back(1, 0, 1.0f);
    triplets.emplace_back(1, 1, 3.0f);
    triplets.emplace_back(1, 2, 2.0f);

    triplets.emplace_back(2, 1, 2.0f);
    triplets.emplace_back(2, 2, 5.0f);

    Matrix A(3, 3);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end());

    A.makeCompressed();

    Eigen::VectorXf x(3);
    x << 1.0f, 2.0f, 3.0f;

    Eigen::VectorXf yReference =
        A * x;

    gpuSolver::HostCSRMatrix hostA(
        static_cast<std::size_t>(A.rows()),
        static_cast<std::size_t>(A.cols()),
        static_cast<std::size_t>(A.nonZeros()),
        A.outerIndexPtr(),
        A.innerIndexPtr(),
        A.valuePtr());

    gpuSolver::CudaBackend backend;

    gpuSolver::DeviceCSRMatrix *deviceA =
        backend.createCSRMatrix(hostA);

    gpuSolver::DeviceVector *deviceX =
        backend.createVector(
            static_cast<std::size_t>(x.size()),
            x.data());

    gpuSolver::DeviceVector *deviceY =
        backend.createVector(
            static_cast<std::size_t>(x.size()));

    gpuSolver::BackendEncoder *encoder =
        backend.createEncoder();

    backend.encodeSpmv(
        *encoder,
        *deviceA,
        *deviceX,
        *deviceY);

    backend.submitAndWait(
        *encoder);

    float *yData =
        deviceY->download();

    for (int i = 0; i < 3; ++i) {

      std::cout
          << "y[" << i << "] = "
          << yData[i]
          << ", reference = "
          << yReference[i]
          << "\n";

      if (std::abs(
              yData[i]
              - yReference[i])
          > 1.0e-5f) {

        throw std::runtime_error(
            "CUDA SpMV result incorrect.");
      }
    }

    std::cout
        << "CUDA CSR SpMV test: PASS\n";

    delete encoder;
    delete deviceY;
    delete deviceX;
    delete deviceA;

    return 0;

  } catch (const std::exception &exception) {

    std::cerr
        << "CUDA CSR SpMV test: FAIL\n"
        << exception.what()
        << "\n";

    return 1;
  }
}
