#include "GPUSolver/Backend.h"
#include "GPUSolver/BackendEncoder.h"
#include <GPUSolver/BlockUtils.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Smoothers/BlockJacobiSmoother.h>

namespace gpuSolver {

BlockJacobiSmoother::BlockJacobiSmoother(float omega) {
  this->omega_ = omega;
  this->inverseDiagonalBlocks_ = nullptr;
  this->Az_ = nullptr;
  this->residual_ = nullptr;
  this->correction_ = nullptr;
}

void BlockJacobiSmoother::initialize(Backend &backend,
                                     const HostCSRMatrix &hostMat,
                                     const DeviceCSRMatrix &devMat) {
  this->matrix_ = &devMat;

  const std::vector<float> inverseBlocks =
      extractInverseDiagonalBlocks3x3(hostMat);

  this->inverseDiagonalBlocks_ =
      backend.createVector(inverseBlocks.size(), inverseBlocks.data());

  this->Az_ = backend.createVector(hostMat.rows());

  this->residual_ = backend.createVector(hostMat.rows());

  this->correction_ = backend.createVector(hostMat.rows());
}


void BlockJacobiSmoother::smooth(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &rhs,
    DeviceVector &solution,
    std::size_t iterations)
{
  for (std::size_t iteration = 0;
       iteration < iterations;
       ++iteration) {

    // Az = A * solution
    backend.encodeSpmv(
        encoder,
        *matrix_,
        solution,
        *Az_
    );

    // residual = rhs
    backend.encodeCopy(
        encoder,
        rhs,
        *residual_
    );

    // residual -= Az
    backend.encodeAxpy(
        encoder,
        -1.0f,
        *Az_,
        *residual_
    );

    // correction =
    //     blockD^-1 * residual
    backend.encodeApplyBlockInverses3x3(
        encoder,
        *inverseDiagonalBlocks_,
        *residual_,
        *correction_
    );

    // solution += omega * correction
    backend.encodeAxpy(
        encoder,
        omega_,
        *correction_,
        solution
    );
  }
}

BlockJacobiSmoother::~BlockJacobiSmoother(){
  delete inverseDiagonalBlocks_;
  delete Az_;
  delete residual_;
  delete correction_;
}


} // namespace gpuSolver
