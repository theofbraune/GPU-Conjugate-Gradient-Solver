#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Preconditioners/DampedJacobiPreconditioner.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace gpuSolver {

DampedJacobiPreconditioner::DampedJacobiPreconditioner(
    const std::size_t nSmoother, const float weight) {
  if (weight <= 0.0f) {
    throw std::runtime_error(
        "DampedJacobiPreconditioner: omega must be positive.");
  }

  if (nSmoother == 0) {
    throw std::runtime_error(
        "DampedJacobiPreconditioner: iterations must be greater than zero.");
  }
  this->omega_ = weight;
  this->smoothingSteps_ = nSmoother;

  this->Az_ = nullptr;
  this->correction_ = nullptr;
  this->inverseDiagonal_ = nullptr;

  this->matrix_ = nullptr;
}

void DampedJacobiPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {

  if (inverseDiagonal_ != nullptr || Az_ != nullptr || correction_ != nullptr) {
    throw std::runtime_error(
        "DampedJacobiPreconditioner::initialize: already initialized.");
  }
  this->matrix_ = &deviceMatrix;

  const std::size_t n = hostMatrix.rows();

  std::vector<float> inverseDiagonal(n);

  const int *rowPtr = hostMatrix.activeRowPtr();

  const int *colPtr = hostMatrix.activeColPtr();

  const float *values = hostMatrix.activeValPtr();

  for (std::size_t row = 0; row < n; ++row) {
    bool diagonalFound = false;

    for (std::size_t entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
      if (colPtr[entry] == row) {
        const float diagonalValue = values[entry];

        if (std::abs(diagonalValue) < 1e-12f) {
          throw std::runtime_error("DampedJacobiPreconditioner::initialize: "
                                   "zero diagonal entry.");
        }

        inverseDiagonal[row] = 1.0f / diagonalValue;

        diagonalFound = true;
        break;
      }
    }

    if (!diagonalFound) {
      throw std::runtime_error("DampedJacobiPreconditioner::initialize: "
                               "missing diagonal entry.");
    }
  }

  inverseDiagonal_ = backend.createVector(n, inverseDiagonal.data());

  this->Az_ = backend.createVector(n);
  this->correction_ = backend.createVector(n);
}

DampedJacobiPreconditioner::~DampedJacobiPreconditioner() {

  delete inverseDiagonal_;
  delete Az_;
  delete correction_;
}

void DampedJacobiPreconditioner::smooth(Backend &backend,
                                        BackendEncoder &encoder,
                                        const DeviceVector &residual,
                                        DeviceVector &z, std::size_t iters) {

  for (std::size_t i = 0; i < iters; i++) {

    backend.encodeSpmv(encoder, *matrix_, z, *Az_);

    backend.encodeCopy(encoder, residual, *this->correction_);

    backend.encodeAxpy(encoder, -1.0f, *Az_, *correction_);

    backend.encodeScaleVector(encoder, *this->inverseDiagonal_, *correction_);

    backend.encodeAxpy(encoder, this->omega_, *correction_, z);
  }
}

void DampedJacobiPreconditioner::apply(Backend &backend,
                                       BackendEncoder &encoder,
                                       const DeviceVector &residual,
                                       DeviceVector &z) {

  backend.encodeSetZero(encoder, z);
  this->smooth(backend, encoder, residual, z, this->smoothingSteps_);
}
} // namespace gpuSolver
