#include <GPUSolver/Backend.h>
#include <GPUSolver/Smoothers/DampedJacobiSmoother.h>

namespace gpuSolver {

DampedJacobiSmoother::DampedJacobiSmoother(float omega) {

  this->omega_ = omega;
  inverseDiagonal_ = nullptr;
  Az_ = nullptr;
  correction_ = nullptr;
}

void DampedJacobiSmoother::initialize(Backend &backend,
                                      const HostCSRMatrix &hostMatrix,
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

DampedJacobiSmoother::~DampedJacobiSmoother() {

  delete inverseDiagonal_;
  delete Az_;
  delete correction_;
}

void DampedJacobiSmoother::smooth(Backend &backend, BackendEncoder &encoder, const DeviceVector &rhs,
            DeviceVector &solution, std::size_t iterations){

  for (std::size_t i = 0; i < iterations; i++) {

    backend.encodeSpmv(encoder, *matrix_, solution, *Az_);

    backend.encodeCopy(encoder,rhs, *this->correction_);

    backend.encodeAxpy(encoder, -1.0f, *Az_, *correction_);

    backend.encodeScaleVector(encoder, *this->inverseDiagonal_, *correction_);

    backend.encodeAxpy(encoder, this->omega_, *correction_, solution);
  }


}

} // namespace gpuSolver
