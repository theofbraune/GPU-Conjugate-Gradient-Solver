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
}

void DampedJacobiPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {

  this->smoother = new DampedJacobiSmoother(this->omega_);

  this->smoother->initialize(backend, hostMatrix, deviceMatrix);

}

DampedJacobiPreconditioner::~DampedJacobiPreconditioner() {
  delete smoother;
}


void DampedJacobiPreconditioner::apply(Backend &backend,
                                       BackendEncoder &encoder,
                                       const DeviceVector &residual,
                                       DeviceVector &z) {

  backend.encodeSetZero(encoder, z);
  this->smoother->smooth(backend, encoder, residual, z, this->smoothingSteps_);
}


} // namespace gpuSolver
