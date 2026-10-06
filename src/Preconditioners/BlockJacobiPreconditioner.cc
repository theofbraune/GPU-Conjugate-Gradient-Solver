#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Preconditioners/BlockJacobiPreconditioner.h>

#include <stdexcept>

namespace gpuSolver {

BlockJacobiPreconditioner::BlockJacobiPreconditioner(
    const std::size_t nSmoother, const float weight) {
  if (weight <= 0.0f) {
    throw std::runtime_error(
        "BlockJacobiPreconditioner: omega must be positive.");
  }

  if (nSmoother == 0) {
    throw std::runtime_error(
        "BlockJacobiPreconditioner: iterations must be greater than zero.");
  }

  this->omega_ = weight;
  this->smoothingSteps_ = nSmoother;
}

void BlockJacobiPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {

  this->smoother = new BlockJacobiSmoother(this->omega_);

  this->smoother->initialize(backend, hostMatrix, deviceMatrix);
}

BlockJacobiPreconditioner::~BlockJacobiPreconditioner() {
  delete smoother;
}

void BlockJacobiPreconditioner::apply(Backend &backend,
                                      BackendEncoder &encoder,
                                      const DeviceVector &residual,
                                      DeviceVector &z) {

  backend.encodeSetZero(encoder, z);
  this->smoother->smooth(backend, encoder, residual, z, this->smoothingSteps_);
}

} // namespace gpuSolver
