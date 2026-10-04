#include <GPUSolver/Backend.h>
#include <GPUSolver/GraphColoring.h>
#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>
#include <GPUSolver/Smoothers/SymmetricGaussSeidelSmoother.h>
#include <GPUSolver/DeviceIndexVector.h>
#include <cstddef>

namespace gpuSolver {

void SymmetricGaussSeidelPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {
  this->smoother = new SymmetricGaussSeidelSmoother(this->omega_);

  this->smoother->initialize(backend, hostMatrix, deviceMatrix);
}

SymmetricGaussSeidelPreconditioner::SymmetricGaussSeidelPreconditioner(
    float omega, std::size_t nbOfIters)
{
  this->nbOfReps = nbOfIters;
  this->omega_ = omega;

  if (omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error("SymmetricGaussSeidelPreconditioner: "
                             "omega must be between 0 and 2.");
  }
}

void SymmetricGaussSeidelPreconditioner::apply(Backend &backend,
                                               BackendEncoder &encoder,
                                               const DeviceVector &input,
                                               DeviceVector &output) {

  backend.encodeSetZero(encoder, output);

  this->smoother->smooth(backend, encoder, input, output, 1);
}

SymmetricGaussSeidelPreconditioner::~SymmetricGaussSeidelPreconditioner() {
  delete smoother;
}
} // namespace gpuSolver
