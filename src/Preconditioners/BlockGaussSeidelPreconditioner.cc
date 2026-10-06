#include <GPUSolver/Backend.h>
#include <GPUSolver/Preconditioners/BlockGaussSeidelPreconditioner.h>

#include <stdexcept>

namespace gpuSolver {

BlockGaussSeidelPreconditioner::BlockGaussSeidelPreconditioner(
    float omega, std::size_t numberOfSweeps)
    : omega_(omega), numberOfSweeps_(numberOfSweeps)
{
  if (omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error(
        "BlockGaussSeidelPreconditioner: omega must be between 0 and 2.");
  }

  if (numberOfSweeps_ == 0) {
    throw std::runtime_error(
        "BlockGaussSeidelPreconditioner: numberOfSweeps must be greater than zero.");
  }
}

void BlockGaussSeidelPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {

  this->smoother = new BlockGaussSeidelSmoother(this->omega_);

  this->smoother->initialize(backend, hostMatrix, deviceMatrix);
}

BlockGaussSeidelPreconditioner::~BlockGaussSeidelPreconditioner() {
  delete smoother;
}

void BlockGaussSeidelPreconditioner::apply(
    Backend &backend, BackendEncoder &encoder,
    const DeviceVector &input, DeviceVector &output) {

  backend.encodeSetZero(encoder, output);
  this->smoother->smooth(backend, encoder, input, output, this->numberOfSweeps_);
}

} // namespace gpuSolver
