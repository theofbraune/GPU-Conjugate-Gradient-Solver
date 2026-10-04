#include <GPUSolver/Backend.h>
#include <GPUSolver/Preconditioners/MultigridGaussSeidelPreconditioner.h>

namespace gpuSolver {

MultigridGaussSeidelPreconditioner::
MultigridGaussSeidelPreconditioner(
    Backend &backend,
    const AMGHierarchy &hierarchy,
    const std::vector<Permutation> &permutations,
    std::size_t preSmoothingSteps,
    std::size_t postSmoothingSteps,
    std::size_t numberOfVCycles,
    float omega)
    : smoother_(
          backend,
          hierarchy,
          permutations,
          preSmoothingSteps,
          postSmoothingSteps,
          preSmoothingSteps + postSmoothingSteps,
          omega),
      numberOfVCycles_(numberOfVCycles)
{
}

void MultigridGaussSeidelPreconditioner::initialize(
    Backend &backend,
    const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix)
{
  smoother_.initialize(
      backend,
      hostMatrix,
      deviceMatrix
  );
}

void MultigridGaussSeidelPreconditioner::apply(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &residual,
    DeviceVector &z)
{
  backend.encodeSetZero(encoder, z);

  smoother_.smooth(
      backend,
      encoder,
      residual,
      z,
      numberOfVCycles_
  );
}

} // namespace gpuSolver
