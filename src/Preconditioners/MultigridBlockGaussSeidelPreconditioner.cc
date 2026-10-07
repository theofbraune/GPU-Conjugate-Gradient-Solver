#include <GPUSolver/Preconditioners/MultigridBlockGaussSeidelPreconditioner.h>
#include <GPUSolver/Smoothers/MultigridSmoother.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>

namespace gpuSolver {

MultigridBlockGaussSeidelPreconditioner::
    MultigridBlockGaussSeidelPreconditioner(
        Backend &backend,
        const AMGHierarchy &hierarchy,
        const std::vector<Permutation> &permutations,
        std::size_t preSmoothingSteps,
        std::size_t postSmoothingSteps,
        std::size_t coarseSmoothingSteps,
        float omega,
        std::size_t numberOfVCycles)
    : smoother_(
          backend,
          hierarchy,
          permutations,
          MultigridSmoother::SmootherType::BlockGaussSeidel,
          preSmoothingSteps,
          postSmoothingSteps,
          coarseSmoothingSteps,
          omega),
      numberOfVCycles_(numberOfVCycles)
{
}

void MultigridBlockGaussSeidelPreconditioner::initialize(
    Backend & /*backend*/,
    const HostCSRMatrix & /*hostMatrix*/,
    const DeviceCSRMatrix & /*deviceMatrix*/)
{
  // Nothing to do.
  //
  // MultigridSmoother is already initialized
  // in its constructor from the AMG hierarchy.
}

void MultigridBlockGaussSeidelPreconditioner::apply(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &residual,
    DeviceVector &z)
{
  backend.encodeSetZero(
      encoder,
      z
  );

  smoother_.smooth(
      backend,
      encoder,
      residual,
      z,
      numberOfVCycles_
  );
}

} // namespace gpuSolver
