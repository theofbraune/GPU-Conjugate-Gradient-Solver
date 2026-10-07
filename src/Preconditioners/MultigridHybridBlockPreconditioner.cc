#include <GPUSolver/Preconditioners/MultigridHybridBlockPreconditioner.h> 

#include <GPUSolver/Smoothers/MultigridSmoother.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>

namespace gpuSolver {

MultigridHybridBlockPreconditioner::MultigridHybridBlockPreconditioner(
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
          MultigridSmoother::SmootherType::BlockJacobi,
          preSmoothingSteps,
          postSmoothingSteps,
          coarseSmoothingSteps,
          omega),
      numberOfVCycles_(numberOfVCycles)
{
}

void MultigridHybridBlockPreconditioner::initialize(
    Backend & /*backend*/,
    const HostCSRMatrix & /*hostMatrix*/,
    const DeviceCSRMatrix & /*deviceMatrix*/)
{
  // Nothing to do.
  //
  // MultigridSmoother is already initialized
  // in its constructor from the AMG hierarchy.
}

void MultigridHybridBlockPreconditioner::apply(
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
