#include <GPUSolver/Preconditioners/MultigridBlockJacobiPreconditioner.h>
#include <GPUSolver/Smoothers/MultigridSmoother.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>

namespace gpuSolver {

MultigridBlockJacobiPreconditioner::
    MultigridBlockJacobiPreconditioner(
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
          MultigridSmoother::SmootherType::BlockJacobi,
          preSmoothingSteps,
          postSmoothingSteps,
          coarseSmoothingSteps,
          omega),
      numberOfVCycles_(numberOfVCycles)
{
}

void MultigridBlockJacobiPreconditioner::initialize(
    Backend & /*backend*/,
    const HostCSRMatrix & /*hostMatrix*/,
    const DeviceCSRMatrix & /*deviceMatrix*/)
{
  // Nothing to do.
  //
  // MultigridSmoother is already completely initialized
  // in its constructor from the AMG hierarchy.
}

void MultigridBlockJacobiPreconditioner::apply(
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
