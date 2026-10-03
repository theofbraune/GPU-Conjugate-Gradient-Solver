#include <GPUSolver/Preconditioners/MultigridGaussSeidelPreconditioner.h>

#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/SparseMatrixUtils.h>

#include <Eigen/SparseCore>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gpuSolver {

MultigridGaussSeidelPreconditioner::MultigridGaussSeidelPreconditioner(
    Backend &backend, const AMGHierarchy &hierarchy,
    const std::vector<Permutation> &permutations, std::size_t preSmoothingSteps,
    std::size_t postSmoothingSteps, std::size_t nbOfReps, float omega)
    : preSmoothingSteps_(preSmoothingSteps),
      postSmoothingSteps_(postSmoothingSteps), nbOfReps_(nbOfReps),
      omega_(omega) {
  const std::size_t numberOfLevels = hierarchy.A.size();

  if (numberOfLevels == 0) {
    throw std::runtime_error(
        "MultigridGaussSeidelPreconditioner: empty hierarchy.");
  }

  if (hierarchy.P.size() + 1 != numberOfLevels ||
      hierarchy.R.size() + 1 != numberOfLevels) {
    throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                             "invalid number of transfer matrices.");
  }

  if (permutations.size() != numberOfLevels) {
    throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                             "expected one permutation per level.");
  }

  if (nbOfReps_ == 0) {
    throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                             "number of V-cycles must be positive.");
  }

  if (preSmoothingSteps_ != postSmoothingSteps_) {
    throw std::runtime_error(
        "MultigridGaussSeidelPreconditioner: "
        "use equal pre- and post-smoothing steps for PCG.");
  }

  if (!std::isfinite(omega_) || omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                             "omega must be in (0, 2).");
  }

  levels_.resize(numberOfLevels);
  transfers_.resize(numberOfLevels - 1);

  // --------------------------------------------------------
  // Construct the device representation of every level.
  // --------------------------------------------------------

  for (std::size_t k = 0; k < numberOfLevels; ++k) {

    const int *oldToNewK = permutations[k].oldToNew();

    // ------------------------------------------------------
    // Permute Galerkin matrix.
    //
    // A'_k = Pi_k A_k Pi_k^T
    // ------------------------------------------------------

    AMGHierarchy::Matrix Akp =
        permuteMatrix(hierarchy.A[k], oldToNewK, oldToNewK);

    Akp.makeCompressed();

    const std::size_t nRows = static_cast<std::size_t>(Akp.rows());

    const std::size_t nCols = static_cast<std::size_t>(Akp.cols());

    const std::size_t nnz = static_cast<std::size_t>(Akp.nonZeros());

    if (nRows != nCols) {
      throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                               "Galerkin matrix must be square.");
    }

    HostCSRMatrix hostAkp(nRows, nCols, nnz, Akp.outerIndexPtr(),
                          Akp.innerIndexPtr(), Akp.valuePtr());

    // ------------------------------------------------------
    // Upload the matrix.
    // ------------------------------------------------------

    DeviceCSRMatrix *deviceAkp = backend.createCSRMatrix(hostAkp);

    levels_[k].matrix = deviceAkp;

    // ------------------------------------------------------
    // Construct the symmetric Gauss-Seidel smoother.
    //
    // initialize() will:
    //
    //   1. color the permuted Galerkin matrix;
    //   2. upload colorVertices;
    //   3. store the color offsets;
    //   4. borrow the device matrix.
    // ------------------------------------------------------

    SymmetricGaussSeidelPreconditioner *smoother =
        new SymmetricGaussSeidelPreconditioner(omega_);

    levels_[k].smoother = smoother;

    smoother->initialize(backend, hostAkp, *deviceAkp);

    // ------------------------------------------------------
    // Allocate V-cycle scratch vectors.
    // ------------------------------------------------------

    levels_[k].rhs = backend.createVector(nRows);

    levels_[k].residual = backend.createVector(nRows);

    levels_[k].correction = backend.createVector(nRows);

    levels_[k].Az = backend.createVector(nRows);

    // ------------------------------------------------------
    // Upload transfers between levels k-1 and k.
    // ------------------------------------------------------

    if (k > 0) {

      const int *oldToNewKM1 = permutations[k - 1].oldToNew();

      // P'_(k-1) = Pi_(k-1) P_(k-1) Pi_k^T

      AMGHierarchy::Matrix Pkp =
          permuteMatrix(hierarchy.P[k - 1], oldToNewKM1, oldToNewK);

      // R'_(k-1) = Pi_k R_(k-1) Pi_(k-1)^T

      AMGHierarchy::Matrix Rkp =
          permuteMatrix(hierarchy.R[k - 1], oldToNewK, oldToNewKM1);

      Pkp.makeCompressed();
      Rkp.makeCompressed();

      if (Pkp.rows() != hierarchy.A[k - 1].rows() ||
          Pkp.cols() != hierarchy.A[k].rows() ||
          Rkp.rows() != hierarchy.A[k].rows() ||
          Rkp.cols() != hierarchy.A[k - 1].rows()) {
        throw std::runtime_error("MultigridGaussSeidelPreconditioner: "
                                 "invalid transfer dimensions.");
      }

      HostCSRMatrix hostPkp(static_cast<std::size_t>(Pkp.rows()),
                            static_cast<std::size_t>(Pkp.cols()),
                            static_cast<std::size_t>(Pkp.nonZeros()),
                            Pkp.outerIndexPtr(), Pkp.innerIndexPtr(),
                            Pkp.valuePtr());

      HostCSRMatrix hostRkp(static_cast<std::size_t>(Rkp.rows()),
                            static_cast<std::size_t>(Rkp.cols()),
                            static_cast<std::size_t>(Rkp.nonZeros()),
                            Rkp.outerIndexPtr(), Rkp.innerIndexPtr(),
                            Rkp.valuePtr());

      transfers_[k - 1].prolongation = backend.createCSRMatrix(hostPkp);

      transfers_[k - 1].restriction = backend.createCSRMatrix(hostRkp);
    }
  }
}

MultigridGaussSeidelPreconditioner::~MultigridGaussSeidelPreconditioner() {
  for (Level &level : levels_) {

    // Destroy the smoother before its borrowed matrix.
    delete level.smoother;
    delete level.matrix;

    delete level.rhs;
    delete level.residual;
    delete level.correction;
    delete level.Az;
  }

  for (Transfer &transfer : transfers_) {
    delete transfer.restriction;
    delete transfer.prolongation;
  }
}

void MultigridGaussSeidelPreconditioner::initialize(
    Backend & /*backend*/, const HostCSRMatrix & /*hostMatrix*/,
    const DeviceCSRMatrix & /*deviceMatrix*/) {
  // The full hierarchy is already constructed in the
  // constructor.
}

void MultigridGaussSeidelPreconditioner::vCycle(
    Backend &backend,
    BackendEncoder &encoder,
    std::size_t level,
    const DeviceVector &rhs,
    DeviceVector &solution)
{
  Level &currentLevel = levels_[level];

  const bool isCoarsest =
      (level + 1 == levels_.size());

  // --------------------------------------------------------
  // Coarsest level.
  //
  // For now, use symmetric Gauss-Seidel as an approximate
  // coarse solver.
  // --------------------------------------------------------

  if (isCoarsest) {

    currentLevel.smoother->smooth(
        backend,
        encoder,
        rhs,
        solution,
        preSmoothingSteps_ + postSmoothingSteps_
    );

    return;
  }

  // --------------------------------------------------------
  // Pre-smoothing.
  // --------------------------------------------------------

  currentLevel.smoother->smooth(
      backend,
      encoder,
      rhs,
      solution,
      preSmoothingSteps_
  );

  // --------------------------------------------------------
  // Compute residual:
  //
  // r_k = rhs - A_k solution
  // --------------------------------------------------------

  backend.encodeSpmv(
      encoder,
      *currentLevel.matrix,
      solution,
      *currentLevel.Az
  );

  backend.encodeCopy(
      encoder,
      rhs,
      *currentLevel.residual
  );

  backend.encodeAxpy(
      encoder,
      -1.0f,
      *currentLevel.Az,
      *currentLevel.residual
  );

  // --------------------------------------------------------
  // Restrict residual to coarse level.
  // --------------------------------------------------------

  Transfer &transfer =
      transfers_[level];

  Level &coarseLevel =
      levels_[level + 1];

  backend.encodeSpmv(
      encoder,
      *transfer.restriction,
      *currentLevel.residual,
      *coarseLevel.rhs
  );

  // --------------------------------------------------------
  // Solve coarse error equation.
  //
  // A_(k+1) e_(k+1) = R_k r_k
  // --------------------------------------------------------

  backend.encodeSetZero(
      encoder,
      *coarseLevel.correction
  );

  vCycle(
      backend,
      encoder,
      level + 1,
      *coarseLevel.rhs,
      *coarseLevel.correction
  );

  // --------------------------------------------------------
  // Prolongate coarse correction.
  // --------------------------------------------------------

  backend.encodeSpmv(
      encoder,
      *transfer.prolongation,
      *coarseLevel.correction,
      *currentLevel.correction
  );

  // solution += P_k e_(k+1)

  backend.encodeAxpy(
      encoder,
      1.0f,
      *currentLevel.correction,
      solution
  );

  // --------------------------------------------------------
  // Post-smoothing.
  // --------------------------------------------------------

  currentLevel.smoother->smooth(
      backend,
      encoder,
      rhs,
      solution,
      postSmoothingSteps_
  );
}
void MultigridGaussSeidelPreconditioner::applyVCycles(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &rhs,
    DeviceVector &solution,
    std::size_t numberOfCycles)
{
  for (std::size_t cycle = 0;
       cycle < numberOfCycles;
       ++cycle) {

    vCycle(
        backend,
        encoder,
        0,
        rhs,
        solution
    );
  }
}

void MultigridGaussSeidelPreconditioner::apply(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &residual,
    DeviceVector &z)
{
  // Start the approximate solve from zero.
  backend.encodeSetZero(
      encoder,
      z
  );

  applyVCycles(
      backend,
      encoder,
      residual,
      z,
      nbOfReps_
  );
}

} // namespace gpuSolver

