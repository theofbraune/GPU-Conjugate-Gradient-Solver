#include "GPUSolver/Permutation.h"
#include <GPUSolver/Smoothers/MultigridSmoother.h>

#include <GPUSolver/Backend.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>

#include <GPUSolver/Smoothers/BlockGaussSeidelSmoother.h>
#include <GPUSolver/Smoothers/BlockJacobiSmoother.h>
#include <GPUSolver/Smoothers/DampedJacobiSmoother.h>
#include <GPUSolver/Smoothers/SymmetricGaussSeidelSmoother.h>

// Use the header containing your existing permuteMatrix().
#include <GPUSolver/SparseMatrixUtils.h>

#include <stdexcept>
#include <vector>

namespace gpuSolver {

MultigridSmoother::MultigridSmoother(
    Backend &backend, const AMGHierarchy &hierarchy,
    const std::vector<Permutation> &permutations, SmootherType fineSmootherType,
    SmootherType coarseSmootherType, std::size_t preSmoothingSteps,
    std::size_t postSmoothingSteps, std::size_t coarseSmoothingSteps,
    float omega) {

  const std::size_t numberOfLevels = hierarchy.A.size();
  this->preSmoothingSteps_ = preSmoothingSteps;
  this->postSmoothingSteps_ = postSmoothingSteps;
  this->coarseSmoothingSteps_ = coarseSmoothingSteps;

  if (numberOfLevels == 0) {
    throw std::runtime_error("MultigridSmoother: empty hierarchy.");
  }

  if (permutations.size() != numberOfLevels) {
    throw std::runtime_error(
        "MultigridSmoother: invalid number of permutations.");
  }

  if (hierarchy.P.size() + 1 != numberOfLevels ||
      hierarchy.R.size() + 1 != numberOfLevels) {
    throw std::runtime_error("MultigridSmoother: invalid number of transfers.");
  }

  levels_.resize(numberOfLevels);
  transfers_.resize(numberOfLevels - 1);
  // ------------------------------------------------------

  for (std::size_t k = 0; k < numberOfLevels; ++k) {

    const int *oldToNewK = permutations[k].oldToNew();

    // A'_k = Pi_k A_k Pi_k^T

    AMGHierarchy::Matrix Ak =
        permuteMatrix(hierarchy.A[k], oldToNewK, oldToNewK);

    Ak.makeCompressed();

    const std::size_t n = static_cast<std::size_t>(Ak.rows());

    HostCSRMatrix hostAk(n, n, static_cast<std::size_t>(Ak.nonZeros()),
                         Ak.outerIndexPtr(), Ak.innerIndexPtr(), Ak.valuePtr());

    Level &currentLevel = levels_[k];

    // Upload the Galerkin matrix.
    currentLevel.matrix = backend.createCSRMatrix(hostAk);

    // ----------------------------------------------------
    // Construct the requested smoother.
    // ----------------------------------------------------

    if (k == 0) {
      switch (fineSmootherType) {

      case SmootherType::DampedJacobi:

        currentLevel.smoother = new DampedJacobiSmoother(omega);

        break;

      case SmootherType::SymmetricGaussSeidel:

        currentLevel.smoother = new SymmetricGaussSeidelSmoother(omega);

        break;
      case SmootherType::BlockJacobi:

        currentLevel.smoother = new BlockJacobiSmoother(omega);

        break;

      case SmootherType::BlockGaussSeidel:

        currentLevel.smoother = new BlockGaussSeidelSmoother(omega);

        break;

      default:

        throw std::runtime_error(
            "MultigridSmoother: unsupported smoother type.");
      }
    }
    if(k>0){

      switch (coarseSmootherType) {

      case SmootherType::DampedJacobi:

        currentLevel.smoother = new DampedJacobiSmoother(omega);

        break;

      case SmootherType::SymmetricGaussSeidel:

        currentLevel.smoother = new SymmetricGaussSeidelSmoother(omega);

        break;
      case SmootherType::BlockJacobi:

        currentLevel.smoother = new BlockJacobiSmoother(omega);

        break;

      case SmootherType::BlockGaussSeidel:

        currentLevel.smoother = new BlockGaussSeidelSmoother(omega);

        break;

      default:

        throw std::runtime_error(
            "MultigridSmoother: unsupported smoother type.");
      }
    }

    // Initialize using the matrix in this level's ordering.
    currentLevel.smoother->initialize(backend, hostAk, *currentLevel.matrix);

    // Allocate V-cycle scratch vectors.
    currentLevel.rhs = backend.createVector(n);

    currentLevel.residual = backend.createVector(n);

    currentLevel.correction = backend.createVector(n);

    currentLevel.Az = backend.createVector(n);

    // ----------------------------------------------------
    // Construct the transfer from level k-1 to level k.
    // ----------------------------------------------------

    if (k > 0) {

      const int *oldToNewPrevious = permutations[k - 1].oldToNew();

      // P'_(k-1) = Pi_(k-1) P_(k-1) Pi_k^T

      AMGHierarchy::Matrix Pk =
          permuteMatrix(hierarchy.P[k - 1], oldToNewPrevious, oldToNewK);

      // R'_(k-1) = Pi_k R_(k-1) Pi_(k-1)^T

      AMGHierarchy::Matrix Rk =
          permuteMatrix(hierarchy.R[k - 1], oldToNewK, oldToNewPrevious);

      Pk.makeCompressed();
      Rk.makeCompressed();

      HostCSRMatrix hostPk(static_cast<std::size_t>(Pk.rows()),
                           static_cast<std::size_t>(Pk.cols()),
                           static_cast<std::size_t>(Pk.nonZeros()),
                           Pk.outerIndexPtr(), Pk.innerIndexPtr(),
                           Pk.valuePtr());

      HostCSRMatrix hostRk(static_cast<std::size_t>(Rk.rows()),
                           static_cast<std::size_t>(Rk.cols()),
                           static_cast<std::size_t>(Rk.nonZeros()),
                           Rk.outerIndexPtr(), Rk.innerIndexPtr(),
                           Rk.valuePtr());

      Transfer &transfer = transfers_[k - 1];

      transfer.prolongation = backend.createCSRMatrix(hostPk);

      transfer.restriction = backend.createCSRMatrix(hostRk);
    }
  }
}

MultigridSmoother::MultigridSmoother(
    Backend &backend, const AMGHierarchy &hierarchy,
    const std::vector<Permutation> &permutations, SmootherType smootherType,
    std::size_t preSmoothingSteps, std::size_t postSmoothingSteps,
    std::size_t coarseSmoothingSteps, float omega)
    : preSmoothingSteps_(preSmoothingSteps),
      postSmoothingSteps_(postSmoothingSteps),
      coarseSmoothingSteps_(coarseSmoothingSteps) {
  const std::size_t numberOfLevels = hierarchy.A.size();

  if (numberOfLevels == 0) {
    throw std::runtime_error("MultigridSmoother: empty hierarchy.");
  }

  if (permutations.size() != numberOfLevels) {
    throw std::runtime_error(
        "MultigridSmoother: invalid number of permutations.");
  }

  if (hierarchy.P.size() + 1 != numberOfLevels ||
      hierarchy.R.size() + 1 != numberOfLevels) {
    throw std::runtime_error("MultigridSmoother: invalid number of transfers.");
  }

  levels_.resize(numberOfLevels);
  transfers_.resize(numberOfLevels - 1);

  // ------------------------------------------------------
  // Construct every multigrid level.
  // ------------------------------------------------------

  for (std::size_t k = 0; k < numberOfLevels; ++k) {

    const int *oldToNewK = permutations[k].oldToNew();

    // A'_k = Pi_k A_k Pi_k^T

    AMGHierarchy::Matrix Ak =
        permuteMatrix(hierarchy.A[k], oldToNewK, oldToNewK);

    Ak.makeCompressed();

    const std::size_t n = static_cast<std::size_t>(Ak.rows());

    HostCSRMatrix hostAk(n, n, static_cast<std::size_t>(Ak.nonZeros()),
                         Ak.outerIndexPtr(), Ak.innerIndexPtr(), Ak.valuePtr());

    Level &currentLevel = levels_[k];

    // Upload the Galerkin matrix.
    currentLevel.matrix = backend.createCSRMatrix(hostAk);

    // ----------------------------------------------------
    // Construct the requested smoother.
    // ----------------------------------------------------

    switch (smootherType) {

    case SmootherType::DampedJacobi:

      currentLevel.smoother = new DampedJacobiSmoother(omega);

      break;

    case SmootherType::SymmetricGaussSeidel:

      currentLevel.smoother = new SymmetricGaussSeidelSmoother(omega);

      break;
    case SmootherType::BlockJacobi:

      currentLevel.smoother = new BlockJacobiSmoother(omega);

      break;

    case SmootherType::BlockGaussSeidel:

      currentLevel.smoother = new BlockGaussSeidelSmoother(omega);

      break;

    default:

      throw std::runtime_error("MultigridSmoother: unsupported smoother type.");
    }

    // Initialize using the matrix in this level's ordering.
    currentLevel.smoother->initialize(backend, hostAk, *currentLevel.matrix);

    // Allocate V-cycle scratch vectors.
    currentLevel.rhs = backend.createVector(n);

    currentLevel.residual = backend.createVector(n);

    currentLevel.correction = backend.createVector(n);

    currentLevel.Az = backend.createVector(n);

    // ----------------------------------------------------
    // Construct the transfer from level k-1 to level k.
    // ----------------------------------------------------

    if (k > 0) {

      const int *oldToNewPrevious = permutations[k - 1].oldToNew();

      // P'_(k-1) = Pi_(k-1) P_(k-1) Pi_k^T

      AMGHierarchy::Matrix Pk =
          permuteMatrix(hierarchy.P[k - 1], oldToNewPrevious, oldToNewK);

      // R'_(k-1) = Pi_k R_(k-1) Pi_(k-1)^T

      AMGHierarchy::Matrix Rk =
          permuteMatrix(hierarchy.R[k - 1], oldToNewK, oldToNewPrevious);

      Pk.makeCompressed();
      Rk.makeCompressed();

      HostCSRMatrix hostPk(static_cast<std::size_t>(Pk.rows()),
                           static_cast<std::size_t>(Pk.cols()),
                           static_cast<std::size_t>(Pk.nonZeros()),
                           Pk.outerIndexPtr(), Pk.innerIndexPtr(),
                           Pk.valuePtr());

      HostCSRMatrix hostRk(static_cast<std::size_t>(Rk.rows()),
                           static_cast<std::size_t>(Rk.cols()),
                           static_cast<std::size_t>(Rk.nonZeros()),
                           Rk.outerIndexPtr(), Rk.innerIndexPtr(),
                           Rk.valuePtr());

      Transfer &transfer = transfers_[k - 1];

      transfer.prolongation = backend.createCSRMatrix(hostPk);

      transfer.restriction = backend.createCSRMatrix(hostRk);
    }
  }
}

void MultigridSmoother::vCycle(Backend &backend, BackendEncoder &encoder,
                               std::size_t level, const DeviceVector &rhs,
                               DeviceVector &solution) {
  Level &currentLevel = levels_[level];

  const bool isCoarsest = level + 1 == levels_.size();

  // ------------------------------------------------------
  // Coarsest level.
  // ------------------------------------------------------

  if (isCoarsest) {

    currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                  coarseSmoothingSteps_);

    return;
  }

  // ------------------------------------------------------
  // Pre-smoothing.
  // ------------------------------------------------------

  currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                preSmoothingSteps_);

  // ------------------------------------------------------
  // r = rhs - A * solution
  // ------------------------------------------------------

  backend.encodeSpmv(encoder, *currentLevel.matrix, solution, *currentLevel.Az);

  backend.encodeCopy(encoder, rhs, *currentLevel.residual);

  backend.encodeAxpy(encoder, -1.0f, *currentLevel.Az, *currentLevel.residual);

  // ------------------------------------------------------
  // Restrict residual.
  // ------------------------------------------------------

  Transfer &transfer = transfers_[level];

  Level &coarseLevel = levels_[level + 1];

  backend.encodeSpmv(encoder, *transfer.restriction, *currentLevel.residual,
                     *coarseLevel.rhs);

  // ------------------------------------------------------
  // Initialize coarse correction.
  // ------------------------------------------------------

  backend.encodeSetZero(encoder, *coarseLevel.correction);

  // ------------------------------------------------------
  // Recursive coarse solve.
  // ------------------------------------------------------

  vCycle(backend, encoder, level + 1, *coarseLevel.rhs,
         *coarseLevel.correction);

  // ------------------------------------------------------
  // Prolongate coarse correction.
  // ------------------------------------------------------

  backend.encodeSpmv(encoder, *transfer.prolongation, *coarseLevel.correction,
                     *currentLevel.correction);

  backend.encodeAxpy(encoder, 1.0f, *currentLevel.correction, solution);

  // ------------------------------------------------------
  // Post-smoothing.
  // ------------------------------------------------------

  currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                postSmoothingSteps_);
}

void MultigridSmoother::smooth(Backend &backend, BackendEncoder &encoder,
                               const DeviceVector &rhs, DeviceVector &solution,
                               std::size_t iterations) {
  for (std::size_t i = 0; i < iterations; ++i) {
    vCycle(backend, encoder, 0, rhs, solution);
  }
}

void MultigridSmoother::initialize(Backend & /*backend*/,
                                   const HostCSRMatrix & /*hostMatrix*/,
                                   const DeviceCSRMatrix & /*deviceMatrix*/) {
  // Already initialized in the constructor.
}

MultigridSmoother::~MultigridSmoother() {
  for (Level &level : levels_) {

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

} // namespace gpuSolver
