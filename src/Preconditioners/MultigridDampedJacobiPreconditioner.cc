#include "GPUSolver/DeviceSparseMatrix.h"
#include "GPUSolver/DeviceVector.h"
#include <GPUSolver/Backend.h>
#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/SparseMatrixUtils.h>
#include <cstddef>

namespace gpuSolver {
MultigridDampedJacobiPreconditioner::Level::Level()
    : matrix(nullptr),
      smoother(nullptr),
      residual(nullptr),
      correction(nullptr),
      Az(nullptr),
      rhs(nullptr)
{
}

MultigridDampedJacobiPreconditioner::Transfer::Transfer()
    : restriction(nullptr),
      prolongation(nullptr)
{
}
MultigridDampedJacobiPreconditioner::MultigridDampedJacobiPreconditioner(
    Backend &backend, const AMGHierarchy &hierarchy,
    const std::vector<Permutation> &permutations, std::size_t preSmoothingSteps,
    std::size_t postSmoothingSteps, std::size_t nbOfReps, float omega) {
  this->omega_ = omega;
  this->preSmoothingSteps_ = preSmoothingSteps;
  this->postSmoothingSteps_ = postSmoothingSteps;
  this->nbOfReps_ = nbOfReps;

  const std::size_t numberOfLevels = hierarchy.A.size();

  if (permutations.size() != numberOfLevels) {
    throw std::runtime_error("MultigridDampedJacobiPreconditioner: "
                             "expected one Permutation per hierarchy level.");
  }
  if (numberOfLevels == 0) {
    throw std::runtime_error(
        "MultigridDampedJacobiPreconditioner: empty hierarchy.");
  }

  levels_.resize(numberOfLevels);
  transfers_.resize(numberOfLevels - 1);
  for (std::size_t k = 0; k < numberOfLevels; ++k) {
    const int *oldToNewK = permutations[k].oldToNew();
    Eigen::SparseMatrix<float, Eigen::RowMajor> Akp =
        permuteMatrix(hierarchy.A[k], oldToNewK, oldToNewK);

    std::size_t nRows = Akp.rows();
    std::size_t nCols = Akp.cols();
    std::size_t nnz = Akp.nonZeros();

    const int *rowPtrA = Akp.outerIndexPtr();
    const int *colPtrA = Akp.innerIndexPtr();
    const float *valPtr = Akp.valuePtr();

    HostCSRMatrix hostAkp(nRows, nCols, nnz, rowPtrA, colPtrA, valPtr);
    DeviceCSRMatrix *deviceAkp = backend.createCSRMatrix(hostAkp);

    DampedJacobiPreconditioner *smoother =
        new DampedJacobiPreconditioner(preSmoothingSteps_, omega_);
    smoother->initialize(backend, hostAkp, *deviceAkp);

    levels_[k].matrix = deviceAkp;
    levels_[k].smoother = smoother;
    levels_[k].residual = backend.createVector(nRows);
    levels_[k].correction = backend.createVector(nRows);
    levels_[k].Az = backend.createVector(nRows);
    levels_[k].rhs = backend.createVector(nRows);

    if (k > 0) {
      const int *oldToNewKM1 = permutations[k - 1].oldToNew();
      Eigen::SparseMatrix<float, Eigen::RowMajor> Pkp =
          permuteMatrix(hierarchy.P[k - 1], oldToNewKM1, oldToNewK);
      Eigen::SparseMatrix<float, Eigen::RowMajor> Rkp =
          permuteMatrix(hierarchy.R[k - 1], oldToNewK, oldToNewKM1);

      // fetch the pointers from the restriction and permutation matrix

      std::size_t nRowsP = Pkp.rows();
      std::size_t nColsP = Pkp.cols();
      std::size_t nnzP = Pkp.nonZeros();
      const int *rowPtrP = Pkp.outerIndexPtr();
      const int *colPtrP = Pkp.innerIndexPtr();
      const float *valPtrP = Pkp.valuePtr();

      HostCSRMatrix hostPkp(nRowsP, nColsP, nnzP, rowPtrP, colPtrP, valPtrP);
      //----------------------------------------
      std::size_t nRowsR = Rkp.rows();
      std::size_t nColsR = Rkp.cols();
      std::size_t nnzR = Rkp.nonZeros();
      const int *rowPtrR = Rkp.outerIndexPtr();
      const int *colPtrR = Rkp.innerIndexPtr();
      const float *valPtrR = Rkp.valuePtr();
      HostCSRMatrix hostRkp(nRowsR, nColsR, nnzR, rowPtrR, colPtrR, valPtrR);

      transfers_[k - 1].prolongation = backend.createCSRMatrix(hostPkp);
      transfers_[k - 1].restriction = backend.createCSRMatrix(hostRkp);
    }
  }
}

void MultigridDampedJacobiPreconditioner::initialize(
    Backend & /*backend*/, const HostCSRMatrix & /*hostMatrix*/,
    const DeviceCSRMatrix & /*deviceMatrix*/) {
  // Intentionally empty: MultigridDampedJacobiPreconditioner does all of
  // its setup in the constructor, since it needs the full AMG hierarchy
  // and one Permutation per level — information the base Preconditioner
  // interface has no way to carry. This override exists only to satisfy
  // the pure-virtual base contract; CGSolver calling initialize() on this
  // class is a correct, deliberate no-op.
}
MultigridDampedJacobiPreconditioner::~MultigridDampedJacobiPreconditioner() {
  for (Level &level : levels_) {
    delete level.matrix;
    delete level.smoother;
    delete level.residual;
    delete level.correction;
    delete level.Az;
    delete level.rhs;
  }
  for (Transfer &transfer : transfers_) {
    delete transfer.restriction;
    delete transfer.prolongation;
  }
}

void MultigridDampedJacobiPreconditioner::smooth(Backend &backend,
                                                 BackendEncoder &encoder,
                                                 const DeviceVector &rhs,
                                                 DeviceVector &solution,
                                                 std::size_t numberOfSteps) {

  for (std::size_t itr = 0; itr < numberOfSteps; itr++) {
    this->vCycle(backend, encoder, 0, rhs, solution);
  }
}

void MultigridDampedJacobiPreconditioner::vCycle(Backend &backend,
                                                 BackendEncoder &encoder,
                                                 std::size_t level,
                                                 const DeviceVector &rhs,
                                                 DeviceVector &solution) {

  Level &currentLevel = this->levels_[level];

  const bool isCoarsest = (level + 1 == this->levels_.size());

  if (isCoarsest) {

    currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                  (preSmoothingSteps_ + postSmoothingSteps_));
    return;
  }

  currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                preSmoothingSteps_);
  // now after smoothing correct the residual
  // Residual: lvl.residual = residual - A_k * solution.
  backend.encodeSpmv(encoder, *currentLevel.matrix, solution,
                     *currentLevel.Az); // now level.residual = Az
  backend.encodeCopy(encoder, rhs, *currentLevel.residual);
  backend.encodeAxpy(encoder, -1.0f, *currentLevel.Az, *currentLevel.residual);

  // now restrict the residual on the coarser level
  Transfer &transf = this->transfers_[level];
  Level &coarseLevel = this->levels_[level + 1];
  // now restrict the residual on the coarser level
  backend.encodeSpmv(encoder, *transf.restriction, *currentLevel.residual,
                     *coarseLevel.rhs);
  backend.encodeSetZero(encoder, *coarseLevel.correction);
  this->vCycle(backend, encoder, level + 1, *coarseLevel.rhs,
               *coarseLevel.correction);

  // now do the prolongation and the post smoothing
  backend.encodeSpmv(encoder, *transf.prolongation, *coarseLevel.correction,
                     *currentLevel.correction);

  backend.encodeAxpy(encoder, 1.0f, *currentLevel.correction, solution);

  // now apply the smoother again
  currentLevel.smoother->smooth(backend, encoder, rhs, solution,
                                this->postSmoothingSteps_);
}

void MultigridDampedJacobiPreconditioner::apply(Backend &backend,
                                                BackendEncoder &encoder,
                                                const DeviceVector &residual,
                                                DeviceVector &z) {

  backend.encodeSetZero(encoder, z);
  this->smooth(backend, encoder, residual, z, this->nbOfReps_);
}
} // namespace gpuSolver
