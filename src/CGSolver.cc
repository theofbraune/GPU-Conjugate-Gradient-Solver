#include "GPUSolver/DeviceVector.h"
#include "GPUSolver/Permutation.h"
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/CGSolver.h>
#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Preconditioner.h>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <vector>

namespace gpuSolver {

CGSolver::CGSolver(Backend &backend, Preconditioner &preconditioner,
                   const Matrix &matrix)
    : backend_(backend), preconditioner_(preconditioner), matrix_(matrix),
      hostMatrix_(nullptr), deviceMatrix_(nullptr), maxIterations_(1000),
      tolerance_(1e-6f) {
  if (matrix_.rows() == 0 || matrix_.cols() == 0) {
    throw std::runtime_error("CGSolver: matrix must be non-empty.");
  }

  if (matrix_.rows() != matrix_.cols()) {
    throw std::runtime_error("CGSolver: matrix must be square.");
  }

  matrix_.makeCompressed();

  this->hostMatrix_ = new HostCSRMatrix(
      static_cast<std::size_t>(matrix_.rows()),
      static_cast<std::size_t>(matrix_.cols()),
      static_cast<std::size_t>(matrix_.nonZeros()), matrix_.outerIndexPtr(),
      matrix_.innerIndexPtr(), matrix_.valuePtr());

  this->deviceMatrix_ = this->backend_.createCSRMatrix(*hostMatrix_);

  this->preconditioner_.initialize(backend, *hostMatrix_, *deviceMatrix_);
}

CGSolver::CGSolver(Backend &backend, Preconditioner &preconditioner,
                   const Matrix &matrix, const Permutation &permutationMatrix)
    : backend_(backend), preconditioner_(preconditioner), matrix_(matrix),
      hostMatrix_(nullptr), deviceMatrix_(nullptr), maxIterations_(1000),
      tolerance_(1e-6f) {
  if (matrix_.rows() == 0 || matrix_.cols() == 0) {
    throw std::runtime_error("CGSolver: matrix must be non-empty.");
  }

  if (matrix_.rows() != matrix_.cols()) {
    throw std::runtime_error("CGSolver: matrix must be square.");
  }

  this->permutation_ = &permutationMatrix;

  matrix_.makeCompressed();

  this->hostMatrix_ = new HostCSRMatrix(
      static_cast<std::size_t>(matrix_.rows()),
      static_cast<std::size_t>(matrix_.cols()),
      static_cast<std::size_t>(matrix_.nonZeros()), matrix_.outerIndexPtr(),
      matrix_.innerIndexPtr(), matrix_.valuePtr(), permutationMatrix);

  this->deviceMatrix_ = this->backend_.createCSRMatrix(*hostMatrix_);

  this->preconditioner_.initialize(backend, *hostMatrix_, *deviceMatrix_);
}

CGSolver::~CGSolver() {
  delete deviceMatrix_;
  delete hostMatrix_;
}

void CGSolver::setMaxIterations(std::size_t maxIterations) {
  this->maxIterations_ = maxIterations;
}

void CGSolver::setTolerance(float tolerance) { this->tolerance_ = tolerance; }

std::size_t CGSolver::maxIterations() const { return this->maxIterations_; }

float CGSolver::tolerance() const { return this->tolerance_; }

void CGSolver::solve(const Eigen::VectorXf &b, Eigen::VectorXf &x) {

  // first do the basic sanity checks if the vectors have proper size etc
  if (b.size() != matrix_.rows()) {
    throw std::runtime_error(
        "CGSolver::solve: RHS size does not match matrix.");
  }

  if (x.size() != matrix_.cols()) {
    throw std::runtime_error(
        "CGSolver::solve: initial guess size does not match matrix.");
  }
  Eigen::VectorXf bPermuted = b;
  Eigen::VectorXf xPermuted = x;
  if (this->permutation_) {
    const int *newToOld = this->permutation_->newToOld();
    int size = this->permutation_->size();
    if (size != x.size() || size != b.size()) {
      throw std::runtime_error(
          " the permutation is incompatible with the passed vectors");
    }
    for (std::size_t i = 0; i < x.size(); i++) {
      bPermuted(i) = b(newToOld[i]);
      xPermuted(i) = x(newToOld[i]);
    }
  }
  const float rhsNorm = b.norm();
  constexpr float rhsZeroTolerance = 1e-12f;
  if (rhsNorm < rhsZeroTolerance) {
    x.setZero();
    return;
  }

  const std::size_t n = static_cast<std::size_t>(b.size());

  float *xDat = xPermuted.data();
  const float *bDat = bPermuted.data();

  DeviceVector *xDev = this->backend_.createVector(n, xDat);

  DeviceVector *bDev = this->backend_.createVector(n, bDat);

  DeviceVector *resDev = this->backend_.createVector(n);

  DeviceVector *AxDev = this->backend_.createVector(n);

  BackendEncoder *backendEncoder_ = backend_.createEncoder();
  // compute Ax
  this->backend_.encodeSpmv(*backendEncoder_, *deviceMatrix_, *xDev, *AxDev);

  this->backend_.encodeCopy(*backendEncoder_, *bDev, *resDev);

  // compute the residual vector
  this->backend_.encodeAxpy(*backendEncoder_, -1.0f, *AxDev, *resDev);

  // create the z0 vector
  DeviceVector *zkDev = this->backend_.createVector(n);
  DeviceVector *zkPOneDev = this->backend_.createVector(n);
  this->preconditioner_.apply(backend_, *backendEncoder_, *resDev, *zkDev);

  DeviceVector *pkDev = this->backend_.createVector(n);
  this->backend_.encodeCopy(*backendEncoder_, *zkDev, *pkDev);

  // DeviceScalar *rDotZk = this->backend_.createScalar();
  DeviceScalar *pDotAp = this->backend_.createScalar();
  DeviceVector *ApkDev = this->backend_.createVector(n);
  DeviceVector *resKpOneDev = this->backend_.createVector(n);

  DeviceScalar *alphaKDev = this->backend_.createScalar();
  DeviceScalar *negAlphaKDev = this->backend_.createScalar();
  DeviceScalar *betaKDev = this->backend_.createScalar();

  DeviceScalar *normResSqDev = this->backend_.createScalar();
  DeviceScalar *normRHSSqDev = this->backend_.createScalar();

  DeviceScalar *relativeResidualSquaredDev = this->backend_.createScalar();

  // DeviceScalar *rkPOneDotZkPOne = this->backend_.createScalar();
  DeviceScalar *rhoDev = backend_.createScalar();
  DeviceScalar *rhoNewDev = backend_.createScalar();

  backend_.encodeDot(*backendEncoder_, *bDev, *bDev, *normRHSSqDev);

  this->backend_.encodeDot(*backendEncoder_, *resDev, *zkDev,
                           *rhoDev); // TODO later put here proper class
  // do the check that if the norm of b is too small, just return zero for x and
  // never go in the loop
  backend_.encodeDot(*backendEncoder_, *resDev, *resDev, *normResSqDev);
  backend_.encodeScalarDivide(*backendEncoder_, *normResSqDev, *normRHSSqDev,
                              *relativeResidualSquaredDev);
  backend_.submitAndWait(*backendEncoder_);
  float firstRes = relativeResidualSquaredDev->download();
  delete backendEncoder_;
  backendEncoder_ = nullptr;
  if (!std::isfinite(firstRes)) {
    throw std::runtime_error("CGSolver: initial residual is NaN/Inf — check matrix and RHS.");
  }
  if (firstRes < tolerance_ * tolerance_) {

    delete xDev;
    delete bDev;
    delete resDev;
    delete AxDev;
    delete zkDev;
    delete zkPOneDev;
    delete pkDev;
    delete pDotAp;
    delete ApkDev;
    delete resKpOneDev;
    delete alphaKDev;
    delete negAlphaKDev;
    delete betaKDev;
    delete normRHSSqDev;
    delete normResSqDev;
    delete relativeResidualSquaredDev;
    delete rhoDev;
    delete rhoNewDev;
    this->nOfIterations = 0;
    return;
  }

  for (std::size_t itr = 0; itr < this->maxIterations_; itr++) {
    backendEncoder_ = backend_.createEncoder();
    this->backend_.encodeSpmv(*backendEncoder_, *deviceMatrix_, *pkDev,
                              *ApkDev);
    this->backend_.encodeDot(*backendEncoder_, *pkDev, *ApkDev, *pDotAp);

    this->backend_.encodeScalarDivide(*backendEncoder_, *rhoDev, *pDotAp,
                                      *alphaKDev);

    // now compute the update of x
    backend_.encodeAxpy(*backendEncoder_, *alphaKDev, *pkDev, *xDev);

    backend_.encodeScalarNegate(*backendEncoder_, *alphaKDev, *negAlphaKDev);

    backend_.encodeCopy(*backendEncoder_, *resDev, *resKpOneDev);
    // now compute resNew = res - alpha_k Ap_k
    backend_.encodeAxpy(*backendEncoder_, *negAlphaKDev, *ApkDev, *resKpOneDev);

    backend_.encodeDot(*backendEncoder_, *resKpOneDev, *resKpOneDev,
                       *normResSqDev);

    backend_.encodeScalarDivide(*backendEncoder_, *normResSqDev, *normRHSSqDev,
                                *relativeResidualSquaredDev);

    this->preconditioner_.apply(backend_, *backendEncoder_, *resKpOneDev,
                                *zkPOneDev);
    // now compute the beta_k scalar
    // first compute the 2 dot products that are needed
    backend_.encodeDot(*backendEncoder_, *resKpOneDev, *zkPOneDev, *rhoNewDev);

    backend_.encodeScalarDivide(*backendEncoder_, *rhoNewDev, *rhoDev,
                                *betaKDev);

    // scale pk with beta
    backend_.encodeScale(*backendEncoder_, *pkDev, *betaKDev);

    // now build the new p vector
    backend_.encodeAxpy(*backendEncoder_, 1.0f, *zkPOneDev, *pkDev);

    backend_.submitAndWait(*backendEncoder_);
    float residualHost = relativeResidualSquaredDev->download();
    delete backendEncoder_;
    backendEncoder_ = nullptr;

    if (!std::isfinite(residualHost)) {
      // clean up then throw
      // (same delete block that already exists below)
      throw std::runtime_error("CGSolver: solver diverged (NaN/Inf residual).");
    }
    if (residualHost < tolerance_ * tolerance_) {
      this->nOfIterations = itr;
      break;
    }

    // swap the two rho's
    std::swap(rhoDev, rhoNewDev);
    std::swap(resDev, resKpOneDev);
    std::swap(zkDev, zkPOneDev);
    if (itr == maxIterations_ - 1) {
      this->nOfIterations = this->maxIterations_;
    }
  }

  // convert back to the vector x of what xDev has been
  const float *xResult = xDev->download();

  // check if we need the permutation
  // check if we need the permutation
  if (permutation_ != nullptr) {
    const int *newToOldPerm = this->permutation_->newToOld();
    for (std::size_t i = 0; i < n; ++i) {
      x[static_cast<Eigen::Index>(newToOldPerm[i])] = xResult[i];
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      x[static_cast<Eigen::Index>(i)] = xResult[i];
    }
  }
  // this->nOfIterations = this->maxIterations_;
  //--------------------------clean up----------------
  delete xDev;
  delete bDev;
  delete resDev;
  delete AxDev;
  delete zkDev;
  delete zkPOneDev;
  delete pkDev;
  delete pDotAp;
  delete ApkDev;
  delete resKpOneDev;
  delete alphaKDev;
  delete negAlphaKDev;
  delete betaKDev;
  delete normRHSSqDev;
  delete normResSqDev;
  delete relativeResidualSquaredDev;
  delete rhoDev;
  delete rhoNewDev;
}

std::size_t CGSolver::getNbOfIterations() const { return this->nOfIterations; }
} // namespace gpuSolver
