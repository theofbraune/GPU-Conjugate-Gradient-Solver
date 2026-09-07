#include "GPUSolver/DeviceVector.h"
#include <GPUSolver/CGSolver.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>
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

  // this->deviceMatrix_ = this->backend_
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

  const std::size_t n = static_cast<std::size_t>(b.size());

  float* xDat = x.data();
  const float* bDat = b.data();

  // DeviceVector xDev = DeviceVector()
}

} // namespace gpuSolver
