#include <GPUSolver/CGSolver.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/DeviceSparseMatrix.h>

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

void CGSolver::setMaxIterations(std::size_t maxIterations){
  this->maxIterations_ = maxIterations;
}

void CGSolver::setTolerance(float tolerance){
  this->tolerance_ = tolerance;
}

std::size_t CGSolver::maxIterations() const{
  return this->maxIterations_;
}

float CGSolver::tolerance() const {
  return this->tolerance_;
}

} // namespace gpuSolver
