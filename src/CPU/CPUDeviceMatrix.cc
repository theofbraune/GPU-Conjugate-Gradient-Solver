#include <GPUSolver/CPU/CPUDeviceMatrix.h>

#include <GPUSolver/HostSparseMatrix.h>

#include <algorithm>
#include <stdexcept>

namespace gpuSolver {

CPUDeviceCSRMatrix::CPUDeviceCSRMatrix(std::size_t nRows, std::size_t nCols,
                                       std::size_t nnz, const int *rowPtr,
                                       const int *colIdxPtr,
                                       const float *valPtr)
    : nRows_(nRows), nCols_(nCols), nnz_(nnz),
      rowPtr_(rowPtr, rowPtr + nRows + 1),
      colIdx_(colIdxPtr, colIdxPtr + nnz), values_(valPtr, valPtr + nnz) {
  if (nRows == 0 || nCols == 0) {
    throw std::runtime_error(
        "CPUDeviceCSRMatrix: matrix dimensions must be nonzero.");
  }
  if (rowPtr == nullptr || colIdxPtr == nullptr || valPtr == nullptr) {
    throw std::runtime_error(
        "CPUDeviceCSRMatrix: nullptr passed to constructor.");
  }
  if (rowPtr_[0] != 0) {
    throw std::runtime_error(
        "CPUDeviceCSRMatrix: CSR row pointer must start at zero.");
  }
  if (rowPtr_[nRows] != static_cast<int>(nnz)) {
    throw std::runtime_error(
        "CPUDeviceCSRMatrix: CSR row pointer does not match nnz.");
  }
}

// Delegates to the raw-pointer constructor using the host matrix's
// *active* arrays -- active means "permuted, if a permutation has
// been applied" (see HostCSRMatrix::applyPermutation). This mirrors
// what MetalDeviceCSRMatrix's HostCSRMatrix-constructor overload does.
CPUDeviceCSRMatrix::CPUDeviceCSRMatrix(const HostCSRMatrix &hostMatrix)
    : CPUDeviceCSRMatrix(hostMatrix.rows(), hostMatrix.cols(),
                         hostMatrix.nnz(), hostMatrix.activeRowPtr(),
                         hostMatrix.activeColPtr(),
                         hostMatrix.activeValPtr()) {}

std::size_t CPUDeviceCSRMatrix::rows() const { return nRows_; }
std::size_t CPUDeviceCSRMatrix::cols() const { return nCols_; }
std::size_t CPUDeviceCSRMatrix::nnz() const { return nnz_; }

void CPUDeviceCSRMatrix::updateValues(const float *values,
                                      std::size_t numberOfValues) {
  if (numberOfValues != nnz_) {
    throw std::runtime_error(
        "CPUDeviceCSRMatrix::updateValues: size mismatch.");
  }
  std::copy(values, values + numberOfValues, values_.begin());
}

const int *CPUDeviceCSRMatrix::rowPtr() const { return rowPtr_.data(); }
const int *CPUDeviceCSRMatrix::colIdx() const { return colIdx_.data(); }
const float *CPUDeviceCSRMatrix::values() const { return values_.data(); }

} // namespace gpuSolver
