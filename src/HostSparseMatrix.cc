#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace gpuSolver {
HostCSRMatrix::HostCSRMatrix(std::size_t nRows, std::size_t nCols,
                             std::size_t nnz, int *rowPtr, int *colIdxPtr,
                             float *valPtr) {

  if (nRows == 0 || nCols == 0) {
    throw std::runtime_error(
        "HostCSRMatrix: matrix dimensions must be nonzero.");
  }

  if (rowPtr == nullptr || colIdxPtr == nullptr || valPtr == nullptr) {
    throw std::runtime_error("HostCSRMatrix: input pointers must not be null.");
  }
  this->nRows_ = nRows;
  this->nCols_ = nCols;
  this->nnz_ = nnz;

  this->activeRowPtr_ = nullptr;
  this->activeColPtr_ = nullptr;
  this->activeValPtr_ = nullptr;

  this->rowPtr_ = nullptr;
  this->colPtr_ = nullptr;
  this->valPtr_ = nullptr;
  this->permutation_ = nullptr;

  this->permutedRowPtr_ = nullptr;
  this->permutedColPtr_ = nullptr;
  this->permutedValPtr_ = nullptr;

  // allocate the internal pointers
  //
  this->rowPtr_ = new int[nRows + 1];

  this->colPtr_ = new int[nnz_];

  this->valPtr_ = new float[nnz_];

  std::memcpy(rowPtr_, rowPtr, (nRows_ + 1) * sizeof(int));

  std::memcpy(colPtr_, colIdxPtr, nnz_ * sizeof(int));

  std::memcpy(valPtr_, valPtr, nnz_ * sizeof(float));

  activeRowPtr_ = rowPtr_;
  activeColPtr_ = colPtr_;
  activeValPtr_ = valPtr_;
}

HostCSRMatrix::HostCSRMatrix(std::size_t nRows, std::size_t nCols,
                             std::size_t nnz, int *rowPtr, int *colIdxPtr,
                             float *valPtr, const Permutation &permutation)
    : HostCSRMatrix(nRows, nCols, nnz, rowPtr, colIdxPtr, valPtr) {

  this->applyPermutation(permutation);
}

HostCSRMatrix::~HostCSRMatrix() {

  delete[] rowPtr_;
  delete[] colPtr_;
  delete[] valPtr_;

  delete[] permutedRowPtr_;
  delete[] permutedColPtr_;
  delete[] permutedValPtr_;

  delete permutation_;
}

const int *HostCSRMatrix::activeRowPtr() const {
  if (activeRowPtr_ != nullptr) {
    return activeRowPtr_;
  }

  return rowPtr_;
}

const int *HostCSRMatrix::activeColPtr() const {
  if (activeColPtr_ != nullptr) {
    return activeColPtr_;
  }

  return colPtr_;
}

const float *HostCSRMatrix::activeValPtr() const {
  if (activeValPtr_ != nullptr) {
    return activeValPtr_;
  }

  return valPtr_;
}

bool HostCSRMatrix::isPermuted() const { return permutation_ != nullptr; }

void HostCSRMatrix::applyPermutation(const Permutation &permutation) {
  // For now we only support a simultaneous row/column permutation,
  // i.e. A' = P A P^T, so the matrix must be square.
  if (nRows_ != nCols_) {
    throw std::runtime_error(
        "HostCSRMatrix::applyPermutation: matrix must be square.");
  }

  if (permutation.size() != nRows_) {
    throw std::runtime_error(
        "HostCSRMatrix::applyPermutation: permutation size does not "
        "match matrix dimensions.");
  }

  const int *oldToNew = permutation.oldToNew();

  const int *newToOld = permutation.newToOld();

  if (oldToNew == nullptr || newToOld == nullptr) {
    throw std::runtime_error(
        "HostCSRMatrix::applyPermutation: invalid permutation.");
  }

  // Remove a previously stored permutation, if there is one.
  delete[] permutedRowPtr_;
  delete[] permutedColPtr_;
  delete[] permutedValPtr_;
  delete permutation_;

  permutedRowPtr_ = nullptr;
  permutedColPtr_ = nullptr;
  permutedValPtr_ = nullptr;
  permutation_ = nullptr;

  // Store our own copy of the permutation.
  permutation_ = new Permutation(permutation.size(), permutation.oldToNew(),
                                 permutation.newToOld());

  // Allocate the CSR representation of
  //
  //     A' = P A P^T.
  //
  permutedRowPtr_ = new int[nRows_ + 1];

  permutedColPtr_ = new int[nnz_];

  permutedValPtr_ = new float[nnz_];

  // --------------------------------------------------------
  // Build the row pointer.
  //
  // newToOld[newRow] tells us which row of the original
  // matrix corresponds to this row in the new ordering.
  //
  // Permutation does not change the number of entries in
  // a row, only which row they belong to.
  // --------------------------------------------------------

  permutedRowPtr_[0] = 0;

  for (std::size_t newRow = 0; newRow < nRows_; ++newRow) {
    const int oldRow = newToOld[newRow];

    if (oldRow < 0 || oldRow >= static_cast<int>(nRows_)) {
      throw std::runtime_error(
          "HostCSRMatrix::applyPermutation: invalid newToOld index.");
    }

    const int numberOfEntries = rowPtr_[oldRow + 1] - rowPtr_[oldRow];

    permutedRowPtr_[newRow + 1] = permutedRowPtr_[newRow] + numberOfEntries;
  }

  // --------------------------------------------------------
  // Fill column indices and values.
  //
  // An original entry
  //
  //     A(oldRow, oldCol)
  //
  // becomes
  //
  //     A'(newRow, newCol)
  //
  // where
  //
  //     newRow = oldToNew[oldRow]
  //     newCol = oldToNew[oldCol].
  //
  // Since we iterate over newRow, oldRow is obtained through
  // newToOld[newRow].
  // --------------------------------------------------------

  for (std::size_t newRow = 0; newRow < nRows_; ++newRow) {
    const int oldRow = newToOld[newRow];

    int destination = permutedRowPtr_[newRow];

    for (int k = rowPtr_[oldRow]; k < rowPtr_[oldRow + 1]; ++k) {
      const int oldCol = colPtr_[k];

      if (oldCol < 0 || oldCol >= static_cast<int>(nCols_)) {
        throw std::runtime_error(
            "HostCSRMatrix::applyPermutation: original column "
            "index out of bounds.");
      }

      const int newCol = oldToNew[oldCol];

      if (newCol < 0 || newCol >= static_cast<int>(nCols_)) {
        throw std::runtime_error(
            "HostCSRMatrix::applyPermutation: invalid oldToNew index.");
      }

      permutedColPtr_[destination] = newCol;

      permutedValPtr_[destination] = valPtr_[k];

      ++destination;
    }
  }

  if (permutedRowPtr_[nRows_] != static_cast<int>(nnz_)) {
    throw std::runtime_error(
        "HostCSRMatrix::applyPermutation: permuted CSR does not "
        "contain the expected number of nonzeros.");
  }

  this->activeRowPtr_ = permutedRowPtr_;
  this->activeColPtr_ = permutedColPtr_;
  this->activeValPtr_ = permutedValPtr_;
}

std::size_t HostCSRMatrix::rows() const { return nRows_; }

std::size_t HostCSRMatrix::cols() const { return nCols_; }

std::size_t HostCSRMatrix::nnz() const { return nnz_; }

const int *HostCSRMatrix::rowPtr() const { return rowPtr_; }

const int *HostCSRMatrix::colIdx() const { return colPtr_; }

const float *HostCSRMatrix::values() const { return valPtr_; }

const Permutation *HostCSRMatrix::permutation() const { return permutation_; }

} // namespace gpuSolver
