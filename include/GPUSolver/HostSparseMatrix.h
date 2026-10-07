#pragma once

#include <Eigen/SparseCore>
#include <cstddef>
#include <Eigen/Sparse>

namespace gpuSolver {

class Permutation;

class HostCSRMatrix {

private:
  Permutation *permutation_;

  std::size_t nRows_;
  std::size_t nCols_;
  std::size_t nnz_;

  int *rowPtr_;
  int *colPtr_;
  float *valPtr_;

  int *activeRowPtr_;
  int *activeColPtr_;
  float *activeValPtr_;

  // permuted CSR, also owned by this object
  int* permutedRowPtr_;
  int* permutedColPtr_;
  float* permutedValPtr_;

public:
  HostCSRMatrix(std::size_t nRows, std::size_t nCols, std::size_t nnz,
                const int *rowPtr, const int *colIdxPtr, const float *valPtr);

  HostCSRMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);

  HostCSRMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A, const Permutation& permutation);

  HostCSRMatrix(std::size_t nRows, std::size_t nCols, std::size_t nnz,
                const int *rowPtr, const int *colIdxPtr, const float *valPtr, const Permutation& permutation);

  ~HostCSRMatrix();

  HostCSRMatrix(const HostCSRMatrix &) = delete;
  HostCSRMatrix &operator=(const HostCSRMatrix &) = delete;

  std::size_t rows() const;
  std::size_t cols() const;
  std::size_t nnz() const;

  const int *rowPtr() const;
  const int *colIdx() const;
  const float *values() const;

  void applyPermutation(const Permutation &permutation);

  bool isPermuted() const;

  const int *activeRowPtr() const;
  const int *activeColPtr() const;
  const float *activeValPtr() const;

  const Permutation *permutation() const;
};

} // namespace gpuSolver
