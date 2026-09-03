#pragma once

#include <cstddef>

namespace gpuSolver {

class Permutation;

class HostCSRMatrix {

private:
  const Permutation *permutation_;

  std::size_t nRows_;
  std::size_t nCols_;
  std::size_t nnz_;

  int* rowPtr_;
  int* colPtr_;
  float* valPtr_;

  int* activeRowPtr_;
  int* activeColPtr_;
  float* activeValPtr_;

public:
  HostCSRMatrix(std::size_t nRows, std::size_t nCols, std::size_t nnz,
                int *rowPtr, int *colIdxPtr, float *valPtr);

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

  bool isPermuted = false;

  const int *activeRowPtr();
  const int *activeColPtr();
  const float *activeValPtr();

  const Permutation* permutation();




};

} // namespace gpuSolver
