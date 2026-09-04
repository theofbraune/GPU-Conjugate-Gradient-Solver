#pragma once

#include <cstddef>
namespace gpuSolver {

class HostCSRMatrix;

class HostELLMatrix {
  std::size_t nRows_;
  std::size_t nCols_;
  std::size_t nnz_;

  std::size_t ellWidth_;
  std::size_t overflowNnz_;

  // ELL part, stored slot-major:
  //
  // index = slot * nRows + row
  //
  int *ellColIdx_;
  float *ellValues_;

  // Overflow part stored as CSR.
  int *overflowRowPtr_;
  int *overflowColIdx_;
  float *overflowValues_;

public:

  HostELLMatrix(const HostCSRMatrix &csr, std::size_t ellWidth);

  ~HostELLMatrix();

  HostELLMatrix(const HostELLMatrix &) = delete;
  HostELLMatrix &operator=(const HostELLMatrix &) = delete;

  std::size_t rows() const;
  std::size_t cols() const;
  std::size_t nnz() const;

  std::size_t ellWidth() const;
  std::size_t overflowNnz() const;

  const int *ellColIdx() const;
  const float *ellValues() const;

  const int *overflowRowPtr() const;
  const int *overflowColIdx() const;
  const float *overflowValues() const;
};

} // namespace gpuSolver
