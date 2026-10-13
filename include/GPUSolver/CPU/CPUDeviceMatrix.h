#pragma once

#include <GPUSolver/DeviceSparseMatrix.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class HostCSRMatrix;

// CPU-resident CSR matrix. Just owns copies of the three CSR arrays
// -- same storage HostCSRMatrix already uses, duplicated here rather
// than wrapped, so CPUDeviceCSRMatrix behaves like a real "device"
// object you can construct/destroy independently of the host matrix
// that built it (matching what MetalDeviceCSRMatrix/CudaDeviceCSRMatrix
// do: copy in, own the copy).
class CPUDeviceCSRMatrix : public DeviceCSRMatrix {
public:
  CPUDeviceCSRMatrix(std::size_t nRows, std::size_t nCols, std::size_t nnz,
                     const int *rowPtr, const int *colIdxPtr,
                     const float *valPtr);

  explicit CPUDeviceCSRMatrix(const HostCSRMatrix &hostMatrix);

  ~CPUDeviceCSRMatrix() override = default;

  CPUDeviceCSRMatrix(const CPUDeviceCSRMatrix &) = delete;
  CPUDeviceCSRMatrix &operator=(const CPUDeviceCSRMatrix &) = delete;

  std::size_t rows() const override;
  std::size_t cols() const override;
  std::size_t nnz() const override;

  void updateValues(const float *values, std::size_t numberOfValues) override;

  const int *rowPtr() const;
  const int *colIdx() const;
  const float *values() const;

private:
  std::size_t nRows_;
  std::size_t nCols_;
  std::size_t nnz_;

  std::vector<int> rowPtr_;
  std::vector<int> colIdx_;
  std::vector<float> values_;
};

} // namespace gpuSolver
