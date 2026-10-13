#pragma once

#include <GPUSolver/DeviceELLMatrix.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class HostELLMatrix;

// Backend::createELLMatrix is pure virtual, so every backend has to
// provide *some* concrete DeviceELLMatrix, even backends (like this
// one, for now) that don't yet have their own ELL SpMV kernel. This
// just stores the data; add a CPU ELL SpMV later if you want to
// benchmark the ELL format itself across backends, same way
// MetalBackend::encodeSpmvELL does it as an extra, non-Backend-
// interface method.
class CPUDeviceELLMatrix : public DeviceELLMatrix {
public:
  explicit CPUDeviceELLMatrix(const HostELLMatrix &hostMatrix);

  ~CPUDeviceELLMatrix() override = default;

  CPUDeviceELLMatrix(const CPUDeviceELLMatrix &) = delete;
  CPUDeviceELLMatrix &operator=(const CPUDeviceELLMatrix &) = delete;

  std::size_t rows() const override;
  std::size_t cols() const override;

  std::size_t ellWidth() const override;
  std::size_t overflowNnz() const override;

  const int *ellColIdx() const;
  const float *ellValues() const;

  const int *overflowRowPtr() const;
  const int *overflowColIdx() const;
  const float *overflowValues() const;

private:
  std::size_t nRows_;
  std::size_t nCols_;
  std::size_t ellWidth_;
  std::size_t overflowNnz_;

  std::vector<int> ellColIdx_;
  std::vector<float> ellValues_;

  std::vector<int> overflowRowPtr_;
  std::vector<int> overflowColIdx_;
  std::vector<float> overflowValues_;
};

} // namespace gpuSolver
