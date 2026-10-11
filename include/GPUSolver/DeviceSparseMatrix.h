#pragma once
#include <cstddef>
namespace gpuSolver {

class DeviceCSRMatrix {
public:
  virtual ~DeviceCSRMatrix() = default;
  virtual std::size_t rows() const = 0;
  virtual std::size_t cols() const = 0;
  virtual std::size_t nnz() const = 0;

  virtual void updateValues(const float *values, std::size_t numberOfValues) = 0;

  DeviceCSRMatrix(const DeviceCSRMatrix &) = delete;
  DeviceCSRMatrix &operator=(const DeviceCSRMatrix &) = delete;
protected:
  DeviceCSRMatrix() = default;
};
} // namespace gpuSolver
