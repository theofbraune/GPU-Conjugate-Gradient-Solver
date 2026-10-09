#pragma once
#include <cstddef>

namespace gpuSolver {

class DeviceVector {
public:
  virtual ~DeviceVector() = default;

  virtual std::size_t size() const = 0;
  virtual void updateValues(const float* values, std::size_t sizeOfValues) = 0;
  virtual float* download() const = 0;

protected:
  DeviceVector() = default;

public:
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;
};

} // namespace gpuSolver

