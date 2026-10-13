#pragma once

#include <cstddef>

namespace gpuSolver {


class DeviceIndexVector {

public:
  virtual ~DeviceIndexVector() = default;

  DeviceIndexVector(const DeviceIndexVector &) = delete;
  DeviceIndexVector &operator=(const DeviceIndexVector &) = delete;

  virtual std::size_t size() const = 0;

  virtual std::size_t size() = 0;

  virtual void updateValues(const int *values, const size_t sizeOfValues) = 0;

  virtual int *download() const = 0;

protected:
  DeviceIndexVector() = default;
};
} // namespace gpuSolver
