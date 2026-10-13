#pragma once

#include <GPUSolver/DeviceIndexVector.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

// Same "device memory == host memory" idea as CPUDeviceVector, just
// for the int-valued index vectors (used for the graph-coloring
// vertex/block order).
class CPUDeviceIndexVector : public DeviceIndexVector {
public:
  explicit CPUDeviceIndexVector(std::size_t size);

  CPUDeviceIndexVector(std::size_t size, const int *values);

  ~CPUDeviceIndexVector() override = default;

  CPUDeviceIndexVector(const CPUDeviceIndexVector &) = delete;
  CPUDeviceIndexVector &operator=(const CPUDeviceIndexVector &) = delete;

  std::size_t size() const override;
  std::size_t size() override;

  void updateValues(const int *values, std::size_t sizeOfValues) override;

  int *download() const override;

  int *data();
  const int *data() const;

private:
  mutable std::vector<int> data_;
};

} // namespace gpuSolver
