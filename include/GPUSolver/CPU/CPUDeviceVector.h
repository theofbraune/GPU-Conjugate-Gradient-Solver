#pragma once

#include <GPUSolver/DeviceVector.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

// CPUDeviceVector is the CPU backend's concrete DeviceVector.
//
// There is no separate "device memory" here: the vector IS a
// std::vector<float> living in ordinary host RAM. That makes
// upload/download trivial compared to CudaDeviceVector (no PCIe
// transfer) and even simpler than MetalDeviceVector (no buffer
// wrapper object at all) -- "device" and "host" are the same
// address space by construction, same as Metal's shared-storage
// buffers on Apple Silicon, just without needing a special
// storage mode to get there.
class CPUDeviceVector : public DeviceVector {
public:
  explicit CPUDeviceVector(std::size_t size);

  CPUDeviceVector(std::size_t size, const float *values);

  ~CPUDeviceVector() override = default;

  CPUDeviceVector(const CPUDeviceVector &) = delete;
  CPUDeviceVector &operator=(const CPUDeviceVector &) = delete;

  std::size_t size() const override;

  void updateValues(const float *values, std::size_t sizeOfValues) override;

  float *download() const override;

  // Direct access for the CPUBackend kernels -- avoids going through
  // download()/updateValues() for every internal operation.
  float *data();
  const float *data() const;

private:
  // mutable: download() is logically const (it doesn't change the
  // *value*), but we hand back a non-const pointer into our own
  // storage, matching DeviceVector::download()'s signature.
  mutable std::vector<float> data_;
};

} // namespace gpuSolver
