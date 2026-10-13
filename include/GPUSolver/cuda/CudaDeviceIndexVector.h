
#pragma once

#include <GPUSolver/DeviceIndexVector.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class CudaDeviceIndexVector : public DeviceIndexVector
{
public:
  explicit CudaDeviceIndexVector(
      std::size_t size);

  CudaDeviceIndexVector(
      const std::vector<int> &values);

  ~CudaDeviceIndexVector() override;

  CudaDeviceIndexVector(
      const CudaDeviceIndexVector &) = delete;

  CudaDeviceIndexVector &
  operator=(
      const CudaDeviceIndexVector &) = delete;

  std::size_t size() const override;
  std::size_t size() override;

  void upload(
      const int* values);

  void updateValues(const int* values, std::size_t sizeOfValues) override;

  int *download() const  override;

  int *data();

  const int *data() const;

private:
  int *data_;
  std::size_t size_;

  mutable std::vector<int> hostBuffer_;
};

} // namespace gpuSolver
