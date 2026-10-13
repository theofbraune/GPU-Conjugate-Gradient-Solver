#pragma once

#include <GPUSolver/DeviceVector.h>
#include <cusparse.h>

#include <cstddef>
#include <vector>

namespace gpuSolver {

class CudaDeviceVector : public DeviceVector
{
public:
  explicit CudaDeviceVector(
      std::size_t size);

  CudaDeviceVector(
      const std::vector<float> &values);

  ~CudaDeviceVector() override;

  CudaDeviceVector(
      const CudaDeviceVector &) = delete;

  CudaDeviceVector &
  operator=(
      const CudaDeviceVector &) = delete;

  std::size_t size() const override;

  void upload(
      const float *values);

  void updateValues(const float* values, std::size_t sizeOfValues) override;

  float *download() const  override;

  float *data();

  const float *data() const;

  cusparseDnVecDescr_t descriptor() const;
private:
  float *data_;
  std::size_t size_;

  mutable std::vector<float> hostBuffer_;
  cusparseDnVecDescr_t descriptor_;
};

} // namespace gpuSolver
