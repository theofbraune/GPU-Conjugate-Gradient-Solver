#pragma once 
#include <GPUSolver/DeviceScalar.h>

namespace gpuSolver {
class CudaDeviceScalar : public DeviceScalar {
public:
  CudaDeviceScalar();
  explicit CudaDeviceScalar(float value);

  ~CudaDeviceScalar() override;

  void updateValue(float value);
  float download() const override;

  float *data();
  const float *data() const;

private:
  float *data_;
};
}
