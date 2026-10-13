#pragma once

#include <GPUSolver/DeviceScalar.h>

namespace gpuSolver {

class CPUDeviceScalar : public DeviceScalar {
public:
  CPUDeviceScalar();
  explicit CPUDeviceScalar(float value);

  ~CPUDeviceScalar() override = default;

  CPUDeviceScalar(const CPUDeviceScalar &) = delete;
  CPUDeviceScalar &operator=(const CPUDeviceScalar &) = delete;

  float download() const override;

  float value() const;
  void setValue(float value);

private:
  float value_;
};

} // namespace gpuSolver
