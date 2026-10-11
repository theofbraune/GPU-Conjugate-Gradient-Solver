#pragma once


namespace gpuSolver {


class DeviceScalar {

private:

public:

  virtual ~DeviceScalar() = default;

  DeviceScalar(const DeviceScalar &) = delete;
  DeviceScalar &operator=(const DeviceScalar &) = delete;

  // Debug/testing only.
  // The GPU must have completed before calling this.
  virtual float download() const = 0;

protected:
  DeviceScalar() = default;
};

} // namespace gpuSolver
