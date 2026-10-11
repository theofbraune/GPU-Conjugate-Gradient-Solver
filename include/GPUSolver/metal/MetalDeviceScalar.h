#pragma once

#include <GPUSolver/DeviceScalar.h>
#include <cstddef>

namespace MTL {
class Buffer;
}

namespace gpuSolver {

class MetalContext;
class MetalBackend;

class MetalDeviceScalar : public DeviceScalar {

private:
  struct Impl;
  Impl *impl_;

  MTL::Buffer *getNativeBuffer() const;

  friend class MetalBackend;

public:
  MetalDeviceScalar(MetalContext &context);
  MetalDeviceScalar(MetalContext &context, float value);

  ~MetalDeviceScalar() override;

  MetalDeviceScalar(const DeviceScalar &) = delete;
  MetalDeviceScalar &operator=(const DeviceScalar &) = delete;

  // Debug/testing only.
  // The GPU must have completed before calling this.
  float download() const override;
};

} // namespace gpuSolver
