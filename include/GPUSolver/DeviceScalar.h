#pragma once

#include "metal/MetalContext.h"
#include <cstddef>

namespace MTL {
class Buffer;
}

namespace gpuSolver {

class MetalContext;
class MetalBackend;

class DeviceScalar {

private:
  struct Impl;
  Impl *impl_;

  MTL::Buffer *getNativeBuffer() const;

  friend class MetalBackend;

public:
  explicit DeviceScalar(MetalContext &context);
  DeviceScalar(MetalContext &context, float value);

  ~DeviceScalar();

  DeviceScalar(const DeviceScalar &) = delete;
  DeviceScalar &operator=(const DeviceScalar &) = delete;

  // Debug/testing only.
  // The GPU must have completed before calling this.
  float download() const;
};

} // namespace gpuSolver
