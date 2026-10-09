#pragma once
#include <GPUSolver/DeviceVector.h>
#include <cstddef>
#include <vector>

namespace MTL { class Buffer; }

namespace gpuSolver {

class MetalContext;
class MetalBackend;

class MetalDeviceVector : public DeviceVector {
public:
  MetalDeviceVector(MetalContext& context, std::size_t size);
  MetalDeviceVector(MetalContext& context, const std::vector<float>& values);
  MetalDeviceVector(MetalContext& context, std::size_t size, const float* values);
  ~MetalDeviceVector() override;

  std::size_t size() const override;
  void updateValues(const float* values, std::size_t sizeOfValues) override;
  float* download() const override;

private:
  struct Impl;
  Impl* impl_;

  MTL::Buffer* getNativeBuffer();
  MTL::Buffer* getNativeBuffer() const;

  friend class MetalBackend;
};

} // namespace gpuSolver
