#pragma once

#include <cstddef>
#include <vector>
#include <GPUSolver/DeviceIndexVector.h>

using std::size_t;


namespace MTL
{
    class Buffer;
}

namespace gpuSolver {
  
  class MetalContext;
  class MetalBackend;

  class MetalDeviceIndexVector: public DeviceIndexVector{ 

    private:

      struct Impl;
      Impl* impl_;

      MTL::Buffer* getNativeBuffer();

      MTL::Buffer* getNativeBuffer() const;

      friend class MetalBackend;

    public: 
      ~MetalDeviceIndexVector() override;
  
      MetalDeviceIndexVector(MetalContext& context, std::size_t size);
      MetalDeviceIndexVector(MetalContext& context, const std::vector<int>&values);
      MetalDeviceIndexVector(MetalContext& context, std::size_t size, const int* values);

      MetalDeviceIndexVector(const MetalDeviceIndexVector&) = delete;
      MetalDeviceIndexVector& operator=(const MetalDeviceIndexVector&) = delete;

      size_t size() const override;

      size_t size() override;

      void updateValues(const int* values, const size_t sizeOfValues) override;

      int* download() const override;
  };
}
