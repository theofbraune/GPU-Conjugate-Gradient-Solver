#pragma once

#include <cstddef>
#include <vector>

using std::size_t;


namespace MTL
{
    class Buffer;
}

namespace gpuSolver {
  
  class MetalContext;
  class MetalBackend;

  class DeviceVector{

    private:

      struct Impl;
      Impl* impl_;

      MTL::Buffer* getNativeBuffer();

      MTL::Buffer* getNativeBuffer() const;

      friend class MetalBackend;

    public:
      ~DeviceVector();
  
      DeviceVector(MetalContext& context, std::size_t size);
      DeviceVector(MetalContext& context, const std::vector<float>&values);
      DeviceVector(MetalContext& context, std::size_t size, const float* values);

      DeviceVector(const DeviceVector&) = delete;
      DeviceVector& operator=(const DeviceVector&) = delete;

      size_t size() const;

      size_t size();

      void updateValues(const float* values, const size_t sizeOfValues);

      float* download() const;





  };
}
