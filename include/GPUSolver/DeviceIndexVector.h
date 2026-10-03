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

  class DeviceIndexVector{

    private:

      struct Impl;
      Impl* impl_;

      MTL::Buffer* getNativeBuffer();

      MTL::Buffer* getNativeBuffer() const;

      friend class MetalBackend;

    public: 
      ~DeviceIndexVector();
  
      DeviceIndexVector(MetalContext& context, std::size_t size);
      DeviceIndexVector(MetalContext& context, const std::vector<int>&values);
      DeviceIndexVector(MetalContext& context, std::size_t size, const int* values);

      DeviceIndexVector(const DeviceIndexVector&) = delete;
      DeviceIndexVector& operator=(const DeviceIndexVector&) = delete;

      size_t size() const;

      size_t size();

      void updateValues(const int* values, const size_t sizeOfValues);

      int* download() const;





  };
}
