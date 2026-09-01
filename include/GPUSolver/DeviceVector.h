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

      MTL::Buffer* getBuffer();

      MTL::Buffer* getBuffer() const;

      friend class MetalBackend;

    public:
      ~DeviceVector();
  
      DeviceVector(MetalContext& context, std::size_t size);
      DeviceVector(MetalContext& context, const std::vector<float>&values);

      DeviceVector(const DeviceVector&) = delete;
      DeviceVector& operator=(const DeviceVector&) = delete;

      size_t size() const;

      void upload(MetalContext& context, const float* values, const size_t sizeOfValues);

      float* download() const;


      size_t getSizeOfVector();

      size_t getSizeOfVector() const;



  };
}
