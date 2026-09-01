#pragma once

#include <cstddef>

namespace MTL{

  class Device;
  class CommandQueue;
  class Library;
}

namespace gpuSolver{

class MetalContext{
  public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MTL::Device* device() const;
    MTL::CommandQueue* queue() const;
    MTL::Library* library() const;

  private:
    struct Impl;
    Impl* impl_;

};


}
