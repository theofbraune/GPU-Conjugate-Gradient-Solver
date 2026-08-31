#pragma once

#include <cstddef>

namespace gpuSolver{

class MetalContext{
  public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    void timesTwo(float* values, std::size_t size);
  private:
    struct Impl;
    Impl* impl_;

};


}
