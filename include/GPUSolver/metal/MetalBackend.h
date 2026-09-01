#pragma once


#include <GPUSolver/metal/MetalContext.h>
#include <GPUSolver/DeviceVector.h>


namespace MTL
{
    class ComputeCommandEncoder;
    class ComputePipelineState;
    class Device;
    class Library;
}

namespace gpuSolver{
class MetalBackend{

  private:
    struct Impl;
    Impl* impl_;
 
    void dispatch1D(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline, size_t n);
    
    MTL::ComputePipelineState* makePipeline(MTL::Device* device, MTL::Library* library, const char* name);


  public:
    explicit MetalBackend(MetalContext& context);

    ~MetalBackend();

    void scale(DeviceVector& x, float scalar);

    void axpy(float alpha, const DeviceVector& x, DeviceVector& y);



};
}
