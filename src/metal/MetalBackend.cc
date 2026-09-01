#include "GPUSolver/DeviceVector.h"
#include <Foundation/NSError.hpp>
#include <Foundation/NSString.hpp>
#include <GPUSolver/metal/MetalBackend.h>

#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLTypes.hpp>




#include <cstring>
#include <iterator>
#include <stdexcept>





namespace gpuSolver {

struct MetalBackend::Impl {

  MetalContext &context;
  MTL::ComputePipelineState *scalePipeline = nullptr;
  MTL::ComputePipelineState *axpyPipeline = nullptr;

  explicit Impl(MetalContext &context_) : context(context_) {}
};

MTL::ComputePipelineState *MetalBackend::makePipeline(MTL::Device *device,
                                                      MTL::Library *library,
                                                      const char *name) {

  // chwck if the path is ligit for a proper kernel first

  MTL::Function *pipelineFunction =
      library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  if (!pipelineFunction) {
    throw std::runtime_error(
        std::string(
            " the name that you passed for the kernel is invalid. Can't find") +
        name);
  }

  // device->newComputePipelineState
  NS::Error *error = nullptr;

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(pipelineFunction, &error);

  pipelineFunction->release();

  if (!pipeline) {
    std::string message = "Could not create Metal pipeline for ";

    message += name;

    if (error) {
      message += ": ";
      message += error->localizedDescription()->utf8String();
    }

    throw std::runtime_error(message);
  }

  return pipeline;
}

void MetalBackend::dispatch1D(MTL::ComputeCommandEncoder *encoder,
                              MTL::ComputePipelineState *pipeline, size_t n) {

  NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

  if (threadGroupSize > n) {
    threadGroupSize = n;
  }

  MTL::Size gridSize(n, 1, 1);
  MTL::Size threadsPerThreadgroup(threadGroupSize, 1, 1);

  encoder->dispatchThreads(gridSize, threadsPerThreadgroup);
}

MetalBackend::MetalBackend(MetalContext &context) {

  impl_ = new Impl(context);

  impl_->axpyPipeline =
      makePipeline(context.device(), context.library(), "axpy");

  impl_->scalePipeline =
      makePipeline(context.device(), context.library(), "scale");
}

MetalBackend::~MetalBackend() {

  if (impl_->scalePipeline) {
    impl_->scalePipeline->release();
  }

  if (impl_->axpyPipeline) {
    impl_->axpyPipeline->release();
  }

  delete impl_;
}

void MetalBackend::scale(DeviceVector &x, float scalar) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->scalePipeline);

  MTL::Buffer *bufferForX = x.getBuffer();

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBytes(&scalar, sizeof(float), 1);

  size_t sizeOfTheVector = x.getSizeOfVector();

  dispatch1D(encoder, impl_->scalePipeline, sizeOfTheVector);

  encoder->endEncoding();
  
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::axpy(float alpha, const DeviceVector &x, DeviceVector &y) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->axpyPipeline);

  MTL::Buffer *bufferForX = x.getBuffer();

  MTL::Buffer *bufferForY = y.getBuffer();
  
  size_t sizeOfx = x.getSizeOfVector();
  size_t sizeOfy = y.getSizeOfVector();
  
  if(sizeOfx!=sizeOfy){
    throw std::runtime_error(" the two vectorsd tat you are passing dont have the same size");
  }

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBuffer(bufferForY, 0, 1);

  encoder->setBytes(&alpha, sizeof(float), 2);

  dispatch1D(encoder, impl_->axpyPipeline, sizeOfx);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();



}

} // namespace gpuSolver
