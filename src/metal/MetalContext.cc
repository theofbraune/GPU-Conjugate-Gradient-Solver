#include <Foundation/NSError.hpp>
#include <Foundation/NSString.hpp>
#include <Foundation/NSTypes.hpp>
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
#include <iterator>
#include <stdexcept>
#include <string>

namespace gpuSolver {

struct MetalContext::Impl {

  MTL::Device *device = nullptr;

  // MTL::Buffer* buffer = nullptr;

  MTL::CommandQueue *commandQueue = nullptr;

  MTL::Library *library = nullptr;

  MTL::Function *timesTwoFunction = nullptr;

  MTL::ComputePipelineState *timesTwoPipeline = nullptr;

  // MTL::CommandBuffer* commandBuffer = nullptr;

  // MTL::ComputeCommandEncoder* encoder = nullptr;
};

MetalContext::MetalContext() : impl_(new Impl) {
  impl_->device = MTL::CreateSystemDefaultDevice();

  if (!impl_->device)
    throw std::runtime_error("Could not load device");

  impl_->commandQueue = impl_->device->newCommandQueue();

  if (!impl_->commandQueue) {
    throw std::runtime_error("Could not create metal quuee");
  }

  NS::Error *error = nullptr;

  NS::String *libraryPath =
      NS::String::string(GPUSOLVER_METAL_LIBRARY, NS::UTF8StringEncoding);

  impl_->library = impl_->device->newLibrary(libraryPath, &error);

  if (!impl_->library) {
    std::string message = "Could not load Metal library.";

    if (error) {
      message += " ";
      message += error->localizedDescription()->utf8String();
    }

    throw std::runtime_error(message);
  }

  impl_->timesTwoFunction = impl_->library->newFunction(
      NS::String::string("timesTwo", NS::UTF8StringEncoding));

  if (!impl_->timesTwoFunction) {
    throw std::runtime_error("Could not find the twimes two implementation");
  }

  impl_->timesTwoPipeline =
      impl_->device->newComputePipelineState(impl_->timesTwoFunction, &error);

  if (!impl_->timesTwoPipeline)
    throw std::runtime_error("Could not create timesTwo compute pipeline.");
}

MetalContext::~MetalContext() {
  if(impl_->timesTwoPipeline){
    impl_->timesTwoPipeline->release();
  }

  if(impl_->timesTwoFunction){
    impl_->timesTwoFunction->release();
  }

  if(impl_->library){
    impl_->library->release();
  }

  if(impl_->commandQueue){
    impl_->commandQueue->release();
  }

  if(impl_->device){
    impl_->device->release();
  }

  delete impl_; 

}


void MetalContext::timesTwo(float* values, std::size_t size){

  const std::size_t bytes = size * sizeof(float);

  MTL::Buffer *buffer = this->impl_->device->newBuffer(values, bytes, MTL::ResourceStorageModeShared);


  MTL::CommandBuffer* commandBuffer = impl_->commandQueue->commandBuffer();

  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->timesTwoPipeline);

  encoder->setBuffer(buffer,0,0);

  MTL::Size gridSize = MTL::Size(size,1,1);

  NS::UInteger groupSize = impl_->timesTwoPipeline->maxTotalThreadsPerThreadgroup();

  // make sure that the grid has actually the right size here
  if(groupSize>size){
    groupSize = size;
  }

  MTL::Size threadsPerGroup = MTL::Size(groupSize,1,1);

  encoder->dispatchThreads(gridSize,threadsPerGroup);

  commandBuffer->commit();

  commandBuffer->waitUntilCompleted();

  float* result = static_cast<float*>(buffer->contents());

  for(std::size_t i=0; i<size; i++){
    values[i] = result[i];
  }

  buffer->release();

  
}

} // namespace gpusolver
