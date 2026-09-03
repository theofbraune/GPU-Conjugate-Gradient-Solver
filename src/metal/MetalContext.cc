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

  // MTL::Function *timesTwoFunction = nullptr;
  //
  // MTL::ComputePipelineState *timesTwoPipeline = nullptr;

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

}

MetalContext::~MetalContext() {

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



MTL::Device* MetalContext::device() const{
  return impl_->device;
}

MTL::CommandQueue* MetalContext::queue() const{
  return impl_->commandQueue;
}

MTL::Library* MetalContext::library() const{
  return impl_->library;
}

} // namespace gpusolver
