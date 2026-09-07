#include <GPUSolver/metal/MetalEncoder.h>

#include <GPUSolver/metal/MetalContext.h>

#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>

#include <stdexcept>

namespace gpuSolver
{

MetalEncoder::MetalEncoder(
    MetalContext& context)
    : commandBuffer_(nullptr),
      encoder_(nullptr)
{
    commandBuffer_ =
        context.queue()->commandBuffer();

    if (commandBuffer_ == nullptr)
    {
        throw std::runtime_error(
            "MetalEncoder: could not create command buffer."
        );
    }

    encoder_ =
        commandBuffer_->computeCommandEncoder();

    if (encoder_ == nullptr)
    {
        throw std::runtime_error(
            "MetalEncoder: could not create compute command encoder."
        );
    }
}


MetalEncoder::~MetalEncoder()
{
    // The command buffer and encoder returned by Metal here are
    // autoreleased objects. MetalEncoder does not own them through
    // new/retain, so we do not release them manually here.
}

} // namespace gpuSolver
