#pragma once

#include <GPUSolver/BackendEncoder.h>

namespace MTL
{
class CommandBuffer;
class ComputeCommandEncoder;
}

namespace gpuSolver
{

class MetalContext;
class MetalBackend;

class MetalEncoder : public BackendEncoder
{
private:
    MTL::CommandBuffer* commandBuffer_;
    MTL::ComputeCommandEncoder* encoder_;

    friend class MetalBackend;

public:
    explicit MetalEncoder(
        MetalContext& context
    );

    ~MetalEncoder() override;

    MetalEncoder(
        const MetalEncoder&
    ) = delete;

    MetalEncoder& operator=(
        const MetalEncoder&
    ) = delete;
};

} // namespace gpuSolver
