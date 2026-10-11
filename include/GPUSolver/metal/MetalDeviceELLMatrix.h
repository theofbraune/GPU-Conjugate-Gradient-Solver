#pragma once

#include <GPUSolver/DeviceELLMatrix.h>

#include <cstddef>

namespace MTL {
class Buffer;
}

namespace gpuSolver {

class MetalContext;
class MetalBackend;
class HostELLMatrix;

class MetalDeviceELLMatrix : public DeviceELLMatrix {

private:
    struct Impl;
    Impl* impl_;

    MTL::Buffer* ellColIdxBuffer() const;
    MTL::Buffer* ellValuesBuffer() const;

    MTL::Buffer* overflowRowPtrBuffer() const;
    MTL::Buffer* overflowColIdxBuffer() const;
    MTL::Buffer* overflowValuesBuffer() const;

    friend class MetalBackend;

public:
    MetalDeviceELLMatrix(
        MetalContext& context,
        const HostELLMatrix& matrix
    );

    ~MetalDeviceELLMatrix() override;

    MetalDeviceELLMatrix(const DeviceELLMatrix&) = delete;
    MetalDeviceELLMatrix& operator=(const DeviceELLMatrix&) = delete;

    std::size_t rows() const override;
    std::size_t cols() const override;

    std::size_t ellWidth() const override;
    std::size_t overflowNnz() const override;
};

} // namespace gpuSolver
