#pragma once

#include <cstddef>

namespace MTL {
class Buffer;
}

namespace gpuSolver {

class MetalContext;
class MetalBackend;
class HostELLMatrix;

class DeviceELLMatrix {

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
    DeviceELLMatrix(
        MetalContext& context,
        const HostELLMatrix& matrix
    );

    ~DeviceELLMatrix();

    DeviceELLMatrix(const DeviceELLMatrix&) = delete;
    DeviceELLMatrix& operator=(const DeviceELLMatrix&) = delete;

    std::size_t rows() const;
    std::size_t cols() const;

    std::size_t ellWidth() const;
    std::size_t overflowNnz() const;
};

} // namespace gpuSolver
