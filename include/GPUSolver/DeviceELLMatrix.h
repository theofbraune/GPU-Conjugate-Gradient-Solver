#pragma once

#include <cstddef>


namespace gpuSolver {

class DeviceELLMatrix {


public:

    virtual ~DeviceELLMatrix() = default;

    DeviceELLMatrix(const DeviceELLMatrix&) = delete;
    DeviceELLMatrix& operator=(const DeviceELLMatrix&) = delete;

    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;

    virtual std::size_t ellWidth() const = 0;
    virtual std::size_t overflowNnz() const = 0;
protected:
    DeviceELLMatrix() = default;
};

} // namespace gpuSolver
