#pragma once

#include <GPUSolver/DeviceSparseMatrix.h>

#include <cusparse.h>
#include <cstddef>

namespace gpuSolver {

class CudaDeviceCSRMatrix : public DeviceCSRMatrix
{
public:
    CudaDeviceCSRMatrix(
        std::size_t rows,
        std::size_t cols,
        std::size_t nnz,
        const int* rowPtr,
        const int* colIdx,
        const float* values
    );

    ~CudaDeviceCSRMatrix() override;

    CudaDeviceCSRMatrix(const CudaDeviceCSRMatrix&) = delete;

    CudaDeviceCSRMatrix& operator=(const CudaDeviceCSRMatrix&) = delete;

    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t nnz() const override;

    int* rowPtr();
    const int* rowPtr() const;

    int* colIdx();
    const int* colIdx() const;

    float* values();
    const float* values() const;

    cusparseSpMatDescr_t descriptor() const;

    void ensureSpmvBuffer(std::size_t requiredSize) const;

    void* spmvBuffer() const;

    std::size_t spmvBufferSize() const;

    void updateValues(const float *values, std::size_t numberOfValues) override;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t nnz_;

    int* rowPtr_;
    int* colIdx_;
    float* values_;

    cusparseSpMatDescr_t descriptor_;
    mutable void* spmvBuffer_;
    mutable std::size_t spmvBufferSize_;
};

} // namespace gpuSolver
