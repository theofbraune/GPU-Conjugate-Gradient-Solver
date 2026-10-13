#include <GPUSolver/cuda/CudaDeviceCSRMatrix.h>
#include <GPUSolver/cuda/CudaHelper.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>

namespace gpuSolver {

CudaDeviceCSRMatrix::CudaDeviceCSRMatrix(
    std::size_t rows,
    std::size_t cols,
    std::size_t nnz,
    const int* rowPtr,
    const int* colIdx,
    const float* values
)
    : rows_(rows)
    , cols_(cols)
    , nnz_(nnz)
    , rowPtr_(nullptr)
    , colIdx_(nullptr)
    , values_(nullptr)
    , descriptor_(nullptr)
    , spmvBuffer_(nullptr)
    , spmvBufferSize_(0)
{
    cuda::check(
        cudaMalloc(reinterpret_cast<void**>(&rowPtr_), (rows_ + 1) * sizeof(int)),
        "cudaMalloc failed for CSR rowPtr"
    );

    cuda::check(
        cudaMalloc(reinterpret_cast<void**>(&colIdx_), nnz_ * sizeof(int)),
        "cudaMalloc failed for CSR colIdx"
    );

    cuda::check(
        cudaMalloc(reinterpret_cast<void**>(&values_), nnz_ * sizeof(float)),
        "cudaMalloc failed for CSR values"
    );

    cuda::check(
        cudaMemcpy(rowPtr_, rowPtr, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice),
        "CSR rowPtr upload failed"
    );

    cuda::check(
        cudaMemcpy(colIdx_, colIdx, nnz_ * sizeof(int), cudaMemcpyHostToDevice),
        "CSR colIdx upload failed"
    );

    cuda::check(
        cudaMemcpy(values_, values, nnz_ * sizeof(float), cudaMemcpyHostToDevice),
        "CSR values upload failed"
    );

    const cusparseStatus_t status = cusparseCreateCsr(
        &descriptor_,
        static_cast<int64_t>(rows_),
        static_cast<int64_t>(cols_),
        static_cast<int64_t>(nnz_),
        rowPtr_,
        colIdx_,
        values_,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseCreateCsr failed");
    }
}

CudaDeviceCSRMatrix::~CudaDeviceCSRMatrix()
{
    if (spmvBuffer_ != nullptr) {
        cudaFree(spmvBuffer_);
    }
    if (descriptor_ != nullptr) {
        cusparseDestroySpMat(descriptor_);
    }

    if (values_ != nullptr) {
        cudaFree(values_);
    }

    if (colIdx_ != nullptr) {
        cudaFree(colIdx_);
    }

    if (rowPtr_ != nullptr) {
        cudaFree(rowPtr_);
    }
}

void CudaDeviceCSRMatrix::updateValues(const float* values, std::size_t numberOfValues)
{
    if (numberOfValues != nnz_) {
        throw std::runtime_error(
            "CudaDeviceCSRMatrix::updateValues: "
            "number of values does not match nnz."
        );
    }

    if (nnz_ == 0) {
        return;
    }

    cuda::check(
        cudaMemcpy(values_, values, nnz_ * sizeof(float), cudaMemcpyHostToDevice),
        "cudaDeviceCSRMatrix::updateValues failed"
    );
}

std::size_t CudaDeviceCSRMatrix::rows() const
{
    return rows_;
}

std::size_t CudaDeviceCSRMatrix::cols() const
{
    return cols_;
}

std::size_t CudaDeviceCSRMatrix::nnz() const
{
    return nnz_;
}

int* CudaDeviceCSRMatrix::rowPtr()
{
    return rowPtr_;
}

const int* CudaDeviceCSRMatrix::rowPtr() const
{
    return rowPtr_;
}

int* CudaDeviceCSRMatrix::colIdx()
{
    return colIdx_;
}

const int* CudaDeviceCSRMatrix::colIdx() const
{
    return colIdx_;
}

float* CudaDeviceCSRMatrix::values()
{
    return values_;
}

const float* CudaDeviceCSRMatrix::values() const
{
    return values_;
}

cusparseSpMatDescr_t CudaDeviceCSRMatrix::descriptor() const
{
    return descriptor_;
}
void CudaDeviceCSRMatrix::ensureSpmvBuffer(std::size_t requiredSize) const
{
    if (requiredSize <= spmvBufferSize_) {
        return;
    }

    void* newBuffer = nullptr;

    if (requiredSize > 0) {
        cuda::check(
            cudaMalloc(&newBuffer, requiredSize),
            "Failed to allocate cuSPARSE SpMV workspace"
        );
    }

    if (spmvBuffer_ != nullptr) {
        cudaFree(spmvBuffer_);
    }

    spmvBuffer_ = newBuffer;
    spmvBufferSize_ = requiredSize;
}


void* CudaDeviceCSRMatrix::spmvBuffer() const
{
    return spmvBuffer_;
}


std::size_t CudaDeviceCSRMatrix::spmvBufferSize() const
{
    return spmvBufferSize_;
}
} // namespace gpuSolver
