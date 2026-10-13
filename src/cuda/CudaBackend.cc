#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaEncoder.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <GPUSolver/cuda/CudaKernels.h>
#include <GPUSolver/cuda/VectorKernels.h>
#include "GPUSolver/BackendEncoder.h"
#include "GPUSolver/DeviceVector.h"
#include "GPUSolver/cuda/CudaDeviceVector.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

namespace gpuSolver {

CudaBackend::CudaBackend()
{
    int deviceCount = 0;

    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    if (deviceCount <= 0) {
        throw std::runtime_error("No CUDA device found.");
    }
}

CudaBackend::~CudaBackend() = default;

void CudaBackend::timesTwo(const std::vector<float>& input, std::vector<float>& output) const
{
    if (input.empty()) {
        output.clear();
        return;
    }

    output.resize(input.size());

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;

    const std::size_t bytes = input.size() * sizeof(float);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), bytes), "cudaMalloc input failed");

    try {
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&deviceOutput), bytes),
            "cudaMalloc output failed"
        );

        checkCuda(
            cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device failed"
        );

        cuda::launchTimesTwo(deviceInput, deviceOutput, input.size());

        checkCuda(cudaDeviceSynchronize(), "CUDA kernel execution failed");

        checkCuda(
            cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host failed"
        );

        cudaFree(deviceOutput);
        cudaFree(deviceInput);

    } catch (...) {
        if (deviceOutput != nullptr) {
            cudaFree(deviceOutput);
        }

        if (deviceInput != nullptr) {
            cudaFree(deviceInput);
        }

        throw;
    }
}

BackendEncoder* CudaBackend::createEncoder()
{
    return new CudaEncoder();
}

void CudaBackend::submitAndWait(BackendEncoder& encoder)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    cuda::check(cudaStreamSynchronize(cudaEncoder.stream()), "CUDA stream execution failed");
}

void CudaBackend::submit(BackendEncoder& encoder)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    cuda::check(cudaStreamSynchronize(cudaEncoder.stream()), "CUDA stream execution failed");
}

DeviceVector* CudaBackend::createVector(std::size_t size)
{
    return new CudaDeviceVector(size);
}

DeviceVector* CudaBackend::createVector(std::size_t size, const float* values)
{
    DeviceVector* devVec = new CudaDeviceVector(size);
    devVec->updateValues(values, size);
    return devVec;
}

DeviceIndexVector* CudaBackend::createIndexVector(std::size_t size)
{
    throw std::runtime_error("not implemented yet");
}

DeviceIndexVector* CudaBackend::createIndexVector(std::size_t size, const int* values)
{
    throw std::runtime_error("not implemented yet");
}

DeviceScalar* CudaBackend::createScalar()
{
    throw std::runtime_error("not implemented yet");
}

DeviceScalar* CudaBackend::createScalar(float value)
{
    throw std::runtime_error("not implemented yet");
}

DeviceCSRMatrix* CudaBackend::createCSRMatrix(const HostCSRMatrix& matrix)
{
    throw std::runtime_error("not implemented yet");
}

DeviceELLMatrix* CudaBackend::createELLMatrix(const HostELLMatrix& matrix)
{
    throw std::runtime_error("not implemented yet");
}

//-------------------------------------------------------------
//-------------------------------------------------------------
//-------------------------------------------------------------

void CudaBackend::encodeScale(BackendEncoder& encoder, DeviceVector& x, float alpha)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceVector& cudaX = static_cast<CudaDeviceVector&>(x);

    std::size_t size = x.size();

    cuda::launchScale(cudaX.data(), alpha, size, cudaEncoder.stream());
}

void CudaBackend::encodeScale(BackendEncoder& encoder, DeviceVector& x, const DeviceScalar& alpha)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeCopy(BackendEncoder& encoder, const DeviceVector& x, DeviceVector& xCopy)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);
    CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(xCopy);

    cuda::launchCopy(cudaX.data(), cudaY.data(), cudaX.size(), cudaEncoder.stream());
}

void CudaBackend::encodeScaleVector(
    BackendEncoder& encoder,
    const DeviceVector& scaleVector,
    DeviceVector& output
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeSetZero(BackendEncoder& encoder, DeviceVector& input)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceVector& cudaIn = static_cast<CudaDeviceVector&>(input);

    cuda::launchSetZero(cudaIn.data(), cudaIn.size(), cudaEncoder.stream());
}

void CudaBackend::encodeAxpy(
    BackendEncoder& encoder,
    float alpha,
    const DeviceVector& x,
    DeviceVector& y
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);
    CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(y);

    cuda::launchAxpy(cudaY.data(), cudaX.data(), alpha, cudaX.size(), cudaEncoder.stream());
}


void CudaBackend::encodeAxpy(
    BackendEncoder& encoder,
    const DeviceScalar& alpha,
    const DeviceVector& x,
    DeviceVector& y
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeSpmv(
    BackendEncoder& encoder,
    const DeviceCSRMatrix& A,
    const DeviceVector& x,
    DeviceVector& y
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeGaussSeidelColor(
    BackendEncoder& encoder,
    const DeviceCSRMatrix& matrix,
    const DeviceIndexVector& colorVertices,
    std::size_t colorStart,
    std::size_t colorCount,
    const DeviceVector& rhs,
    DeviceVector& solution,
    float omega
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeDot(
    BackendEncoder& encoder,
    const DeviceVector& x,
    const DeviceVector& y,
    DeviceScalar& result
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeScalarDivide(
    BackendEncoder& encoder,
    const DeviceScalar& numerator,
    const DeviceScalar& denominator,
    DeviceScalar& result
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeScalarMultiply(
    BackendEncoder& encoder,
    const DeviceScalar& factor1,
    const DeviceScalar& factor2,
    DeviceScalar& result
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeScalarSet(BackendEncoder& encoder, float value, DeviceScalar& output)
{
    throw std::runtime_error("not implemented yet");
}
void CudaBackend::encodeScalarCopy(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& output
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeScalarNegate(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& output
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeScalarSqrt(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& output
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeApplyBlockInverses3x3(
    BackendEncoder& encoder,
    const DeviceVector& inverseDiagBlocks,
    const DeviceVector& residual,
    DeviceVector& correction
)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::encodeBlockGaussSeidelColor3x3(
    BackendEncoder& encoder,
    const DeviceCSRMatrix& matrix,
    const DeviceIndexVector& colorBlocks,
    const DeviceVector& diagonalInverses,
    std::size_t begin,
    std::size_t end,
    const DeviceVector& rhs,
    DeviceVector& solution,
    float omega
)
{
    throw std::runtime_error("not implemented yet");
}


void CudaBackend::scale(DeviceVector& x, float alpha)
{
    BackendEncoder* encoder = this->createEncoder();
    encodeScale(*encoder, x, alpha);
    delete encoder;
}

void CudaBackend::scale(DeviceVector& x, const DeviceScalar& alpha)
{
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scaleVectorByVector(const DeviceVector& scaleVector, DeviceVector& output){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::axpy(float alpha, const DeviceVector& x, DeviceVector& y) {

    BackendEncoder* encoder = this->createEncoder();
    encodeAxpy(*encoder,alpha,x,y);
    delete encoder;
}

void CudaBackend::axpy(const DeviceScalar& alpha, const DeviceVector& x, DeviceVector& y){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::spmv(const DeviceCSRMatrix& A, const DeviceVector& x, DeviceVector& y){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::dot(const DeviceVector& x, const DeviceVector& y, DeviceScalar& result){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarSet(float value, DeviceScalar& result){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarCopy(const DeviceScalar& input, DeviceScalar& output){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarDivide(
    const DeviceScalar& numerator,
    const DeviceScalar& denominator,
    DeviceScalar& result
) {

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarMultiply(
    const DeviceScalar& factor1,
    const DeviceScalar& factor2,
    DeviceScalar& result
) {

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarNegate(const DeviceScalar& input, DeviceScalar& output){

    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarSqrt(const DeviceScalar& input, DeviceScalar& output){

    throw std::runtime_error("not implemented yet");
}


} // namespace gpuSolver
