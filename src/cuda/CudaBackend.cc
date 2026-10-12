#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaKernels.h>
#include <GPUSolver/cuda/CudaEncoder.h>
#include "GPUSolver/BackendEncoder.h"
#include "GPUSolver/DeviceVector.h"
#include "GPUSolver/cuda/CudaDeviceVector.h"
#include <GPUSolver/cuda/CudaHelper.h>
#include <GPUSolver/cuda/VectorKernels.h>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace {

void checkCuda(
    cudaError_t error,
    const char *message)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(message)
        + ": "
        + cudaGetErrorString(error));
}
}

} // namespace

namespace gpuSolver {

CudaBackend::CudaBackend()
{
  int deviceCount = 0;

  checkCuda(
      cudaGetDeviceCount(&deviceCount),
      "cudaGetDeviceCount failed");

  if (deviceCount <= 0) {
    throw std::runtime_error(
        "No CUDA device found.");
  }
}

CudaBackend::~CudaBackend() = default;

void
CudaBackend::timesTwo(
    const std::vector<float> &input,
    std::vector<float> &output) const
{
  if (input.empty()) {
    output.clear();
    return;
  }

  output.resize(
      input.size());

  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;

  const std::size_t bytes =
      input.size()
      * sizeof(float);

  checkCuda(
      cudaMalloc(
          reinterpret_cast<void **>(&deviceInput),
          bytes),
      "cudaMalloc input failed");

  try {

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void **>(&deviceOutput),
            bytes),
        "cudaMalloc output failed");

    checkCuda(
        cudaMemcpy(
            deviceInput,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy host-to-device failed");

    cuda::launchTimesTwo(
        deviceInput,
        deviceOutput,
        input.size());

    checkCuda(
        cudaDeviceSynchronize(),
        "CUDA kernel execution failed");

    checkCuda(
        cudaMemcpy(
            output.data(),
            deviceOutput,
            bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy device-to-host failed");

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

BackendEncoder* CudaBackend::createEncoder(){
  return new CudaEncoder();
}

void CudaBackend::submitAndWait(
    BackendEncoder &encoder)
{
  CudaEncoder &cudaEncoder =
      static_cast<CudaEncoder &>(
          encoder);

  cuda::check(
      cudaStreamSynchronize(
          cudaEncoder.stream()),
      "CUDA stream execution failed");
}

void CudaBackend::submit(
    BackendEncoder &encoder)
{
  CudaEncoder &cudaEncoder =
      static_cast<CudaEncoder &>(
          encoder);

  cuda::check(
      cudaStreamSynchronize(
          cudaEncoder.stream()),
      "CUDA stream execution failed");
}

DeviceVector* CudaBackend::createVector(std::size_t size){
  return new CudaDeviceVector(size);
}

DeviceVector* CudaBackend::createVector(std::size_t size, const float* values){

  DeviceVector* devVec = new CudaDeviceVector(size);
  devVec->updateValues(values,size);
  return devVec;

}

void CudaBackend::encodeScale(BackendEncoder &encoder, DeviceVector &x, float alpha){


  CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

  CudaDeviceVector& cudaX = static_cast<CudaDeviceVector&>(x); 

  std::size_t size = x.size();

  cuda::launchScale(cudaX.data(), alpha, size, cudaEncoder.stream());


}

void CudaBackend::encodeScale(BackendEncoder &encoder, DeviceVector &x, const DeviceScalar &alpha){

  // TODO 
}

void CudaBackend::encodeCopy(BackendEncoder &encoder, const DeviceVector &x, DeviceVector &xCopy){

  CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

  const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x); 
  CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(xCopy);


  cuda::launchCopy(cudaX.data(), cudaY.data(),cudaX.size(), cudaEncoder.stream());

}

void CudaBackend::encodeScaleVector(BackendEncoder &encoder, const DeviceVector &scaleVector, DeviceVector &output){
  //TODO 
}

void CudaBackend::encodeSetZero(BackendEncoder &encoder, DeviceVector &input){

  CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

  CudaDeviceVector& cudaIn = static_cast<CudaDeviceVector&>(input);


  cuda::launchSetZero(cudaIn.data(),cudaIn.size(),cudaEncoder.stream());

}


void CudaBackend::encodeAxpy(BackendEncoder &encoder, float alpha, const DeviceVector &x, DeviceVector &y){


  CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

  const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);
  CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(y);

  cuda::launchAxpy(cudaY.data(),cudaX.data(),alpha,cudaX.size(),cudaEncoder.stream());
}




} // namespace gpuSolver
