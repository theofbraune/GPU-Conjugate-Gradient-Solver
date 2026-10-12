#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaKernels.h>

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

} // namespace gpuSolver
