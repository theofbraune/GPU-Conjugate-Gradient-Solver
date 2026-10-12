#include <GPUSolver/cuda/CudaKernels.h>

#include <cuda_runtime.h>

#include <cstddef>
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


__global__
void timesTwoKernel(
    const float *input,
    float *output,
    std::size_t size)
{
  const std::size_t index =
      static_cast<std::size_t>(
          blockIdx.x)
      * static_cast<std::size_t>(
          blockDim.x)
      + static_cast<std::size_t>(
          threadIdx.x);

  if (index < size) {
    output[index] =
        2.0f
        * input[index];
  }
}

} // namespace


namespace gpuSolver {
namespace cuda {

void launchTimesTwo(
    const float *input,
    float *output,
    std::size_t size)
{
  constexpr int threadsPerBlock =
      256;

  const int numberOfBlocks =
      static_cast<int>(
          (size + threadsPerBlock - 1)
          / threadsPerBlock);

  timesTwoKernel<<<
      numberOfBlocks,
      threadsPerBlock>>>(
          input,
          output,
          size);

  checkCuda(
      cudaGetLastError(),
      "timesTwoKernel launch failed");
}

} // namespace cuda
} // namespace gpuSolver
