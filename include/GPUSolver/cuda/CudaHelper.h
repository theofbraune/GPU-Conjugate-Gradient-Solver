#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace gpuSolver {
namespace cuda {

inline void check(
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

} // namespace cuda
} // namespace gpuSolver
