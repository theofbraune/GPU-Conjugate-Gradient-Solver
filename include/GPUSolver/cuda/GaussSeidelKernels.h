#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gpuSolver {
namespace cuda {

void launchGaussSeidelColor(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorVertices,
    std::size_t colorStart,
    std::size_t colorCount,
    const float *rhs,
    float *solution,
    float omega,
    cudaStream_t stream);

void launchBlockGaussSeidelColor3x3(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorBlocks,
    const float *inverseBlocks,
    const float *rhs,
    float *solution,
    std::size_t colorStart,
    std::size_t colorCount,
    float omega,
    cudaStream_t stream);

} // namespace cuda
} // namespace gpuSolver
