#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gpuSolver {
namespace cuda {

void launchSetZero(
    float *x,
    std::size_t size,
    cudaStream_t stream);

void launchCopy(
    const float *source,
    float *destination,
    std::size_t size,
    cudaStream_t stream);

void launchScale(
    float *x,
    float alpha,
    std::size_t size,
    cudaStream_t stream);

void launchAxpy(
    float *y,
    const float *x,
    float alpha,
    std::size_t size,
    cudaStream_t stream);

void launchScaleVector(
    const float *scaleVector,
    float *output,
    std::size_t size,
    cudaStream_t stream);
} // namespace cuda
} // namespace gpuSolver
