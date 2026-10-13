#pragma once

#include <cuda_runtime_api.h>

namespace gpuSolver {
namespace cuda {

void launchScalarAdd(
    const float *a,
    const float *b,
    float *result,
    cudaStream_t stream);

void launchScalarMultiply(
    const float *a,
    const float *b,
    float *result,
    cudaStream_t stream);

void launchScalarDivide(
    const float *numerator,
    const float *denominator,
    float *result,
    cudaStream_t stream);

void launchScalarSqrt(
    const float *input,
    float *result,
    cudaStream_t stream);

void launchScalarSet(
    float value,
    float *output,
    cudaStream_t stream);

void launchScalarNegate(
    const float *input,
    float *output,
    cudaStream_t stream);

} // namespace cuda
} // namespace gpuSolver
