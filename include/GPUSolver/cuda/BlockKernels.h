#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gpuSolver {
namespace cuda {

void launchApplyBlockInverses3x3(
    const float *inverseDiagBlocks,
    const float *residual,
    float *correction,
    std::size_t numberOfBlocks,
    cudaStream_t stream);

void launchBlockGaussSeidelColor3x3(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorBlocks,
    const float *diagonalInverses,
    std::size_t begin,
    std::size_t end,
    const float *rhs,
    float *solution,
    float omega,
    cudaStream_t stream);

} // namespace cuda
} // namespace gpuSolver
