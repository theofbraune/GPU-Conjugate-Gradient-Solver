#include <GPUSolver/cuda/GaussSeidelKernels.h>


#include <GPUSolver/cuda/CudaHelper.h>

namespace {

constexpr int threadsPerBlock = 256;

__global__
void gaussSeidelColorKernel(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorVertices,
    std::size_t colorStart,
    std::size_t colorCount,
    const float *rhs,
    float *solution,
    float omega)
{
  const std::size_t localIndex =
      static_cast<std::size_t>(blockIdx.x)
          * static_cast<std::size_t>(blockDim.x)
      + static_cast<std::size_t>(threadIdx.x);

  if (localIndex >= colorCount) {
    return;
  }

  const int row =
      colorVertices[colorStart + localIndex];

  float diagonal = 0.0f;
  float offDiagonalSum = 0.0f;

  const int begin = rowPtr[row];
  const int end = rowPtr[row + 1];

  for (int k = begin; k < end; ++k) {

    const int column =
        colIdx[k];

    const float value =
        values[k];

    if (column == row) {
      diagonal = value;
    } else {
      offDiagonalSum +=
          value * solution[column];
    }
  }

  const float gaussSeidelValue =
      (rhs[row] - offDiagonalSum)
      / diagonal;

  solution[row] =
      (1.0f - omega) * solution[row]
      + omega * gaussSeidelValue;
}

} // namespace

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
    cudaStream_t stream)
{
  if (colorCount == 0) {
    return;
  }

  const int numberOfBlocks =
      static_cast<int>(
          (colorCount + threadsPerBlock - 1)
          / threadsPerBlock);

  gaussSeidelColorKernel<<<
      numberOfBlocks,
      threadsPerBlock,
      0,
      stream>>>(
          rowPtr,
          colIdx,
          values,
          colorVertices,
          colorStart,
          colorCount,
          rhs,
          solution,
          omega);

  cuda::check(
      cudaGetLastError(),
      "gaussSeidelColorKernel launch failed");
}

} // namespace cuda
} // namespace gpuSolver
