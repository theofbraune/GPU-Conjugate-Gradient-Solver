#include <GPUSolver/cuda/ScalarKernels.h>

#include <GPUSolver/cuda/CudaHelper.h>

#include <cmath>

namespace {

__global__
void scalarAddKernel(
    const float *a,
    const float *b,
    float *result)
{
  *result = *a + *b;
}


__global__
void scalarMultiplyKernel(
    const float *a,
    const float *b,
    float *result)
{
  *result = (*a) * (*b);
}


__global__
void scalarDivideKernel(
    const float *numerator,
    const float *denominator,
    float *result)
{
  *result = (*numerator) / (*denominator);
}


__global__
void scalarSqrtKernel(
    const float *input,
    float *result)
{
  *result = sqrtf(*input);
}

__global__
void scalarSetKernel(
    float value,
    float *output)
{
  *output = value;
}


__global__
void scalarNegateKernel(
    const float *input,
    float *output)
{
  *output = -(*input);
}


} // namespace

namespace gpuSolver {
namespace cuda {

void launchScalarAdd(
    const float *a,
    const float *b,
    float *result,
    cudaStream_t stream)
{
  scalarAddKernel<<<1, 1, 0, stream>>>(
      a,
      b,
      result);

  cuda::check(
      cudaGetLastError(),
      "scalarAddKernel launch failed");
}


void launchScalarMultiply(
    const float *a,
    const float *b,
    float *result,
    cudaStream_t stream)
{
  scalarMultiplyKernel<<<1, 1, 0, stream>>>(
      a,
      b,
      result);

  cuda::check(
      cudaGetLastError(),
      "scalarMultiplyKernel launch failed");
}


void launchScalarDivide(
    const float *numerator,
    const float *denominator,
    float *result,
    cudaStream_t stream)
{
  scalarDivideKernel<<<1, 1, 0, stream>>>(
      numerator,
      denominator,
      result);

  cuda::check(
      cudaGetLastError(),
      "scalarDivideKernel launch failed");
}


void launchScalarSqrt(
    const float *input,
    float *result,
    cudaStream_t stream)
{
  scalarSqrtKernel<<<1, 1, 0, stream>>>(
      input,
      result);

  cuda::check(
      cudaGetLastError(),
      "scalarSqrtKernel launch failed");
}

void launchScalarSet(
    float value,
    float *output,
    cudaStream_t stream)
{
  scalarSetKernel<<<1, 1, 0, stream>>>(
      value,
      output);

  cuda::check(
      cudaGetLastError(),
      "scalarSetKernel launch failed");
}


void launchScalarNegate(
    const float *input,
    float *output,
    cudaStream_t stream)
{
  scalarNegateKernel<<<1, 1, 0, stream>>>(
      input,
      output);

  cuda::check(
      cudaGetLastError(),
      "scalarNegateKernel launch failed");
}

} // namespace cuda
} // namespace gpuSolver
