#include <GPUSolver/cuda/VectorKernels.h>

#include <GPUSolver/cuda/CudaHelper.h>

namespace {

constexpr int threadsPerBlock = 256;

int numberOfBlocks(std::size_t size)
{
    return static_cast<int>(
        (size + threadsPerBlock - 1)
        / threadsPerBlock);
}

__global__ void setZeroKernel(float* x, std::size_t size){

  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if(i < size){
    x[i] = 0;
  }
}

__global__ void copyKernel(const float* source, float* destination, std::size_t size){

  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  
  if(i < size){
    destination[i] = source[i];
  }

} 

__global__
void scaleKernel(float* source, float alpha, std::size_t size){


  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(i< size){
    source[i] = alpha*source[i];
  }
}

__global__ void axpyKernel(float* y, const float* x, float alpha, std::size_t size){

  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(i<size){
    y[i] = y[i] + alpha * x[i];
  }
}

}



namespace gpuSolver {
namespace cuda {

void launchCopy(
    const float *source,
    float *destination,
    std::size_t size,
    cudaStream_t stream)
{
  copyKernel<<<
      numberOfBlocks(size),
      threadsPerBlock,
      0,
      stream>>>(
          source,
          destination,
          size);

  check(
      cudaGetLastError(),
      "copyKernel launch failed");
}

void launchSetZero(
    float *x,
    std::size_t size,
    cudaStream_t stream){

  setZeroKernel<<<
    numberOfBlocks(size),
    threadsPerBlock,
    0,
    stream>>>(
        x,
        size
        );

  check(cudaGetLastError(), "set zero launch failed");

}


void launchScale(
    float *x,
    float alpha,
    std::size_t size,
    cudaStream_t stream){

  scaleKernel<<<
    numberOfBlocks(size),
  threadsPerBlock,
  0,
  stream>>>(
      x,
      alpha, 
      size
      );

  check(cudaGetLastError(), "scale launch failed");
}

void launchAxpy(
    float *y,
    const float *x,
    float alpha,
    std::size_t size,
    cudaStream_t stream){

  axpyKernel<<<
    numberOfBlocks(size),
  threadsPerBlock,
  0,
  stream>>>(y,x,alpha,size);

  check(cudaGetLastError(), "axpy launch failed");
}

} // namespace cuda
} // namespace gpuSolver
