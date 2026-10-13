#include <GPUSolver/cuda/CudaDeviceScalar.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <cuda_runtime_api.h>


namespace gpuSolver {

CudaDeviceScalar::CudaDeviceScalar()
    : data_(nullptr)
{
    cuda::check(
        cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(float)),
        "cudaMalloc failed for CudaDeviceScalar"
    );
}

CudaDeviceScalar::CudaDeviceScalar(float value)
    : CudaDeviceScalar()
{

  this->updateValue(value);
}


CudaDeviceScalar::~CudaDeviceScalar()
{
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}


void CudaDeviceScalar::updateValue(
    float value)
{
  cuda::check(
      cudaMemcpy(
          data_,
          &value,
          sizeof(float),
          cudaMemcpyHostToDevice),
      "CudaDeviceScalar upload failed");
}


float
CudaDeviceScalar::download() const
{
  float value = 0.0f;

  cuda::check(
      cudaMemcpy(
          &value,
          data_,
          sizeof(float),
          cudaMemcpyDeviceToHost),
      "CudaDeviceScalar download failed");

  return value;
}


float *
CudaDeviceScalar::data()
{
  return data_;
}


const float *
CudaDeviceScalar::data() const
{
  return data_;
}

} // namespace gpuSolver
