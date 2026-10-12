#include <GPUSolver/cuda/CudaDeviceVector.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <cstddef>
#include <cuda_runtime_api.h>

namespace gpuSolver {

CudaDeviceVector::CudaDeviceVector(std::size_t size) {
  this->size_ = size;
  this->data_ = nullptr;

  this->hostBuffer_.resize(this->size_);
  cuda::check(
      cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(float)),
      "cudaMalloc failed");
}
CudaDeviceVector::CudaDeviceVector(const std::vector<float> &values)
    : CudaDeviceVector(values.size()) {
  upload(values.data());
}

CudaDeviceVector::~CudaDeviceVector() {
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}

void CudaDeviceVector::upload(const float *values) {

  cuda::check(
      cudaMemcpy(data_, values, size_ * sizeof(float), cudaMemcpyHostToDevice),
      "CUDA vector download failed");
}

void CudaDeviceVector::updateValues(
    const float *values,
    std::size_t sizeOfValues) {

  // Same size: only replace the values.
  if (sizeOfValues == size_) {
    upload(values);
    return;
  }

  // Size changed: allocate a new GPU buffer.
  float *newData = nullptr;

  if (sizeOfValues > 0) {
    cuda::check(
        cudaMalloc(
            reinterpret_cast<void **>(&newData),
            sizeOfValues * sizeof(float)),
        "cudaMalloc failed while resizing CUDA vector");

    try {
      cuda::check(
          cudaMemcpy(
              newData,
              values,
              sizeOfValues * sizeof(float),
              cudaMemcpyHostToDevice),
          "CUDA vector update failed");
    } catch (...) {
      cudaFree(newData);
      throw;
    }
  }

  // Only destroy the old allocation after the new one succeeded.
  if (data_ != nullptr) {
    cudaFree(data_);
  }

  data_ = newData;
  size_ = sizeOfValues;

  hostBuffer_.resize(size_);
}

float *CudaDeviceVector::download() const {
  hostBuffer_.resize(size_);

  cuda::check(cudaMemcpy(hostBuffer_.data(), data_, size_ * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "CUDA vector download failed");

  return hostBuffer_.data();
}

std::size_t CudaDeviceVector::size() const { return this->size_; }

float *CudaDeviceVector::data() { return this->data_; }

const float *CudaDeviceVector::data() const { return this->data_; }

} // namespace gpuSolver
