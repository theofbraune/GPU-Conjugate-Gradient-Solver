
#include <GPUSolver/cuda/CudaDeviceIndexVector.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <cstddef>
#include <cuda_runtime_api.h>

namespace gpuSolver {

CudaDeviceIndexVector::CudaDeviceIndexVector(std::size_t size) {
  this->size_ = size;
  this->data_ = nullptr;

  this->hostBuffer_.resize(this->size_);
  cuda::check(
      cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(int)),
      "cudaMalloc failed");
}
CudaDeviceIndexVector::CudaDeviceIndexVector(const std::vector<int> &values)
    : CudaDeviceIndexVector(values.size()) {
  upload(values.data());
}

CudaDeviceIndexVector::~CudaDeviceIndexVector() {
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}

void CudaDeviceIndexVector::upload(const int *values) {

  cuda::check(
      cudaMemcpy(data_, values, size_ * sizeof(int), cudaMemcpyHostToDevice),
      "CUDA vector download failed");
}

void CudaDeviceIndexVector::updateValues(
    const int *values,
    std::size_t sizeOfValues) {

  // Same size: only replace the values.
  if (sizeOfValues == size_) {
    upload(values);
    return;
  }

  // Size changed: allocate a new GPU buffer.
  int *newData = nullptr;

  if (sizeOfValues > 0) {
    cuda::check(
        cudaMalloc(
            reinterpret_cast<void **>(&newData),
            sizeOfValues * sizeof(int)),
        "cudaMalloc failed while resizing CUDA vector");

    try {
      cuda::check(
          cudaMemcpy(
              newData,
              values,
              sizeOfValues * sizeof(int),
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

int *CudaDeviceIndexVector::download() const {
  hostBuffer_.resize(size_);

  cuda::check(cudaMemcpy(hostBuffer_.data(), data_, size_ * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "CUDA vector download failed");

  return hostBuffer_.data();
}

std::size_t CudaDeviceIndexVector::size() const { return this->size_; }
std::size_t CudaDeviceIndexVector::size() { return this->size_; }

int *CudaDeviceIndexVector::data() { return this->data_; }

const int *CudaDeviceIndexVector::data() const { return this->data_; }

} // namespace gpuSolver
