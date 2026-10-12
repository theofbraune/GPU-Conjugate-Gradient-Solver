#include <GPUSolver/cuda/CudaEncoder.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <cuda_runtime_api.h>

namespace gpuSolver {

CudaEncoder::CudaEncoder() : stream_(nullptr) {
  cuda::check(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
}

CudaEncoder::~CudaEncoder() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

cudaStream_t CudaEncoder::stream() const { return stream_; }

} // namespace gpuSolver
