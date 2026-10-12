#pragma once

#include <GPUSolver/BackendEncoder.h>
#include <cuda_runtime_api.h>


namespace gpuSolver {

class CudaEncoder final
    : public BackendEncoder
{
public:
  CudaEncoder();

  ~CudaEncoder() override;

  cudaStream_t stream() const;

private:
  cudaStream_t stream_;
};

} // namespace gpuSolver
