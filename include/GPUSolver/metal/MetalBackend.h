#pragma once

#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalContext.h>

namespace MTL {
class ComputeCommandEncoder;
class ComputePipelineState;
class Device;
class Library;
} // namespace MTL

namespace gpuSolver {
class MetalBackend {

private:
  struct Impl;
  Impl *impl_;

  void dispatch1D(MTL::ComputeCommandEncoder *encoder,
                  MTL::ComputePipelineState *pipeline, size_t n);

  MTL::ComputePipelineState *
  makePipeline(MTL::Device *device, MTL::Library *library, const char *name);

  void encodeSpmv(MTL::ComputeCommandEncoder *encoder, const DeviceCSRMatrix &A,
                  const DeviceVector &x, DeviceVector &Ax);

public:
  explicit MetalBackend(MetalContext &context);

  ~MetalBackend();

  void scale(DeviceVector &x, float scalar);

  void axpy(float alpha, const DeviceVector &x, DeviceVector &y);

  void spmv(const DeviceCSRMatrix &A, const DeviceVector &x, DeviceVector &Ax);

  void spmvRepeated(const DeviceCSRMatrix &A, const DeviceVector &x,
                    DeviceVector &Ax, std::size_t repetitions);
};
} // namespace gpuSolver
