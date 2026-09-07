#pragma once

#include <GPUSolver/DeviceELLMatrix.h>
#include <GPUSolver/DeviceScalar.h>
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

  void encodeSpmvELL(MTL::ComputeCommandEncoder *encoder,
                     const DeviceELLMatrix &A, const DeviceVector &x,
                     DeviceVector &Ax);

  void encodeDot(MTL::ComputeCommandEncoder *encoder, const DeviceVector &x,
                 const DeviceVector &y, DeviceScalar &result);

  void encodeScale(MTL::ComputeCommandEncoder *encoder, DeviceVector &x,
                   const float scalar);

  void encodeScale(MTL::ComputeCommandEncoder *encoder, DeviceVector &x,
                   const DeviceScalar &scalar);

  void encodeAxpy(MTL::ComputeCommandEncoder *encoder,
                  const DeviceScalar &alpha, const DeviceVector &x,
                  DeviceVector &y);
  void encodeAxpy(MTL::ComputeCommandEncoder *encoder, float alpha,
                  const DeviceVector &x, DeviceVector &y);

  void encodeScalarSet(MTL::ComputeCommandEncoder *encoder, float input,
                       DeviceScalar &output);

  void encodeScalarCopy(MTL::ComputeCommandEncoder *encoder,
                        const DeviceScalar &input, DeviceScalar &output);

  void encodeScalarDivide(MTL::ComputeCommandEncoder *encoder,
                          const DeviceScalar &numerator,
                          const DeviceScalar &denominator,
                          DeviceScalar &result);

  void encodeScalarMultiply(MTL::ComputeCommandEncoder *encoder,
                            const DeviceScalar &factor1,
                            const DeviceScalar &factor2, DeviceScalar &result);

  void encodeScalarNegate(MTL::ComputeCommandEncoder *encoder,
                          const DeviceScalar &input, DeviceScalar &output);

  void encodeScalarSqrt(MTL::ComputeCommandEncoder *encoder,
                        const DeviceScalar &input, DeviceScalar &output);

  // void encodeScale(MTL::ComputeCommandEncoder* encoder, const DeviceScalar& )
  void ensureReductionScratchCapacity(std::size_t requiredCapacity);

public:
  explicit MetalBackend(MetalContext &context);

  ~MetalBackend();

  void scale(DeviceVector &x, float scalar);

  void scale(DeviceVector &x, const DeviceScalar &scalar);

  void axpy(float alpha, const DeviceVector &x, DeviceVector &y);

  void axpy(const DeviceScalar &alpha, const DeviceVector &x, DeviceVector &y);

  void spmv(const DeviceCSRMatrix &A, const DeviceVector &x, DeviceVector &Ax);

  void spmvRepeated(const DeviceCSRMatrix &A, const DeviceVector &x,
                    DeviceVector &Ax, std::size_t repetitions);

  void spmv(const DeviceELLMatrix &A, const DeviceVector &x, DeviceVector &y);

  void spmvRepeated(const DeviceELLMatrix &A, const DeviceVector &x,
                    DeviceVector &y, int repetitions);

  void dot(const DeviceVector &x, const DeviceVector &y, DeviceScalar &result);

  void scalarSet(float value, DeviceScalar &result);

  void scalarCopy(const DeviceScalar &input, DeviceScalar &output);

  void scalarDivide(const DeviceScalar &numerator,
                    const DeviceScalar &denominator, DeviceScalar &result);

  void scalarMultiply(const DeviceScalar &factor1, const DeviceScalar &factor2,
                      DeviceScalar &result);

  void scalarNegate(const DeviceScalar &input, DeviceScalar &output);

  void scalarSqrt(const DeviceScalar &input, DeviceScalar &output);
};
} // namespace gpuSolver
