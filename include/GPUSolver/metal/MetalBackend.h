#pragma once

#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>

#include <GPUSolver/DeviceELLMatrix.h>
#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>

#include <GPUSolver/metal/MetalContext.h>

#include <cstddef>

namespace MTL {
class ComputeCommandEncoder;
class ComputePipelineState;
class Device;
class Library;
} // namespace MTL

namespace gpuSolver {

class MetalEncoder;

class MetalBackend : public Backend {
private:
  struct Impl;
  Impl *impl_;

  // --------------------------------------------------------
  // Metal-specific helpers.
  // --------------------------------------------------------

  void dispatch1D(MTL::ComputeCommandEncoder *encoder,
                  MTL::ComputePipelineState *pipeline, std::size_t n);

  MTL::ComputePipelineState *
  makePipeline(MTL::Device *device, MTL::Library *library, const char *name);

  // --------------------------------------------------------
  // Raw Metal encoding implementations.
  //
  // These operate directly on MTL::ComputeCommandEncoder.
  // The public generic encode methods below unwrap a
  // BackendEncoder into a MetalEncoder and call these.
  // --------------------------------------------------------

  void encodeSpmvMetal(MTL::ComputeCommandEncoder *encoder,
                       const DeviceCSRMatrix &A, const DeviceVector &x,
                       DeviceVector &Ax);

  void encodeSpmvELLMetal(MTL::ComputeCommandEncoder *encoder,
                          const DeviceELLMatrix &A, const DeviceVector &x,
                          DeviceVector &Ax);

  void encodeDotMetal(MTL::ComputeCommandEncoder *encoder,
                      const DeviceVector &x, const DeviceVector &y,
                      DeviceScalar &result);

  void encodeScaleMetal(MTL::ComputeCommandEncoder *encoder, DeviceVector &x,
                        float scalar);

  void encodeScaleMetal(MTL::ComputeCommandEncoder *encoder, DeviceVector &x,
                        const DeviceScalar &scalar);

  void encodeCopyMetal(MTL::ComputeCommandEncoder *encoder,
                       const DeviceVector &x, DeviceVector &xCopy);

  void encodeAxpyMetal(MTL::ComputeCommandEncoder *encoder, float alpha,
                       const DeviceVector &x, DeviceVector &y);

  void encodeAxpyMetal(MTL::ComputeCommandEncoder *encoder,
                       const DeviceScalar &alpha, const DeviceVector &x,
                       DeviceVector &y);

  void encodeScalarSetMetal(MTL::ComputeCommandEncoder *encoder, float value,
                            DeviceScalar &output);

  void encodeScalarCopyMetal(MTL::ComputeCommandEncoder *encoder,
                             const DeviceScalar &input, DeviceScalar &output);

  void encodeScalarDivideMetal(MTL::ComputeCommandEncoder *encoder,
                               const DeviceScalar &numerator,
                               const DeviceScalar &denominator,   
                               DeviceScalar &result);

  void encodeScalarMultiplyMetal(MTL::ComputeCommandEncoder *encoder,
                                 const DeviceScalar &factor1,
                                 const DeviceScalar &factor2,
                                 DeviceScalar &result);

  void encodeScalarNegateMetal(MTL::ComputeCommandEncoder *encoder,
                               const DeviceScalar &input, DeviceScalar &output);

  void encodeScalarSqrtMetal(MTL::ComputeCommandEncoder *encoder,
                             const DeviceScalar &input, DeviceScalar &output);

  void ensureReductionScratchCapacity(std::size_t requiredCapacity);

public:
  explicit MetalBackend(MetalContext &context);

  ~MetalBackend() override;

  MetalBackend(const MetalBackend &) = delete;

  MetalBackend &operator=(const MetalBackend &) = delete;

  // ========================================================
  // Encoder management.
  // ========================================================

  BackendEncoder *createEncoder() override;

  void submit(BackendEncoder &encoder) override;

  void submitAndWait(BackendEncoder &encoder) override;

  // ========================================================
  // Generic asynchronous encoding API.
  //
  // These add work to an already-existing encoder.
  // They do NOT submit and do NOT synchronize.
  // ========================================================

  void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                   float scalar) override;

  void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                   const DeviceScalar &scalar) override;

  void encodeCopy(BackendEncoder &encoder, const DeviceVector &x,
                  DeviceVector &xCopy) override;

  void encodeAxpy(BackendEncoder &encoder, float alpha, const DeviceVector &x,
                  DeviceVector &y) override;

  void encodeAxpy(BackendEncoder &encoder, const DeviceScalar &alpha,
                  const DeviceVector &x, DeviceVector &y) override;

  void encodeSpmv(BackendEncoder &encoder, const DeviceCSRMatrix &A,
                  const DeviceVector &x, DeviceVector &Ax) override;

  void encodeDot(BackendEncoder &encoder, const DeviceVector &x,
                 const DeviceVector &y, DeviceScalar &result) override;

  void encodeScalarSet(BackendEncoder &encoder, float value,
                       DeviceScalar &output) override;

  void encodeScalarCopy(BackendEncoder &encoder, const DeviceScalar &input,
                        DeviceScalar &output) override;

  void encodeScalarDivide(BackendEncoder &encoder,
                          const DeviceScalar &numerator,
                          const DeviceScalar &denominator,
                          DeviceScalar &result) override;

  void encodeScalarMultiply(BackendEncoder &encoder,
                            const DeviceScalar &factor1,
                            const DeviceScalar &factor2,
                            DeviceScalar &result) override;

  void encodeScalarNegate(BackendEncoder &encoder, const DeviceScalar &input,
                          DeviceScalar &output) override;

  void encodeScalarSqrt(BackendEncoder &encoder, const DeviceScalar &input,
                        DeviceScalar &output) override;

  // --------------------------------------------------------
  // Metal-specific ELL encoding.
  //
  // Not part of the generic Backend interface for now.
  // --------------------------------------------------------

  void encodeSpmvELL(BackendEncoder &encoder, const DeviceELLMatrix &A,
                     const DeviceVector &x, DeviceVector &Ax);

  // ========================================================
  // Standalone synchronous convenience API.
  //
  // These create an encoder internally, encode one operation,
  // submit it, and wait for completion.
  //
  // Useful for tests, debugging, benchmarks, etc.
  // The CG solver should use the encode... methods instead.
  // ========================================================

  void scale(DeviceVector &x, float scalar) override;

  void scale(DeviceVector &x, const DeviceScalar &scalar) override;

  void axpy(float alpha, const DeviceVector &x, DeviceVector &y) override;

  void axpy(const DeviceScalar &alpha, const DeviceVector &x,
            DeviceVector &y) override;

  void spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
            DeviceVector &Ax) override;

  void dot(const DeviceVector &x, const DeviceVector &y,
           DeviceScalar &result) override;

  void scalarSet(float value, DeviceScalar &result) override;

  void scalarCopy(const DeviceScalar &input, DeviceScalar &output) override;

  void scalarDivide(const DeviceScalar &numerator,
                    const DeviceScalar &denominator,
                    DeviceScalar &result) override;

  void scalarMultiply(const DeviceScalar &factor1, const DeviceScalar &factor2,
                      DeviceScalar &result) override;

  void scalarNegate(const DeviceScalar &input, DeviceScalar &output) override;

  void scalarSqrt(const DeviceScalar &input, DeviceScalar &output) override;

  // --------------------------------------------------------
  // Metal-specific ELL convenience/benchmark functions.
  // --------------------------------------------------------

  void spmv(const DeviceELLMatrix &A, const DeviceVector &x, DeviceVector &Ax);

  void spmvRepeated(const DeviceCSRMatrix &A, const DeviceVector &x,
                    DeviceVector &Ax, std::size_t repetitions);

  void spmvRepeated(const DeviceELLMatrix &A, const DeviceVector &x,
                    DeviceVector &Ax, std::size_t repetitions);


  DeviceCSRMatrix *createCSRMatrix(const HostCSRMatrix &matrix) override;

  DeviceVector *createVector(std::size_t size) override;

  DeviceVector *createVector(std::size_t size, const float* values) override;

  DeviceScalar *createScalar() override;

  DeviceScalar *createScalar(float value) override;

};

} // namespace gpuSolver
