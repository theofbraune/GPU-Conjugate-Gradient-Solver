#pragma once

#include <GPUSolver/Backend.h>

#include <cstddef>

namespace gpuSolver {

// Reference/benchmarking backend: implements the full Backend
// interface as plain, OpenMP-parallelized loops over host memory.
// No device/host distinction, no command buffers, no streams --
// every encodeXxx below does its work and returns, so submit()/
// submitAndWait() are no-ops and createEncoder() hands back an
// essentially empty CPUEncoder.
//
// Two things this backend is good for beyond "the CUDA machine has
// no GPU today": it is the natural correctness oracle to compare
// MetalBackend/CudaBackend numerical results against (same math,
// no GPU-specific precision/scheduling quirks), and it's what lets
// you plot GPU-vs-CPU speedup curves for the benchmarking you asked
// about.
class CPUBackend : public Backend {
public:
  CPUBackend() = default;
  ~CPUBackend() override = default;

  CPUBackend(const CPUBackend &) = delete;
  CPUBackend &operator=(const CPUBackend &) = delete;

  // ========================================================
  // Encoder management.
  // ========================================================

  BackendEncoder *createEncoder() override;
  void submit(BackendEncoder &encoder) override;
  void submitAndWait(BackendEncoder &encoder) override;

  // ========================================================
  // Object creation.
  // ========================================================

  DeviceVector *createVector(std::size_t size) override;
  DeviceVector *createVector(std::size_t size, const float *values) override;

  DeviceIndexVector *createIndexVector(std::size_t size) override;
  DeviceIndexVector *createIndexVector(std::size_t size,
                                       const int *values) override;

  DeviceScalar *createScalar() override;
  DeviceScalar *createScalar(float value) override;

  DeviceCSRMatrix *createCSRMatrix(const HostCSRMatrix &matrix) override;
  DeviceELLMatrix *createELLMatrix(const HostELLMatrix &matrix) override;

  // ========================================================
  // Low-level asynchronous encoding API.
  //
  // "Asynchronous" is the Backend contract's word for it -- on this
  // backend every one of these runs to completion before returning.
  // ========================================================

  void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                   float alpha) override;
  void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                   const DeviceScalar &alpha) override;

  void encodeCopy(BackendEncoder &encoder, const DeviceVector &x,
                  DeviceVector &xCopy) override;

  void encodeScaleVector(BackendEncoder &encoder,
                         const DeviceVector &scaleVector,
                         DeviceVector &output) override;

  void encodeSetZero(BackendEncoder &encoder, DeviceVector &input) override;

  void encodeAxpy(BackendEncoder &encoder, float alpha, const DeviceVector &x,
                  DeviceVector &y) override;
  void encodeAxpy(BackendEncoder &encoder, const DeviceScalar &alpha,
                  const DeviceVector &x, DeviceVector &y) override;

  void encodeSpmv(BackendEncoder &encoder, const DeviceCSRMatrix &A,
                  const DeviceVector &x, DeviceVector &y) override;

  void encodeGaussSeidelColor(BackendEncoder &encoder,
                              const DeviceCSRMatrix &matrix,
                              const DeviceIndexVector &colorVertices,
                              std::size_t colorStart, std::size_t colorCount,
                              const DeviceVector &rhs, DeviceVector &solution,
                              float omega) override;

  void encodeDot(BackendEncoder &encoder, const DeviceVector &x,
                 const DeviceVector &y, DeviceScalar &result) override;

  void encodeScalarDivide(BackendEncoder &encoder,
                          const DeviceScalar &numerator,
                          const DeviceScalar &denominator,
                          DeviceScalar &result) override;
  void encodeScalarMultiply(BackendEncoder &encoder,
                            const DeviceScalar &factor1,
                            const DeviceScalar &factor2,
                            DeviceScalar &result) override;
  void encodeScalarSet(BackendEncoder &encoder, float value,
                       DeviceScalar &output) override;
  void encodeScalarCopy(BackendEncoder &encoder, const DeviceScalar &input,
                       DeviceScalar &output) override;
  void encodeScalarNegate(BackendEncoder &encoder, const DeviceScalar &input,
                          DeviceScalar &output) override;
  void encodeScalarSqrt(BackendEncoder &encoder, const DeviceScalar &input,
                       DeviceScalar &output) override;

  void encodeApplyBlockInverses3x3(BackendEncoder &encoder,
                                   const DeviceVector &inverseDiagBlocks,
                                   const DeviceVector &residual,
                                   DeviceVector &correction) override;

  void encodeBlockGaussSeidelColor3x3(
      BackendEncoder &encoder, const DeviceCSRMatrix &matrix,
      const DeviceIndexVector &colorBlocks,
      const DeviceVector &diagonalInverses, std::size_t begin,
      std::size_t end, const DeviceVector &rhs, DeviceVector &solution,
      float omega) override;

  // ========================================================
  // High-level standalone API (own encoder, submit, and wait).
  // ========================================================

  void scale(DeviceVector &x, float alpha) override;
  void scale(DeviceVector &x, const DeviceScalar &alpha) override;

  void scaleVectorByVector(const DeviceVector &scaleVector,
                           DeviceVector &output) override;

  void axpy(float alpha, const DeviceVector &x, DeviceVector &y) override;
  void axpy(const DeviceScalar &alpha, const DeviceVector &x,
           DeviceVector &y) override;

  void spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
           DeviceVector &y) override;

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
};

} // namespace gpuSolver
