#pragma once
#include <cstddef>

namespace gpuSolver {

class BackendEncoder;
class DeviceVector;
class DeviceScalar;
class DeviceCSRMatrix;
class HostCSRMatrix;

class Backend {
public:
  virtual ~Backend() = default;

  virtual BackendEncoder *createEncoder() = 0;

  virtual void submit(BackendEncoder &encoder) = 0;

  virtual void submitAndWait(BackendEncoder &encoder) = 0;

  // handles to create objects
  virtual DeviceVector *createVector(std::size_t size) = 0;

  virtual DeviceVector *createVector(std::size_t size, const float* values) = 0;

  virtual DeviceScalar *createScalar() = 0;

  virtual DeviceScalar *createScalar(float value) = 0;

  virtual DeviceCSRMatrix *createCSRMatrix(const HostCSRMatrix &matrix) = 0;

  // --------------------------------------------------------
  // Low-level asynchronous encoding API.
  // --------------------------------------------------------

  virtual void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                           float alpha) = 0;

  virtual void encodeScale(BackendEncoder &encoder, DeviceVector &x,
                           const DeviceScalar &alpha) = 0;

  virtual void encodeCopy(BackendEncoder &encoder, const DeviceVector &x,
                          DeviceVector &xCopy) = 0;

  virtual void encodeAxpy(BackendEncoder &encoder, float alpha,
                          const DeviceVector &x, DeviceVector &y) = 0;

  virtual void encodeAxpy(BackendEncoder &encoder, const DeviceScalar &alpha,
                          const DeviceVector &x, DeviceVector &y) = 0;

  virtual void encodeSpmv(BackendEncoder &encoder, const DeviceCSRMatrix &A,
                          const DeviceVector &x, DeviceVector &y) = 0;

  virtual void encodeDot(BackendEncoder &encoder, const DeviceVector &x,
                         const DeviceVector &y, DeviceScalar &result) = 0;

  virtual void encodeScalarDivide(BackendEncoder &encoder,
                                  const DeviceScalar &numerator,
                                  const DeviceScalar &denominator,
                                  DeviceScalar &result) = 0;

  virtual void encodeScalarMultiply(BackendEncoder &encoder,
                                    const DeviceScalar &factor1,
                                    const DeviceScalar &factor2,
                                    DeviceScalar &result) = 0;

  virtual void encodeScalarSet(BackendEncoder &encoder, float value,
                               DeviceScalar &output) = 0;
  virtual void encodeScalarCopy(BackendEncoder &encoder,
                                const DeviceScalar &input,
                                DeviceScalar &output) = 0;

  virtual void encodeScalarNegate(BackendEncoder &encoder,
                                  const DeviceScalar &input,
                                  DeviceScalar &output) = 0;

  virtual void encodeScalarSqrt(BackendEncoder &encoder,
                                const DeviceScalar &input,
                                DeviceScalar &output) = 0;

  // --------------------------------------------------------
  // High-level standalone API.
  //
  // These create their own encoder, submit, and wait.
  // --------------------------------------------------------

  virtual void scale(DeviceVector &x, float alpha) = 0;

  virtual void scale(DeviceVector &x, const DeviceScalar &alpha) = 0;

  virtual void axpy(float alpha, const DeviceVector &x, DeviceVector &y) = 0;

  virtual void axpy(const DeviceScalar &alpha, const DeviceVector &x,
                    DeviceVector &y) = 0;

  virtual void spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
                    DeviceVector &y) = 0;

  virtual void dot(const DeviceVector &x, const DeviceVector &y,
                   DeviceScalar &result) = 0;

  virtual void scalarSet(float value, DeviceScalar &result) = 0;

  virtual void scalarCopy(const DeviceScalar &input, DeviceScalar &output) = 0;

  virtual void scalarDivide(const DeviceScalar &numerator,
                            const DeviceScalar &denominator,
                            DeviceScalar &result) = 0;

  virtual void scalarMultiply(const DeviceScalar &factor1,
                              const DeviceScalar &factor2,
                              DeviceScalar &result) = 0;

  virtual void scalarNegate(const DeviceScalar &input,
                            DeviceScalar &output) = 0;

  virtual void scalarSqrt(const DeviceScalar &input, DeviceScalar &output) = 0;

  // virtual void updateCSRValues(DeviceCSRMatrix &deviceMatrix,
  //                              const HostCSRMatrix &hostMatrix) = 0;
};

} // namespace gpuSolver
