#include <GPUSolver/CPU/CPUBackend.h>

#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostELLMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>

#include <GPUSolver/CPU/CPUDeviceMatrix.h>
#include <GPUSolver/CPU/CPUDeviceELLMatrix.h>
#include <GPUSolver/CPU/CPUDeviceIndexVector.h>
#include <GPUSolver/CPU/CPUDeviceScalar.h>
#include <GPUSolver/CPU/CPUDeviceVector.h>
#include <GPUSolver/CPU/CPUEncoder.h>

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gpuSolver {

namespace {

// Thin, same-named-everywhere downcast helpers -- mirrors the
// static_cast<MetalDeviceVector&>(...) pattern from MetalBackend.
// static_cast (not dynamic_cast) is fine here precisely because
// only one backend is ever compiled into a given build
// (GPUSOLVER_SELECTED_BACKEND picks exactly one), so a CPUBackend
// object can structurally never be handed anything but CPU* device
// objects -- there's no cross-backend mixing this has to guard
// against at runtime.

CPUDeviceVector &CPUVec(DeviceVector &v) {
  return static_cast<CPUDeviceVector &>(v);
}
const CPUDeviceVector &CPUVec(const DeviceVector &v) {
  return static_cast<const CPUDeviceVector &>(v);
}

CPUDeviceScalar &CPUScalar(DeviceScalar &s) {
  return static_cast<CPUDeviceScalar &>(s);
}
const CPUDeviceScalar &CPUScalar(const DeviceScalar &s) {
  return static_cast<const CPUDeviceScalar &>(s);
}

const CPUDeviceCSRMatrix &CPUMatrix(const DeviceCSRMatrix &m) {
  return static_cast<const CPUDeviceCSRMatrix &>(m);
}

const CPUDeviceIndexVector &CPUIndexVec(const DeviceIndexVector &v) {
  return static_cast<const CPUDeviceIndexVector &>(v);
}

// One standalone CSR row's contribution to y = A*x, used both by
// the plain SpMV and, with a column exclusion, by the Gauss-Seidel
// sweeps below.
float rowDotExcluding(const CPUDeviceCSRMatrix &A, const float *x, int row,
                     int excludeColBegin, int excludeColEnd,
                     float *diagOut) {
  const int *rowPtr = A.rowPtr();
  const int *colIdx = A.colIdx();
  const float *values = A.values();

  float sum = 0.0f;
  if (diagOut != nullptr) {
    *diagOut = 0.0f;
  }

  for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {
    const int col = colIdx[entry];
    if (col == row && diagOut != nullptr) {
      *diagOut = values[entry];
      continue;
    }
    if (col >= excludeColBegin && col < excludeColEnd) {
      continue;
    }
    sum += values[entry] * x[col];
  }
  return sum;
}

} // namespace

// ============================================================
// Encoder management -- everything below already ran by the time
// encodeXxx returns, so there is nothing left for these to do.
// ============================================================

BackendEncoder *CPUBackend::createEncoder() { return new CPUEncoder(); }

void CPUBackend::submit(BackendEncoder &) {}

void CPUBackend::submitAndWait(BackendEncoder &) {}

// ============================================================
// Object creation.
// ============================================================

DeviceVector *CPUBackend::createVector(std::size_t size) {
  return new CPUDeviceVector(size);
}

DeviceVector *CPUBackend::createVector(std::size_t size,
                                       const float *values) {
  return new CPUDeviceVector(size, values);
}

DeviceIndexVector *CPUBackend::createIndexVector(std::size_t size) {
  return new CPUDeviceIndexVector(size);
}

DeviceIndexVector *CPUBackend::createIndexVector(std::size_t size,
                                                 const int *values) {
  return new CPUDeviceIndexVector(size, values);
}

DeviceScalar *CPUBackend::createScalar() { return new CPUDeviceScalar(); }

DeviceScalar *CPUBackend::createScalar(float value) {
  return new CPUDeviceScalar(value);
}

DeviceCSRMatrix *
CPUBackend::createCSRMatrix(const HostCSRMatrix &matrix) {
  return new CPUDeviceCSRMatrix(matrix);
}

DeviceELLMatrix *
CPUBackend::createELLMatrix(const HostELLMatrix &matrix) {
  return new CPUDeviceELLMatrix(matrix);
}

// ============================================================
// Low-level encoding API.
// ============================================================

void CPUBackend::encodeScale(BackendEncoder &, DeviceVector &x,
                             float alpha) {
  float *xData = CPUVec(x).data();
  const std::size_t n = x.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    xData[i] *= alpha;
  }
}

void CPUBackend::encodeScale(BackendEncoder &encoder, DeviceVector &x,
                             const DeviceScalar &alpha) {
  encodeScale(encoder, x, CPUScalar(alpha).value());
}

void CPUBackend::encodeCopy(BackendEncoder &, const DeviceVector &x,
                            DeviceVector &xCopy) {
  if (x.size() != xCopy.size()) {
    throw std::runtime_error("CPUBackend::encodeCopy: size mismatch.");
  }
  const float *xData = CPUVec(x).data();
  float *xCopyData = CPUVec(xCopy).data();
  const std::size_t n = x.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    xCopyData[i] = xData[i];
  }
}

void CPUBackend::encodeScaleVector(BackendEncoder &,
                                   const DeviceVector &scaleVector,
                                   DeviceVector &output) {
  if (scaleVector.size() != output.size()) {
    throw std::runtime_error(
        "CPUBackend::encodeScaleVector: size mismatch.");
  }
  const float *scale = CPUVec(scaleVector).data();
  float *out = CPUVec(output).data();
  const std::size_t n = output.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    out[i] *= scale[i];
  }
}

void CPUBackend::encodeSetZero(BackendEncoder &, DeviceVector &input) {
  float *data = CPUVec(input).data();
  const std::size_t n = input.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = 0.0f;
  }
}

void CPUBackend::encodeAxpy(BackendEncoder &, float alpha,
                            const DeviceVector &x, DeviceVector &y) {
  if (x.size() != y.size()) {
    throw std::runtime_error("CPUBackend::encodeAxpy: size mismatch.");
  }
  const float *xData = CPUVec(x).data();
  float *yData = CPUVec(y).data();
  const std::size_t n = x.size();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    yData[i] += alpha * xData[i];
  }
}

void CPUBackend::encodeAxpy(BackendEncoder &encoder,
                            const DeviceScalar &alpha, const DeviceVector &x,
                            DeviceVector &y) {
  encodeAxpy(encoder, CPUScalar(alpha).value(), x, y);
}

void CPUBackend::encodeSpmv(BackendEncoder &, const DeviceCSRMatrix &A,
                            const DeviceVector &x, DeviceVector &y) {
  const CPUDeviceCSRMatrix &matrix = CPUMatrix(A);
  if (x.size() != matrix.cols() || y.size() != matrix.rows()) {
    throw std::runtime_error("CPUBackend::encodeSpmv: size mismatch.");
  }
  const float *xData = CPUVec(x).data();
  float *yData = CPUVec(y).data();
  const int nRows = static_cast<int>(matrix.rows());

  // rowDotExcluding only skips the diagonal entry when diagOut is
  // non-null; passing nullptr here (and an empty exclusion range,
  // excludeColBegin == excludeColEnd == 0, which excludes nothing
  // since every valid column index is >= 0) makes it sum every
  // entry in the row, diagonal included -- exactly y = A*x.
#pragma omp parallel for
  for (int row = 0; row < nRows; ++row) {
    yData[row] = rowDotExcluding(matrix, xData, row, 0, 0, nullptr);
  }
}

void CPUBackend::encodeGaussSeidelColor(
    BackendEncoder &, const DeviceCSRMatrix &matrixIn,
    const DeviceIndexVector &colorVertices, std::size_t colorStart,
    std::size_t colorCount, const DeviceVector &rhs, DeviceVector &solution,
    float omega) {
  const CPUDeviceCSRMatrix &matrix = CPUMatrix(matrixIn);
  const int *order = CPUIndexVec(colorVertices).data();
  const float *rhsData = CPUVec(rhs).data();
  float *solutionData = CPUVec(solution).data();

  // Rows within one color are a mutually independent set in the
  // matrix graph (that's what "a color" means), so it is safe to
  // update them all in parallel: none of them can be reading a
  // value another thread in this same loop is simultaneously
  // writing. Rows read values from OTHER colors too, via
  // rowDotExcluding below, and those are intentionally whatever the
  // most recently written values are -- that's the Gauss-Seidel
  // part, not a race, since a previous call to this function (for a
  // previous color) has already fully completed before this one is
  // invoked again (see MultigridSmoother / SymmetricGaussSeidelSmoother,
  // which loop over colors and call this once per color).
#pragma omp parallel for
  for (std::size_t k = colorStart; k < colorStart + colorCount; ++k) {
    const int row = order[k];
    float diag = 0.0f;
    const float offDiagSum =
        rowDotExcluding(matrix, solutionData, row, row, row + 1, &diag);
    if (std::abs(diag) < 1e-12f) {
      throw std::runtime_error(
          "CPUBackend::encodeGaussSeidelColor: zero diagonal entry.");
    }
    const float newValue = (rhsData[row] - offDiagSum) / diag;
    solutionData[row] =
        (1.0f - omega) * solutionData[row] + omega * newValue;
  }
}

void CPUBackend::encodeDot(BackendEncoder &, const DeviceVector &x,
                           const DeviceVector &y, DeviceScalar &result) {
  if (x.size() != y.size()) {
    throw std::runtime_error("CPUBackend::encodeDot: size mismatch.");
  }
  const float *xData = CPUVec(x).data();
  const float *yData = CPUVec(y).data();
  const std::size_t n = x.size();

  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (std::size_t i = 0; i < n; ++i) {
    sum += xData[i] * yData[i];
  }
  CPUScalar(result).setValue(sum);
}

void CPUBackend::encodeScalarDivide(BackendEncoder &,
                                    const DeviceScalar &numerator,
                                    const DeviceScalar &denominator,
                                    DeviceScalar &result) {
  CPUScalar(result).setValue(CPUScalar(numerator).value() /
                             CPUScalar(denominator).value());
}

void CPUBackend::encodeScalarMultiply(BackendEncoder &,
                                      const DeviceScalar &factor1,
                                      const DeviceScalar &factor2,
                                      DeviceScalar &result) {
  CPUScalar(result).setValue(CPUScalar(factor1).value() *
                             CPUScalar(factor2).value());
}

void CPUBackend::encodeScalarSet(BackendEncoder &, float value,
                                 DeviceScalar &output) {
  CPUScalar(output).setValue(value);
}

void CPUBackend::encodeScalarCopy(BackendEncoder &, const DeviceScalar &input,
                                  DeviceScalar &output) {
  CPUScalar(output).setValue(CPUScalar(input).value());
}

void CPUBackend::encodeScalarNegate(BackendEncoder &,
                                    const DeviceScalar &input,
                                    DeviceScalar &output) {
  CPUScalar(output).setValue(-CPUScalar(input).value());
}

void CPUBackend::encodeScalarSqrt(BackendEncoder &, const DeviceScalar &input,
                                  DeviceScalar &output) {
  CPUScalar(output).setValue(std::sqrt(CPUScalar(input).value()));
}

void CPUBackend::encodeApplyBlockInverses3x3(BackendEncoder &,
                                             const DeviceVector &inverseDiagBlocks,
                                             const DeviceVector &residual,
                                             DeviceVector &correction) {
  const float *inv = CPUVec(inverseDiagBlocks).data();
  const float *res = CPUVec(residual).data();
  float *corr = CPUVec(correction).data();

  const std::size_t numberOfBlocks = residual.size() / 3;

  // Every node's 3x3 block is independent of every other node's --
  // unlike the colored sweeps below, this needs no coloring at all.
#pragma omp parallel for
  for (std::size_t block = 0; block < numberOfBlocks; ++block) {
    const float *blockInv = inv + 9 * block;
    const float *r = res + 3 * block;
    float *c = corr + 3 * block;
    for (int row = 0; row < 3; ++row) {
      float sum = 0.0f;
      for (int col = 0; col < 3; ++col) {
        sum += blockInv[3 * row + col] * r[col];
      }
      c[row] = sum;
    }
  }
}

void CPUBackend::encodeBlockGaussSeidelColor3x3(
    BackendEncoder &, const DeviceCSRMatrix &matrixIn,
    const DeviceIndexVector &colorBlocks, const DeviceVector &diagonalInverses,
    std::size_t begin, std::size_t end, const DeviceVector &rhs,
    DeviceVector &solution, float omega) {
  const CPUDeviceCSRMatrix &matrix = CPUMatrix(matrixIn);
  const int *order = CPUIndexVec(colorBlocks).data();
  const float *diagInv = CPUVec(diagonalInverses).data();
  const float *rhsData = CPUVec(rhs).data();
  float *solutionData = CPUVec(solution).data();

  // Same independent-set argument as encodeGaussSeidelColor, just one
  // 3-DOF block per entry in colorBlocks instead of one scalar DOF.
#pragma omp parallel for
  for (std::size_t k = begin; k < end; ++k) {
    const int block = order[k];
    const int firstRow = 3 * block;

    float localResidual[3];
    for (int local = 0; local < 3; ++local) {
      const int row = firstRow + local;
      const float offBlockSum = rowDotExcluding(
          matrix, solutionData, row, firstRow, firstRow + 3, nullptr);
      localResidual[local] = rhsData[row] - offBlockSum;
    }

    const float *blockInv = diagInv + 9 * static_cast<std::size_t>(block);
    for (int local = 0; local < 3; ++local) {
      float newValue = 0.0f;
      for (int col = 0; col < 3; ++col) {
        newValue += blockInv[3 * local + col] * localResidual[col];
      }
      const int row = firstRow + local;
      solutionData[row] =
          (1.0f - omega) * solutionData[row] + omega * newValue;
    }
  }
}

// ============================================================
// High-level standalone API: create an encoder, encode once,
// submit+wait, clean up. Since encode* already ran synchronously,
// submitAndWait has nothing left to do, but we still go through it
// for symmetry with Metal/CUDA and in case you ever add real
// asynchrony here later.
// ============================================================

void CPUBackend::scale(DeviceVector &x, float alpha) {
  BackendEncoder *encoder = createEncoder();
  encodeScale(*encoder, x, alpha);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scale(DeviceVector &x, const DeviceScalar &alpha) {
  BackendEncoder *encoder = createEncoder();
  encodeScale(*encoder, x, alpha);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scaleVectorByVector(const DeviceVector &scaleVector,
                                     DeviceVector &output) {
  BackendEncoder *encoder = createEncoder();
  encodeScaleVector(*encoder, scaleVector, output);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::axpy(float alpha, const DeviceVector &x, DeviceVector &y) {
  BackendEncoder *encoder = createEncoder();
  encodeAxpy(*encoder, alpha, x, y);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::axpy(const DeviceScalar &alpha, const DeviceVector &x,
                      DeviceVector &y) {
  BackendEncoder *encoder = createEncoder();
  encodeAxpy(*encoder, alpha, x, y);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
                      DeviceVector &y) {
  BackendEncoder *encoder = createEncoder();
  encodeSpmv(*encoder, A, x, y);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::dot(const DeviceVector &x, const DeviceVector &y,
                     DeviceScalar &result) {
  BackendEncoder *encoder = createEncoder();
  encodeDot(*encoder, x, y, result);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarSet(float value, DeviceScalar &result) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarSet(*encoder, value, result);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarCopy(const DeviceScalar &input, DeviceScalar &output) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarCopy(*encoder, input, output);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarDivide(const DeviceScalar &numerator,
                              const DeviceScalar &denominator,
                              DeviceScalar &result) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarDivide(*encoder, numerator, denominator, result);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarMultiply(const DeviceScalar &factor1,
                                const DeviceScalar &factor2,
                                DeviceScalar &result) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarMultiply(*encoder, factor1, factor2, result);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarNegate(const DeviceScalar &input,
                              DeviceScalar &output) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarNegate(*encoder, input, output);
  submitAndWait(*encoder);
  delete encoder;
}

void CPUBackend::scalarSqrt(const DeviceScalar &input, DeviceScalar &output) {
  BackendEncoder *encoder = createEncoder();
  encodeScalarSqrt(*encoder, input, output);
  submitAndWait(*encoder);
  delete encoder;
}

} // namespace gpuSolver
