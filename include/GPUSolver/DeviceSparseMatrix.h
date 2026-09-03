#pragma once

#include <cstddef>

namespace MTL {
  class Buffer;
}

namespace gpuSolver {

class MetalBackend;
class MetalContext;

class DeviceCSRMatrix {

public:
  DeviceCSRMatrix(MetalContext &context, std::size_t nRows, std::size_t nCols,
                  std::size_t nnz, const int *rowPtr, const int *colIdxPtr,
                  const float *valPtr);

  ~DeviceCSRMatrix();

  DeviceCSRMatrix(const DeviceCSRMatrix &) = delete;
  DeviceCSRMatrix& operator=(const DeviceCSRMatrix&) = delete;

  std::size_t rows() const;
  std::size_t cols() const;
  std::size_t nnz() const;

  void updateValues(const float *values, std::size_t numberOfValues);

private:
  struct Impl;
  Impl *impl_;

  MTL::Buffer* getRowPtrBuffer();
  MTL::Buffer* getRowPtrBuffer() const;

  MTL::Buffer* getColPtrBuffer();
  MTL::Buffer* getColPtrBuffer() const;

  MTL::Buffer* getValPtrBuffer();
  MTL::Buffer* getValPtrBuffer() const;


  friend class MetalBackend;
};

} // namespace gpuSolver
