#pragma once
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>

namespace MTL {
class Buffer;
}

namespace gpuSolver {

class MetalBackend;
class MetalContext;

class MetalDeviceCSRMatrix : public DeviceCSRMatrix {

public:
  MetalDeviceCSRMatrix(MetalContext &context, std::size_t nRows, std::size_t nCols,
                  std::size_t nnz, const int *rowPtr, const int *colIdxPtr,
                  const float *valPtr);

  MetalDeviceCSRMatrix(MetalContext &context, const HostCSRMatrix &hostSparseMatrix);

  ~MetalDeviceCSRMatrix();

  MetalDeviceCSRMatrix(const DeviceCSRMatrix &) = delete;
  MetalDeviceCSRMatrix &operator=(const DeviceCSRMatrix &) = delete;

  std::size_t rows() const override;
  std::size_t cols() const override;
  std::size_t nnz() const override;

  void updateValues(const float *values, std::size_t numberOfValues) override;

private:
  struct Impl;
  Impl *impl_;

  MTL::Buffer *getRowPtrBuffer();
  MTL::Buffer *getRowPtrBuffer() const;

  MTL::Buffer *getColPtrBuffer();
  MTL::Buffer *getColPtrBuffer() const;

  MTL::Buffer *getValPtrBuffer();
  MTL::Buffer *getValPtrBuffer() const;

  friend class MetalBackend;
};

} // namespace gpuSolver
