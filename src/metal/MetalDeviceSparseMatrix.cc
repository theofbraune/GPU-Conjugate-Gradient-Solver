#include <GPUSolver/metal/MetalDeviceSparseMatrix.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace gpuSolver {

struct MetalDeviceCSRMatrix::Impl {

  MTL::Buffer *rowPtrBuffer = nullptr;
  MTL::Buffer *colIdxBuffer = nullptr;
  MTL::Buffer *valBuffer = nullptr;

  std::size_t nRows = 0;
  std::size_t nCols = 0;
  std::size_t nnz = 0;
};

MetalDeviceCSRMatrix::MetalDeviceCSRMatrix(MetalContext &context, std::size_t nRows,
                                 std::size_t nCols, std::size_t nnz,
                                 const int *rowPtr, const int *colIdxPtr,
                                 const float *valPtr) {

  if (nRows == 0 || nCols == 0) {
    throw std::runtime_error(
        "DeviceCSRMatrix: matrix dimensions must be nonzero.");
  }

  if (rowPtr == nullptr || colIdxPtr == nullptr || valPtr == nullptr) {
    throw std::runtime_error("You passed a niullpointer to the constructor of "
                             "the sparse matrix class");
  }

  if (rowPtr[nRows] != static_cast<int>(nnz)) {
    throw std::runtime_error(
        "DeviceCSRMatrix: CSR row pointer does not match nnz.");
  }
  if (nnz == 0) {
    throw std::runtime_error(
        "DeviceCSRMatrix: matrix must contain at least one nonzero.");
  }

  if (rowPtr[0] != 0) {
    throw std::runtime_error(
        "DeviceCSRMatrix: CSR row pointer must start at zero.");
  }

  // create the impl instance
  this->impl_ = new Impl;

  // allocate the three buffers

  impl_->rowPtrBuffer = context.device()->newBuffer(
      rowPtr, (nRows + 1) * sizeof(int), MTL::ResourceStorageModeShared);

  impl_->colIdxBuffer = context.device()->newBuffer(
      colIdxPtr, nnz * sizeof(int), MTL::ResourceStorageModeShared);

  impl_->valBuffer = context.device()->newBuffer(
      valPtr, nnz * sizeof(float), MTL::ResourceStorageModeShared);

  impl_->nRows = nRows;
  impl_->nCols = nCols;
  impl_->nnz = nnz;

  if (!impl_->rowPtrBuffer || !impl_->colIdxBuffer || !impl_->valBuffer) {
    throw std::runtime_error("Could not allocate buffers for DeviceCSRMatrix.");
  }
}

MetalDeviceCSRMatrix::MetalDeviceCSRMatrix(
    MetalContext& context,
    const HostCSRMatrix& matrix)
    : MetalDeviceCSRMatrix(
          context,
          matrix.rows(),
          matrix.cols(),
          matrix.nnz(),
          matrix.activeRowPtr(),
          matrix.activeColPtr(),
          matrix.activeValPtr())
{
}

MetalDeviceCSRMatrix::~MetalDeviceCSRMatrix() {

  if (impl_->rowPtrBuffer) {
    impl_->rowPtrBuffer->release();
  }

  if (impl_->colIdxBuffer) {
    impl_->colIdxBuffer->release();
  }

  if (impl_->valBuffer) {
    impl_->valBuffer->release();
  }

  delete impl_;
}

std::size_t MetalDeviceCSRMatrix::rows() const { return impl_->nRows; }

std::size_t MetalDeviceCSRMatrix::cols() const { return impl_->nCols; }

std::size_t MetalDeviceCSRMatrix::nnz() const { return impl_->nnz; }

MTL::Buffer* MetalDeviceCSRMatrix::getRowPtrBuffer(){
  return this->impl_->rowPtrBuffer; 
}

MTL::Buffer* MetalDeviceCSRMatrix::getRowPtrBuffer() const{
  return this->impl_->rowPtrBuffer;
}

MTL::Buffer* MetalDeviceCSRMatrix::getColPtrBuffer(){
  return this->impl_->colIdxBuffer;
}

MTL::Buffer* MetalDeviceCSRMatrix::getColPtrBuffer() const{
  return this->impl_->colIdxBuffer;
}

MTL::Buffer* MetalDeviceCSRMatrix::getValPtrBuffer(){
  return this->impl_->valBuffer;
}

MTL::Buffer* MetalDeviceCSRMatrix::getValPtrBuffer() const{
  return this->impl_->valBuffer;
}

void MetalDeviceCSRMatrix::updateValues(const float *values, std::size_t nnz) {

  if (nnz != impl_->nnz) {
    throw std::runtime_error(
        "The new nbOfValues does not fir the allocated memory! ");
  }

  if (values == nullptr) {
    throw std::runtime_error(
        "DeviceCSRMatrix::updateValues: values pointer is null.");
  }

  if (impl_->valBuffer == nullptr) {
    throw std::runtime_error(
        "DeviceCSRMatrix::updateValues: value buffer is not allocated.");
  }

  float *destination = static_cast<float *>(impl_->valBuffer->contents());

  std::memcpy(destination, values, nnz * sizeof(float));
}

} // namespace gpuSolver
