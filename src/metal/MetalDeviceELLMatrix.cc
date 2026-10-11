#include <GPUSolver/metal/MetalDeviceELLMatrix.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/HostELLMatrix.h>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <GPUSolver/metal/MetalContext.h>
#include <cstddef>
#include <cstring>

namespace gpuSolver {
struct MetalDeviceELLMatrix::Impl {
  MTL::Buffer *ellColIdx = nullptr;
  MTL::Buffer *ellValues = nullptr;

  MTL::Buffer *overflowRowPtr = nullptr;
  MTL::Buffer *overflowColIdx = nullptr;
  MTL::Buffer *overflowValues = nullptr;

  std::size_t nRows = 0;
  std::size_t nCols = 0;

  std::size_t ellWidth = 0;
  std::size_t overflowNnz = 0;
};

MetalDeviceELLMatrix::MetalDeviceELLMatrix(MetalContext &context,
                                 const HostELLMatrix &matrix) {
  impl_ = new Impl;

  impl_->nRows = matrix.rows();

  impl_->nCols = matrix.cols();

  impl_->ellWidth = matrix.ellWidth();

  impl_->overflowNnz = matrix.overflowNnz();

  const std::size_t ellSize = impl_->nRows * impl_->ellWidth;

  // --------------------------------------------------------
  // ELL part
  // --------------------------------------------------------

  impl_->ellColIdx =
      context.device()->newBuffer(matrix.ellColIdx(), ellSize * sizeof(int),
                                  MTL::ResourceStorageModeShared);

  impl_->ellValues =
      context.device()->newBuffer(matrix.ellValues(), ellSize * sizeof(float),
                                  MTL::ResourceStorageModeShared);

  // --------------------------------------------------------
  // Overflow row pointer.
  //
  // This always exists, even if there are no overflow entries.
  //
  // In that case every row has
  //
  //     overflowRowPtr[row]
  //       ==
  //     overflowRowPtr[row + 1]
  //
  // and the shader loop executes zero iterations.
  // --------------------------------------------------------

  impl_->overflowRowPtr = context.device()->newBuffer(
      matrix.overflowRowPtr(), (impl_->nRows + 1) * sizeof(int),
      MTL::ResourceStorageModeShared);

  // --------------------------------------------------------
  // Overflow column/value arrays.
  //
  // Keep valid Metal buffers even if there is no overflow.
  // This means MetalBackend does not need a special case.
  // --------------------------------------------------------

  if (impl_->overflowNnz > 0) {
    impl_->overflowColIdx = context.device()->newBuffer(
        matrix.overflowColIdx(), impl_->overflowNnz * sizeof(int),
        MTL::ResourceStorageModeShared);

    impl_->overflowValues = context.device()->newBuffer(
        matrix.overflowValues(), impl_->overflowNnz * sizeof(float),
        MTL::ResourceStorageModeShared);
  } else {
    const int dummyColumn = 0;
    const float dummyValue = 0.0f;

    impl_->overflowColIdx = context.device()->newBuffer(
        &dummyColumn, sizeof(int), MTL::ResourceStorageModeShared);

    impl_->overflowValues = context.device()->newBuffer(
        &dummyValue, sizeof(float), MTL::ResourceStorageModeShared);
  }

  if (impl_->ellColIdx == nullptr || impl_->ellValues == nullptr ||
      impl_->overflowRowPtr == nullptr || impl_->overflowColIdx == nullptr ||
      impl_->overflowValues == nullptr) {
    throw std::runtime_error(
        "DeviceELLMatrix: failed to allocate Metal buffers.");
  }
}

MetalDeviceELLMatrix::~MetalDeviceELLMatrix() {
  if (impl_->ellColIdx != nullptr) {
    impl_->ellColIdx->release();
  }

  if (impl_->ellValues != nullptr) {
    impl_->ellValues->release();
  }

  if (impl_->overflowRowPtr != nullptr) {
    impl_->overflowRowPtr->release();
  }

  if (impl_->overflowColIdx != nullptr) {
    impl_->overflowColIdx->release();
  }

  if (impl_->overflowValues != nullptr) {
    impl_->overflowValues->release();
  }

  delete impl_;
}

std::size_t MetalDeviceELLMatrix::rows() const
{
    return impl_->nRows;
}

std::size_t MetalDeviceELLMatrix::cols() const
{
    return impl_->nCols;
}

std::size_t MetalDeviceELLMatrix::ellWidth() const
{
    return impl_->ellWidth;
}

std::size_t MetalDeviceELLMatrix::overflowNnz() const
{
    return impl_->overflowNnz;
}

MTL::Buffer* MetalDeviceELLMatrix::ellColIdxBuffer() const
{
    return impl_->ellColIdx;
}

MTL::Buffer* MetalDeviceELLMatrix::ellValuesBuffer() const
{
    return impl_->ellValues;
}

MTL::Buffer* MetalDeviceELLMatrix::overflowRowPtrBuffer() const
{
    return impl_->overflowRowPtr;
}

MTL::Buffer* MetalDeviceELLMatrix::overflowColIdxBuffer() const
{
    return impl_->overflowColIdx;
}

MTL::Buffer* MetalDeviceELLMatrix::overflowValuesBuffer() const
{
    return impl_->overflowValues;
}

} // namespace gpuSolver
