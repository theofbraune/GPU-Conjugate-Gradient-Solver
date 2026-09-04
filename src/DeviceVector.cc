#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace gpuSolver {

struct DeviceVector::Impl {
  MTL::Buffer *buffer = nullptr;
  size_t size = 0;
};

DeviceVector::DeviceVector(MetalContext &context,
                           const std::vector<float> &values)
    : impl_(new Impl) {
  size_t sizeVector = values.size();
  const float *valuesRaw = values.data();
  impl_->size = sizeVector;
  impl_->buffer =
      context.device()->newBuffer(values.data(), values.size() * sizeof(float),
                                  MTL::ResourceStorageModeShared);
}

DeviceVector::DeviceVector(MetalContext &context, size_t size)
    : impl_(new Impl) {
  impl_->size = size;

  impl_->buffer = context.device()->newBuffer(size * sizeof(float),
                                              MTL::ResourceStorageModeShared);
}

DeviceVector::~DeviceVector() {

  impl_->buffer->release();

  delete impl_;
}

void DeviceVector::upload(MetalContext &context, const float *values,
                          const size_t sizeOfValues) {

  impl_->size = sizeOfValues;
  impl_->buffer = context.device()->newBuffer(
      values, sizeOfValues * sizeof(float), MTL::ResourceStorageModeShared);
}

float *DeviceVector::download() const {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }

  float *resultValsRaw = static_cast<float *>(impl_->buffer->contents());

  return resultValsRaw;
}

MTL::Buffer *DeviceVector::getNativeBuffer() {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}
MTL::Buffer *DeviceVector::getNativeBuffer() const {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}

size_t DeviceVector::getSizeOfVector() { return impl_->size; }

size_t DeviceVector::getSizeOfVector() const { return impl_->size; }

std::size_t DeviceVector::size() const { return impl_->size; }

} // namespace gpuSolver
