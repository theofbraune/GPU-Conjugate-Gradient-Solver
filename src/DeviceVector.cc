#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <cstddef>
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

DeviceVector::DeviceVector(MetalContext &context, std::size_t size,
                           const float *values)
    : impl_(new Impl) {
  impl_->size = size;
  impl_->buffer =
      context.device()->newBuffer(values, size * sizeof(float),
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

void DeviceVector::updateValues(const float *values,
                                const size_t sizeOfValues) {

  // impl_->size = sizeOfValues;
  // impl_->buffer = context.device()->newBuffer(
  //     values, sizeOfValues * sizeof(float), MTL::ResourceStorageModeShared);

  if (impl_->size != sizeOfValues) {
    throw std::runtime_error(" The passed values dont have the right size");
  }

  float *destination = static_cast<float *>(impl_->buffer->contents());

  std::memcpy(destination, values, sizeOfValues * sizeof(float));
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

size_t DeviceVector::size() { return impl_->size; }

size_t DeviceVector::size() const { return impl_->size; }

} // namespace gpuSolver
