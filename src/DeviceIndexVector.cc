#include <GPUSolver/DeviceIndexVector.h>

#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace gpuSolver {

struct DeviceIndexVector::Impl {
  MTL::Buffer *buffer = nullptr;
  size_t size = 0;
};

DeviceIndexVector::DeviceIndexVector(MetalContext &context, std::size_t size)
    : impl_(new Impl) {
  impl_->size = size;
  impl_->buffer = context.device()->newBuffer(size * sizeof(int),MTL::ResourceStorageModeShared);
}


MTL::Buffer *DeviceIndexVector::getNativeBuffer() {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}
MTL::Buffer *DeviceIndexVector::getNativeBuffer() const {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}

DeviceIndexVector::DeviceIndexVector(MetalContext &context, const std::vector<int>& values)
    : impl_(new Impl) {
  impl_->size = values.size();
  impl_->buffer = context.device()->newBuffer(values.data(),impl_->size * sizeof(int),MTL::ResourceStorageModeShared);
}

DeviceIndexVector::DeviceIndexVector(MetalContext &context, std::size_t size, const int* values)
    : impl_(new Impl) {
  impl_->size = size;
  impl_->buffer = context.device()->newBuffer(values,impl_->size * sizeof(int),MTL::ResourceStorageModeShared);
}

size_t DeviceIndexVector::size(){
  return this->impl_->size;
}


size_t DeviceIndexVector::size() const {
  return this->impl_->size;
}



int* DeviceIndexVector::download() const{
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }

  int* resultValsRaw = static_cast<int*>(impl_->buffer->contents());

  return resultValsRaw;
}

DeviceIndexVector::~DeviceIndexVector(){
  impl_->buffer->release();
  delete impl_;
}


void DeviceIndexVector::updateValues(const int* values, const size_t sizeOfValues){

  if (impl_->size != sizeOfValues) {
    throw std::runtime_error(" The passed values dont have the right size");
  }

  int *destination = static_cast<int *>(impl_->buffer->contents());

  std::memcpy(destination, values, sizeOfValues * sizeof(int));
}


} // namespace gpuSolver
