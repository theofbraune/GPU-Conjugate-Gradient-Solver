
#include <GPUSolver/metal/MetalDeviceIndexVector.h>

#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gpuSolver {

struct MetalDeviceIndexVector::Impl {
  MTL::Buffer *buffer = nullptr;
  size_t size = 0;
};

MetalDeviceIndexVector::MetalDeviceIndexVector(MetalContext &context, std::size_t size)
    : impl_(new Impl) {
  impl_->size = size;
  impl_->buffer = context.device()->newBuffer(size * sizeof(int),MTL::ResourceStorageModeShared);
}


MTL::Buffer *MetalDeviceIndexVector::getNativeBuffer() {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}
MTL::Buffer *MetalDeviceIndexVector::getNativeBuffer() const {
  // check if the buffer is loaded
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }
  return impl_->buffer;
}

MetalDeviceIndexVector::MetalDeviceIndexVector(MetalContext &context, const std::vector<int>& values)
    : impl_(new Impl) {
  impl_->size = values.size();
  impl_->buffer = context.device()->newBuffer(values.data(),impl_->size * sizeof(int),MTL::ResourceStorageModeShared);
}

MetalDeviceIndexVector::MetalDeviceIndexVector(MetalContext &context, std::size_t size, const int* values)
    : impl_(new Impl) {
  impl_->size = size;
  impl_->buffer = context.device()->newBuffer(values,impl_->size * sizeof(int),MTL::ResourceStorageModeShared);
}

size_t MetalDeviceIndexVector::size(){
  return this->impl_->size;
}


size_t MetalDeviceIndexVector::size() const {
  return this->impl_->size;
}



int* MetalDeviceIndexVector::download() const{
  if (impl_->buffer == nullptr) {
    throw std::runtime_error("The data in the buffer is not allocated! ");
  }

  int* resultValsRaw = static_cast<int*>(impl_->buffer->contents());

  return resultValsRaw;
}

MetalDeviceIndexVector::~MetalDeviceIndexVector(){
  impl_->buffer->release();
  delete impl_;
}


void MetalDeviceIndexVector::updateValues(const int* values, const size_t sizeOfValues){

  if (impl_->size != sizeOfValues) {
    throw std::runtime_error(" The passed values dont have the right size");
  }

  int *destination = static_cast<int *>(impl_->buffer->contents());

  std::memcpy(destination, values, sizeOfValues * sizeof(int));
}


} // namespace gpuSolver
