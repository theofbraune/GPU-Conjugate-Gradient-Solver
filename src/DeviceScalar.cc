#include <GPUSolver/DeviceScalar.h>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLResource.hpp>

namespace gpuSolver {

struct DeviceScalar::Impl {
  MTL::Buffer *buffer = nullptr;
};

DeviceScalar::DeviceScalar(MetalContext &context) {

  this->impl_ = new Impl;

  this->impl_->buffer = context.device()->newBuffer(
      sizeof(float), MTL::ResourceStorageModeShared);
}

DeviceScalar::DeviceScalar(MetalContext &context, float value) {

  this->impl_ = new Impl;

  this->impl_->buffer = context.device()->newBuffer(
      &value, sizeof(float), MTL::ResourceStorageModeShared);
}

DeviceScalar::~DeviceScalar() {

  if (impl_->buffer != nullptr) {
    impl_->buffer->release();
  }
  delete impl_;
}

MTL::Buffer *DeviceScalar::getNativeBuffer() const { return impl_->buffer; }

// ONLY for debugging purpose. Should never be used in a proper CG
float DeviceScalar::download() const {
  const float *value = static_cast<const float *>(impl_->buffer->contents());

  return value[0];
}
} // namespace gpuSolver
