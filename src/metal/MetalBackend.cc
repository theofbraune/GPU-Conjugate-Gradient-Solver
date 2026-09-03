#include "GPUSolver/DeviceVector.h"
#include <Foundation/NSError.hpp>
#include <Foundation/NSString.hpp>
#include <GPUSolver/metal/MetalBackend.h>

#include <GPUSolver/metal/MetalContext.h>

#include <Foundation/Foundation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLTypes.hpp>

#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace gpuSolver {

struct MetalBackend::Impl {

  MetalContext &context;
  MTL::ComputePipelineState *scalePipeline = nullptr;
  MTL::ComputePipelineState *axpyPipeline = nullptr;
  MTL::ComputePipelineState *spmvPipeline = nullptr;

  explicit Impl(MetalContext &context_) : context(context_) {}
};

MTL::ComputePipelineState *MetalBackend::makePipeline(MTL::Device *device,
                                                      MTL::Library *library,
                                                      const char *name) {

  // chwck if the path is ligit for a proper kernel first

  MTL::Function *pipelineFunction =
      library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  if (!pipelineFunction) {
    throw std::runtime_error(
        std::string(
            " the name that you passed for the kernel is invalid. Can't find") +
        name);
  }

  // device->newComputePipelineState
  NS::Error *error = nullptr;

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(pipelineFunction, &error);

  pipelineFunction->release();

  if (!pipeline) {
    std::string message = "Could not create Metal pipeline for ";

    message += name;

    if (error) {
      message += ": ";
      message += error->localizedDescription()->utf8String();
    }

    throw std::runtime_error(message);
  }

  return pipeline;
}

void MetalBackend::dispatch1D(MTL::ComputeCommandEncoder *encoder,
                              MTL::ComputePipelineState *pipeline, size_t n) {

  NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

  if (threadGroupSize > n) {
    threadGroupSize = n;
  }

  MTL::Size gridSize(n, 1, 1);
  MTL::Size threadsPerThreadgroup(threadGroupSize, 1, 1);

  encoder->dispatchThreads(gridSize, threadsPerThreadgroup);
}

MetalBackend::MetalBackend(MetalContext &context) {

  impl_ = new Impl(context);

  impl_->axpyPipeline =
      makePipeline(context.device(), context.library(), "axpy");

  impl_->scalePipeline =
      makePipeline(context.device(), context.library(), "scale");

  impl_->spmvPipeline =
      makePipeline(context.device(), context.library(), "spmv");
}

MetalBackend::~MetalBackend() {

  if (impl_->scalePipeline) {
    impl_->scalePipeline->release();
  }

  if (impl_->axpyPipeline) {
    impl_->axpyPipeline->release();
  }

  if (impl_->spmvPipeline) {
    impl_->spmvPipeline->release();
  }

  delete impl_;
}

void MetalBackend::scale(DeviceVector &x, float scalar) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->scalePipeline);

  MTL::Buffer *bufferForX = x.getNativeBuffer();
  // MTL::Buffer *bufferForX = x.impl_->buffer;

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBytes(&scalar, sizeof(float), 1);

  size_t sizeOfTheVector = x.getSizeOfVector();

  dispatch1D(encoder, impl_->scalePipeline, sizeOfTheVector);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::axpy(float alpha, const DeviceVector &x, DeviceVector &y) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->axpyPipeline);

  MTL::Buffer *bufferForX = x.getNativeBuffer();
  MTL::Buffer *bufferForY = y.getNativeBuffer();

  // MTL::Buffer *bufferForX = x.impl_->buffer;
  // MTL::Buffer *bufferForY = y.impl_->buffer;

  size_t sizeOfx = x.getSizeOfVector();
  size_t sizeOfy = y.getSizeOfVector();

  if (sizeOfx != sizeOfy) {
    throw std::runtime_error(
        " the two vectorsd tat you are passing dont have the same size");
  }

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBuffer(bufferForY, 0, 1);

  encoder->setBytes(&alpha, sizeof(float), 2);

  dispatch1D(encoder, impl_->axpyPipeline, sizeOfx);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
                        DeviceVector &Ax) {

  if (A.cols() != x.getSizeOfVector()) {
    throw std::runtime_error(
        "MetalBackend::spmv: matrix columns do not match x size.");
  }

  if (A.rows() != Ax.getSizeOfVector()) {
    throw std::runtime_error(
        "MetalBackend::spmv: matrix rows do not match output size.");
  }

  if (A.rows() == 0) {
    return;
  }
  
  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->spmvPipeline);

  MTL::Buffer *bufferRowPtr = A.getRowPtrBuffer();
  MTL::Buffer *bufferColPtr = A.getColPtrBuffer();
  MTL::Buffer *bufferValPtr = A.getValPtrBuffer();

  MTL::Buffer *bufferXvals = x.getNativeBuffer();
  MTL::Buffer *bufferAxVals = Ax.getNativeBuffer();
  std::size_t nbOfRows = A.rows();

  // todo sanity checks
  //
  // set the buffers on the encoder now. Check that they match the order of the
  // kernel buffers
  encoder->setBuffer(bufferRowPtr, 0, 0);

  encoder->setBuffer(bufferColPtr, 0, 1);

  encoder->setBuffer(bufferValPtr, 0, 2);

  encoder->setBuffer(bufferXvals, 0, 3);

  encoder->setBuffer(bufferAxVals, 0, 4);

  // encoder->setBytes(&nbOfRows, sizeof(int), 5);

  dispatch1D(encoder, impl_->spmvPipeline, nbOfRows);

  encoder->endEncoding();

  commandBuffer->commit();

  commandBuffer->waitUntilCompleted();
}

void MetalBackend::encodeSpmv(
    MTL::ComputeCommandEncoder* encoder,
    const DeviceCSRMatrix& A,
    const DeviceVector& x,
    DeviceVector& Ax)
{
    encoder->setComputePipelineState(
        impl_->spmvPipeline
    );

    encoder->setBuffer(
        A.getRowPtrBuffer(),
        0,
        0
    );

    encoder->setBuffer(
        A.getColPtrBuffer(),
        0,
        1
    );

    encoder->setBuffer(
        A.getValPtrBuffer(),
        0,
        2
    );

    encoder->setBuffer(
        x.getNativeBuffer(),
        0,
        3
    );

    encoder->setBuffer(
        Ax.getNativeBuffer(),
        0,
        4
    );

    dispatch1D(
        encoder,
        impl_->spmvPipeline,
        A.rows()
    );
}

void MetalBackend::spmvRepeated(
    const DeviceCSRMatrix& A,
    const DeviceVector& x,
    DeviceVector& Ax,
    std::size_t repetitions)
{
    MTL::CommandBuffer* commandBuffer =
        impl_->context.queue()->commandBuffer();

    MTL::ComputeCommandEncoder* encoder =
        commandBuffer->computeCommandEncoder();

    for (std::size_t iteration = 0;
         iteration < repetitions;
         ++iteration)
    {
        encodeSpmv(
            encoder,
            A,
            x,
            Ax
        );
    }

    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}


} // namespace gpuSolver
