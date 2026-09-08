#include "GPUSolver/Backend.h"
#include "GPUSolver/BackendEncoder.h"
#include "GPUSolver/DeviceScalar.h"
#include "GPUSolver/DeviceVector.h"
#include <Foundation/NSError.hpp>
#include <Foundation/NSString.hpp>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalEncoder.h>

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
  MTL::ComputePipelineState *vectorCopyPipeline = nullptr;
  MTL::ComputePipelineState *scalePipelineDevice = nullptr;
  MTL::ComputePipelineState *axpyPipeline = nullptr;
  MTL::ComputePipelineState *axpyPipelineDevice = nullptr;
  MTL::ComputePipelineState *spmvPipeline = nullptr;
  MTL::ComputePipelineState *spmvELLPipeline = nullptr;

  // scalar kernels
  MTL::ComputePipelineState *scalarSetPipeline = nullptr;
  MTL::ComputePipelineState *scalarCopyPipeline = nullptr;
  MTL::ComputePipelineState *scalarDividePipeline = nullptr;
  MTL::ComputePipelineState *scalarMultiplyPipeline = nullptr;
  MTL::ComputePipelineState *scalarNegatePipeline = nullptr;
  MTL::ComputePipelineState *scalarSqrtPipeline = nullptr;

  // now the dataa needed for the dot pipeline
  MTL::ComputePipelineState *dotPartialPipeline = nullptr;
  MTL::Buffer *reductionScratchA = nullptr;
  MTL::Buffer *reductionScratchB = nullptr;
  MTL::ComputePipelineState *dotReducePipeline = nullptr;

  std::size_t reductionScratchCapacity = 0;

  explicit Impl(MetalContext &context_) : context(context_) {}
};

BackendEncoder *MetalBackend::createEncoder() {
  return new MetalEncoder(impl_->context);
}

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
  if (n == 0) {
    return;
  }
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

  impl_->axpyPipelineDevice =
      makePipeline(context.device(), context.library(), "axpyDevice");

  impl_->scalePipeline =
      makePipeline(context.device(), context.library(), "scale");

  impl_->vectorCopyPipeline =
      makePipeline(context.device(), context.library(), "vectorCopy");

  impl_->scalePipelineDevice =
      makePipeline(context.device(), context.library(), "scaleDevice");

  impl_->spmvPipeline =
      makePipeline(context.device(), context.library(), "spmv");

  impl_->spmvELLPipeline =
      makePipeline(context.device(), context.library(), "spmvELL");

  impl_->dotPartialPipeline =
      makePipeline(context.device(), context.library(), "dotPartial");

  impl_->dotReducePipeline =
      makePipeline(context.device(), context.library(), "reduceSumPartial");
  // make the scalar kernels
  impl_->scalarSetPipeline =
      makePipeline(context.device(), context.library(), "scalarSet");

  impl_->scalarCopyPipeline =
      makePipeline(context.device(), context.library(), "scalarCopy");

  impl_->scalarDividePipeline =
      makePipeline(context.device(), context.library(), "scalarDivide");

  impl_->scalarMultiplyPipeline =
      makePipeline(context.device(), context.library(), "scalarMultiply");

  impl_->scalarNegatePipeline =
      makePipeline(context.device(), context.library(), "scalarNegate");

  impl_->scalarSqrtPipeline =
      makePipeline(context.device(), context.library(), "scalarSqrt");
}

MetalBackend::~MetalBackend() {

  if (impl_->scalePipeline) {
    impl_->scalePipeline->release();
  }

  if (impl_->scalePipelineDevice) {
    impl_->scalePipelineDevice->release();
  }

  if (impl_->axpyPipeline) {
    impl_->axpyPipeline->release();
  }

  if (impl_->axpyPipelineDevice) {
    impl_->axpyPipelineDevice->release();
  }

  if (impl_->spmvPipeline) {
    impl_->spmvPipeline->release();
  }

  if (impl_->spmvELLPipeline) {
    impl_->spmvELLPipeline->release();
  }

  if (impl_->dotPartialPipeline) {
    impl_->dotPartialPipeline->release();
  }

  if (impl_->dotReducePipeline) {
    impl_->dotReducePipeline->release();
  }

  if (impl_->reductionScratchA) {
    impl_->reductionScratchA->release();
  }

  if (impl_->reductionScratchB) {
    impl_->reductionScratchB->release();
  }

  if (impl_->scalarSetPipeline) {
    impl_->scalarSetPipeline->release();
  }

  if (impl_->scalarCopyPipeline) {
    impl_->scalarCopyPipeline->release();
  }

  if (impl_->scalarDividePipeline) {
    impl_->scalarDividePipeline->release();
  }

  if (impl_->scalarMultiplyPipeline) {
    impl_->scalarMultiplyPipeline->release();
  }

  if (impl_->scalarNegatePipeline) {
    impl_->scalarNegatePipeline->release();
  }

  if (impl_->scalarSqrtPipeline) {
    impl_->scalarSqrtPipeline->release();
  }

  if (impl_->vectorCopyPipeline) {
    impl_->vectorCopyPipeline->release();
  }

  delete impl_;
}

void MetalBackend::ensureReductionScratchCapacity(
    std::size_t requiredCapacity) {
  if (requiredCapacity <= impl_->reductionScratchCapacity) {
    return;
  }

  if (impl_->reductionScratchA != nullptr) {
    impl_->reductionScratchA->release();
  }

  if (impl_->reductionScratchB != nullptr) {
    impl_->reductionScratchB->release();
  }

  impl_->reductionScratchA = impl_->context.device()->newBuffer(
      requiredCapacity * sizeof(float), MTL::ResourceStorageModeShared);

  impl_->reductionScratchB = impl_->context.device()->newBuffer(
      requiredCapacity * sizeof(float), MTL::ResourceStorageModeShared);

  impl_->reductionScratchCapacity = requiredCapacity;
}

// encode all the methods before calling them
void MetalBackend::encodeScaleMetal(MTL::ComputeCommandEncoder *encoder,
                                    DeviceVector &x, const float alpha) {

  encoder->setComputePipelineState(impl_->scalePipeline);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBytes(&alpha, sizeof(float), 1);

  dispatch1D(encoder, impl_->scalePipeline, x.size());
}

void MetalBackend::encodeScaleMetal(MTL::ComputeCommandEncoder *encoder,
                                    DeviceVector &x,
                                    const DeviceScalar &alpha) {

  encoder->setComputePipelineState(impl_->scalePipelineDevice);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBuffer(alpha.getNativeBuffer(), 0, 1);

  dispatch1D(encoder, impl_->scalePipelineDevice, x.size());
}

void MetalBackend::encodeCopyMetal(MTL::ComputeCommandEncoder *encoder,
                                   const DeviceVector &x, DeviceVector &xCopy) {
  encoder->setComputePipelineState(impl_->vectorCopyPipeline);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBuffer(xCopy.getNativeBuffer(), 0, 1);

  dispatch1D(encoder, impl_->vectorCopyPipeline, x.size());
}

void MetalBackend::encodeAxpyMetal(MTL::ComputeCommandEncoder *encoder,
                                   const DeviceScalar &alpha,
                                   const DeviceVector &x, DeviceVector &y) {

  encoder->setComputePipelineState(impl_->axpyPipelineDevice);

  MTL::Buffer *bufferForX = x.getNativeBuffer();
  MTL::Buffer *bufferForY = y.getNativeBuffer();

  // MTL::Buffer *bufferForX = x.impl_->buffer;
  // MTL::Buffer *bufferForY = y.impl_->buffer;

  size_t sizeOfx = x.size();
  size_t sizeOfy = y.size();

  if (sizeOfx != sizeOfy) {
    throw std::runtime_error(
        " the two vectorsd tat you are passing dont have the same size");
  }

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBuffer(bufferForY, 0, 1);

  // encoder->setBytes(&alpha, sizeof(float), 2);
  encoder->setBuffer(alpha.getNativeBuffer(), 0, 2);

  dispatch1D(encoder, impl_->axpyPipelineDevice, sizeOfx);
}

void MetalBackend::encodeAxpyMetal(MTL::ComputeCommandEncoder *encoder,
                                   float alpha, const DeviceVector &x,
                                   DeviceVector &y) {

  encoder->setComputePipelineState(impl_->axpyPipeline);

  MTL::Buffer *bufferForX = x.getNativeBuffer();
  MTL::Buffer *bufferForY = y.getNativeBuffer();

  // MTL::Buffer *bufferForX = x.impl_->buffer;
  // MTL::Buffer *bufferForY = y.impl_->buffer;

  size_t sizeOfx = x.size();
  size_t sizeOfy = y.size();

  if (sizeOfx != sizeOfy) {
    throw std::runtime_error(
        " the two vectorsd tat you are passing dont have the same size");
  }

  encoder->setBuffer(bufferForX, 0, 0);

  encoder->setBuffer(bufferForY, 0, 1);

  encoder->setBytes(&alpha, sizeof(float), 2);

  dispatch1D(encoder, impl_->axpyPipeline, sizeOfx);
}

void MetalBackend::encodeSpmvMetal(MTL::ComputeCommandEncoder *encoder,
                                   const DeviceCSRMatrix &A,
                                   const DeviceVector &x, DeviceVector &Ax) {
  encoder->setComputePipelineState(impl_->spmvPipeline);

  encoder->setBuffer(A.getRowPtrBuffer(), 0, 0);

  encoder->setBuffer(A.getColPtrBuffer(), 0, 1);

  encoder->setBuffer(A.getValPtrBuffer(), 0, 2);

  encoder->setBuffer(x.getNativeBuffer(), 0, 3);

  encoder->setBuffer(Ax.getNativeBuffer(), 0, 4);

  dispatch1D(encoder, impl_->spmvPipeline, A.rows());
}

void MetalBackend::encodeSpmvELLMetal(MTL::ComputeCommandEncoder *encoder,
                                      const DeviceELLMatrix &A,
                                      const DeviceVector &x, DeviceVector &y) {
  encoder->setComputePipelineState(impl_->spmvELLPipeline);

  encoder->setBuffer(A.ellColIdxBuffer(), 0, 0);

  encoder->setBuffer(A.ellValuesBuffer(), 0, 1);

  encoder->setBuffer(A.overflowRowPtrBuffer(), 0, 2);

  encoder->setBuffer(A.overflowColIdxBuffer(), 0, 3);

  encoder->setBuffer(A.overflowValuesBuffer(), 0, 4);

  encoder->setBuffer(x.getNativeBuffer(), 0, 5);

  encoder->setBuffer(y.getNativeBuffer(), 0, 6);

  const uint32_t nRows = static_cast<uint32_t>(A.rows());

  const uint32_t ellWidth = static_cast<uint32_t>(A.ellWidth());

  encoder->setBytes(&nRows, sizeof(uint32_t), 7);

  encoder->setBytes(&ellWidth, sizeof(uint32_t), 8);

  dispatch1D(encoder, impl_->spmvELLPipeline, A.rows());
}

void MetalBackend::encodeDotMetal(MTL::ComputeCommandEncoder *encoder,
                                  const DeviceVector &x, const DeviceVector &y,
                                  DeviceScalar &result) {

  if (x.size() != y.size()) {
    throw std::runtime_error(
        "MetalBackend::encodeDot: vector sizes do not match.");
  }

  if (x.size() == 0) {

    throw std::runtime_error(
        "MetalBackend::encodeDot: vector size 0 is forbidden.");
    // Decide whether zero-length dot should yield 0 or be invalid.
  }
  const std::size_t threadsPerGroup = 256;

  // first pass before the recursive vector reduction
  std::size_t currentSize = x.size();
  std::size_t numberOfGroups =
      (currentSize + threadsPerGroup - 1) / threadsPerGroup;

  ensureReductionScratchCapacity(numberOfGroups);

  std::size_t numberOfThreads = numberOfGroups * threadsPerGroup;

  encoder->setComputePipelineState(impl_->dotPartialPipeline);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBuffer(y.getNativeBuffer(), 0, 1);

  MTL::Buffer *firstOutput = nullptr;

  if (numberOfGroups == 1) {
    firstOutput = result.getNativeBuffer();
  } else {
    firstOutput = impl_->reductionScratchA;
  }

  encoder->setBuffer(firstOutput, 0, 2);

  const uint32_t n = static_cast<uint32_t>(x.size());

  encoder->setBytes(&n, sizeof(uint32_t), 3);

  encoder->setThreadgroupMemoryLength(threadsPerGroup * sizeof(float), 0);

  encoder->dispatchThreads(MTL::Size(numberOfThreads, 1, 1),
                           MTL::Size(threadsPerGroup, 1, 1));
  // that's fine for a single step. Now show the next steps in here
  currentSize = numberOfGroups;
  bool inputIsA = true;

  while (currentSize > 1) {

    numberOfGroups = (currentSize + threadsPerGroup - 1) / threadsPerGroup;
    numberOfThreads = numberOfGroups * threadsPerGroup;

    encoder->setComputePipelineState(impl_->dotReducePipeline);

    MTL::Buffer *inputBuffer = nullptr;
    MTL::Buffer *outputBuffer = nullptr;
    if (inputIsA) {
      inputBuffer = impl_->reductionScratchA;

      if (numberOfGroups == 1) {
        outputBuffer = result.getNativeBuffer();
      } else {
        outputBuffer = impl_->reductionScratchB;
      }
    } else {
      inputBuffer = impl_->reductionScratchB;

      if (numberOfGroups == 1) {
        outputBuffer = result.getNativeBuffer();
      } else {
        outputBuffer = impl_->reductionScratchA;
      }
    }

    encoder->setBuffer(inputBuffer, 0, 0);

    encoder->setBuffer(outputBuffer, 0, 1);

    const uint32_t currentN = static_cast<uint32_t>(currentSize);

    encoder->setBytes(&currentN, sizeof(uint32_t), 2);

    encoder->setThreadgroupMemoryLength(threadsPerGroup * sizeof(float), 0);

    encoder->dispatchThreads(MTL::Size(numberOfThreads, 1, 1),
                             MTL::Size(threadsPerGroup, 1, 1));

    currentSize = numberOfGroups;

    inputIsA = !inputIsA;
  }
}

void MetalBackend::encodeScalarDivideMetal(MTL::ComputeCommandEncoder *encoder,
                                           const DeviceScalar &numerator,
                                           const DeviceScalar &denominator,
                                           DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarDividePipeline);

  encoder->setBuffer(numerator.getNativeBuffer(), 0, 0);

  encoder->setBuffer(denominator.getNativeBuffer(), 0, 1);

  encoder->setBuffer(result.getNativeBuffer(), 0, 2);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::encodeScalarMultiplyMetal(
    MTL::ComputeCommandEncoder *encoder, const DeviceScalar &factor1,
    const DeviceScalar &factor2, DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarMultiplyPipeline);

  encoder->setBuffer(factor1.getNativeBuffer(), 0, 0);

  encoder->setBuffer(factor2.getNativeBuffer(), 0, 1);

  encoder->setBuffer(result.getNativeBuffer(), 0, 2);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::encodeScalarNegateMetal(MTL::ComputeCommandEncoder *encoder,
                                           const DeviceScalar &input,
                                           DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarNegatePipeline);

  encoder->setBuffer(input.getNativeBuffer(), 0, 0);

  encoder->setBuffer(result.getNativeBuffer(), 0, 1);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::encodeScalarSqrtMetal(MTL::ComputeCommandEncoder *encoder,
                                         const DeviceScalar &input,
                                         DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarSqrtPipeline);

  encoder->setBuffer(input.getNativeBuffer(), 0, 0);

  encoder->setBuffer(result.getNativeBuffer(), 0, 1);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::encodeScalarSetMetal(MTL::ComputeCommandEncoder *encoder,
                                        float input, DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarSetPipeline);

  encoder->setBytes(&input, sizeof(float), 0);

  encoder->setBuffer(result.getNativeBuffer(), 0, 1);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::encodeScalarCopyMetal(MTL::ComputeCommandEncoder *encoder,
                                         const DeviceScalar &input,
                                         DeviceScalar &result) {
  encoder->setComputePipelineState(impl_->scalarCopyPipeline);

  encoder->setBuffer(input.getNativeBuffer(), 0, 0);

  encoder->setBuffer(result.getNativeBuffer(), 0, 1);

  encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void MetalBackend::submit(BackendEncoder &encoder) {
  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  metalEncoder.encoder_->endEncoding();

  metalEncoder.commandBuffer_->commit();
}

void MetalBackend::submitAndWait(BackendEncoder &encoder) {
  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  metalEncoder.encoder_->endEncoding();

  metalEncoder.commandBuffer_->commit();

  metalEncoder.commandBuffer_->waitUntilCompleted();
}
// now do the generic encode methods that are overwritten

void MetalBackend::encodeScale(BackendEncoder &encoder, DeviceVector &x,
                               float scalar) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);
  const float alp = scalar;
  encodeScaleMetal(metalEncoder.encoder_, x, alp);
}

void MetalBackend::encodeScale(BackendEncoder &encoder, DeviceVector &x,
                               const DeviceScalar &scalar) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);
  encodeScaleMetal(metalEncoder.encoder_, x, scalar);
}

void MetalBackend::encodeCopy(BackendEncoder &encoder, const DeviceVector &x,
                              DeviceVector &xCopy) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);
  encodeCopyMetal(metalEncoder.encoder_, x, xCopy);
}

void MetalBackend::encodeAxpy(BackendEncoder &encoder, float alpha,
                              const DeviceVector &x, DeviceVector &y) {
  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeAxpyMetal(metalEncoder.encoder_, alpha, x, y);
}

void MetalBackend::encodeAxpy(BackendEncoder &encoder,
                              const DeviceScalar &alpha, const DeviceVector &x,
                              DeviceVector &y) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeAxpyMetal(metalEncoder.encoder_, alpha, x, y);
}

void MetalBackend::encodeSpmv(BackendEncoder &encoder, const DeviceCSRMatrix &A,
                              const DeviceVector &x, DeviceVector &Ax) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeSpmvMetal(metalEncoder.encoder_, A, x, Ax);
}

void MetalBackend::encodeDot(BackendEncoder &encoder, const DeviceVector &x,
                             const DeviceVector &y, DeviceScalar &result) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeDotMetal(metalEncoder.encoder_, x, y, result);
}

void MetalBackend::encodeScalarSet(BackendEncoder &encoder, float value,
                                   DeviceScalar &output) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarSetMetal(metalEncoder.encoder_, value, output);
}

void MetalBackend::encodeScalarCopy(BackendEncoder &encoder,
                                    const DeviceScalar &input,
                                    DeviceScalar &output) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarCopyMetal(metalEncoder.encoder_, input, output);
}

void MetalBackend::encodeScalarDivide(BackendEncoder &encoder,
                                      const DeviceScalar &numerator,
                                      const DeviceScalar &denominator,
                                      DeviceScalar &result) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarDivideMetal(metalEncoder.encoder_, numerator, denominator,
                          result);
}

void MetalBackend::encodeScalarMultiply(BackendEncoder &encoder,
                                        const DeviceScalar &factor1,
                                        const DeviceScalar &factor2,
                                        DeviceScalar &result) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarMultiplyMetal(metalEncoder.encoder_, factor1, factor2, result);
}

void MetalBackend::encodeScalarNegate(BackendEncoder &encoder,
                                      const DeviceScalar &input,
                                      DeviceScalar &output) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarNegateMetal(metalEncoder.encoder_, input, output);
}

void MetalBackend::encodeScalarSqrt(BackendEncoder &encoder,
                                    const DeviceScalar &input,
                                    DeviceScalar &output) {

  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);

  encodeScalarSqrtMetal(metalEncoder.encoder_, input, output);
}
void MetalBackend::encodeSpmvELL(BackendEncoder &encoder,
                                 const DeviceELLMatrix &A,
                                 const DeviceVector &x, DeviceVector &Ax) {
  MetalEncoder &metalEncoder = static_cast<MetalEncoder &>(encoder);
  encodeSpmvELLMetal(metalEncoder.encoder_, A, x, Ax);
}

// carry out the actual operations
void MetalBackend::scale(DeviceVector &x, float scalar) {

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();
  BackendEncoder *encoder = createEncoder();

  // encoder->setComputePipelineState(impl_->scalePipeline);

  encodeScale(*encoder, x, scalar);

  this->submitAndWait(*encoder);
  delete encoder;
}

void MetalBackend::scale(DeviceVector &x, const DeviceScalar &scalar) {

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  // encoder->setComputePipelineState(impl_->scalePipeline);

  encodeScale(*encoder, x, scalar);

  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
  this->submitAndWait(*encoder);
  delete encoder;
}

void MetalBackend::axpy(float alpha, const DeviceVector &x, DeviceVector &y) {

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeAxpy(*encoder, alpha, x, y);

  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();

  this->submitAndWait(*encoder);
  delete encoder;
}

void MetalBackend::axpy(const DeviceScalar &alpha, const DeviceVector &x,
                        DeviceVector &y) {

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeAxpy(*encoder, alpha, x, y);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmv(const DeviceCSRMatrix &A, const DeviceVector &x,
                        DeviceVector &Ax) {

  if (A.cols() != x.size()) {
    throw std::runtime_error(
        "MetalBackend::spmv: matrix columns do not match x size.");
  }

  if (A.rows() != Ax.size()) {
    throw std::runtime_error(
        "MetalBackend::spmv: matrix rows do not match output size.");
  }

  if (A.rows() == 0) {
    return;
  }

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();
  //
  // encoder->setComputePipelineState(impl_->spmvPipeline);

  // MTL::Buffer *bufferRowPtr = A.getRowPtrBuffer();
  // MTL::Buffer *bufferColPtr = A.getColPtrBuffer();
  // MTL::Buffer *bufferValPtr = A.getValPtrBuffer();
  //
  // MTL::Buffer *bufferXvals = x.getNativeBuffer();
  // MTL::Buffer *bufferAxVals = Ax.getNativeBuffer();
  // std::size_t nbOfRows = A.rows();

  // todo sanity checks
  //
  // set the buffers on the encoder now. Check that they match the order of the
  // kernel buffers

  BackendEncoder *encoder = createEncoder();
  encodeSpmv(*encoder, A, x, Ax);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  //
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmvRepeated(const DeviceCSRMatrix &A, const DeviceVector &x,
                                DeviceVector &Ax, std::size_t repetitions) {

  BackendEncoder *encoder = createEncoder();
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    encodeSpmv(*encoder, A, x, Ax);
  }

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmv(const DeviceELLMatrix &A, const DeviceVector &x,
                        DeviceVector &y) {
  if (A.cols() != x.size()) {
    throw std::runtime_error("MetalBackend::spmv ELL: matrix and input vector "
                             "dimensions do not match.");
  }

  if (A.rows() != y.size()) {
    throw std::runtime_error("MetalBackend::spmv ELL: matrix and output vector "
                             "dimensions do not match.");
  }

  if (A.rows() == 0) {
    return;
  }

  BackendEncoder *encoder = createEncoder();
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  encodeSpmvELL(*encoder, A, x, y);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmvRepeated(const DeviceELLMatrix &A, const DeviceVector &x,
                                DeviceVector &y, std::size_t repetitions) {
  if (repetitions <= 0) {
    return;
  }

  if (A.cols() != x.size()) {
    throw std::runtime_error("MetalBackend::spmvRepeated ELL: matrix and input "
                             "vector dimensions do not match.");
  }

  if (A.rows() != y.size()) {
    throw std::runtime_error("MetalBackend::spmvRepeated ELL: matrix and "
                             "output vector dimensions do not match.");
  }

  if (A.rows() == 0) {
    return;
  }

  BackendEncoder *encoder = createEncoder();
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    encodeSpmvELL(*encoder, A, x, y);
  }

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  //
  // // One synchronization for the entire batch.
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::dot(const DeviceVector &x, const DeviceVector &y,
                       DeviceScalar &result) {

  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer(); MTL::ComputeCommandEncoder
  // *encoder = commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeDot(*encoder, x, y, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarSet(float value, DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();
  BackendEncoder *encoder = createEncoder();

  encodeScalarSet(*encoder, value, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarCopy(const DeviceScalar &input, DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeScalarCopy(*encoder, input, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarDivide(const DeviceScalar &numerator,
                                const DeviceScalar &denominator,
                                DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeScalarDivide(*encoder, numerator, denominator, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarMultiply(const DeviceScalar &factor1,
                                  const DeviceScalar &factor2,
                                  DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeScalarMultiply(*encoder, factor1, factor2, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarNegate(const DeviceScalar &input,
                                DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();
  encodeScalarNegate(*encoder, input, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

void MetalBackend::scalarSqrt(const DeviceScalar &input, DeviceScalar &result) {
  // MTL::CommandBuffer *commandBuffer =
  // impl_->context.queue()->commandBuffer();
  //
  // MTL::ComputeCommandEncoder *encoder =
  // commandBuffer->computeCommandEncoder();

  BackendEncoder *encoder = createEncoder();

  encodeScalarSqrt(*encoder, input, result);

  this->submitAndWait(*encoder);
  delete encoder;
  // encoder->endEncoding();
  //
  // commandBuffer->commit();
  // commandBuffer->waitUntilCompleted();
}

DeviceCSRMatrix *MetalBackend::createCSRMatrix(const HostCSRMatrix &matrix) {

  DeviceCSRMatrix *csr = new DeviceCSRMatrix(this->impl_->context, matrix);

  return csr;
}

DeviceVector *MetalBackend::createVector(std::size_t size) {
  DeviceVector *devVec = new DeviceVector(this->impl_->context, size);
  return devVec;
}

DeviceVector *MetalBackend::createVector(std::size_t size,
                                         const float *values) {

  DeviceVector *devVec = new DeviceVector(this->impl_->context, size, values);
  return devVec;
}

DeviceScalar *MetalBackend::createScalar() {
  DeviceScalar* devSca = new DeviceScalar(this->impl_->context);

  return devSca;

}

DeviceScalar *MetalBackend::createScalar(float value) {
  DeviceScalar* devSca = new DeviceScalar(this->impl_->context, value);
  return devSca;

}

} // namespace gpuSolver
