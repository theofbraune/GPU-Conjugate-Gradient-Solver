#include "GPUSolver/DeviceScalar.h"
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
  MTL::ComputePipelineState *spmvELLPipeline = nullptr;

  // now the dataa needed for the dot pipeline
  MTL::ComputePipelineState *dotPartialPipeline = nullptr;
  MTL::Buffer *reductionScratchA = nullptr;
  MTL::Buffer *reductionScratchB = nullptr;
  MTL::ComputePipelineState *dotReducePipeline = nullptr;

  std::size_t reductionScratchCapacity = 0;

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

  impl_->scalePipeline =
      makePipeline(context.device(), context.library(), "scale");

  impl_->spmvPipeline =
      makePipeline(context.device(), context.library(), "spmv");

  impl_->spmvELLPipeline =
      makePipeline(context.device(), context.library(), "spmvELL");

  impl_->dotPartialPipeline =
      makePipeline(context.device(), context.library(), "dotPartial");

  impl_->dotReducePipeline =
      makePipeline(context.device(), context.library(), "reduceSumPartial");
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
void MetalBackend::encodeScale(MTL::ComputeCommandEncoder *encoder,
                               DeviceVector &x, const float alpha) {

  encoder->setComputePipelineState(impl_->scalePipeline);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBytes(&alpha, sizeof(float), 1);

  dispatch1D(encoder, impl_->scalePipeline, x.size());
}

void MetalBackend::encodeScale(MTL::ComputeCommandEncoder *encoder,
                               DeviceVector &x, const DeviceScalar &alpha) {

  encoder->setComputePipelineState(impl_->scalePipeline);

  encoder->setBuffer(x.getNativeBuffer(), 0, 0);

  encoder->setBuffer(alpha.getNativeBuffer(), 0, 1);

  dispatch1D(encoder, impl_->scalePipeline, x.size());
}

void MetalBackend::encodeAxpy(MTL::ComputeCommandEncoder *encoder,
                              const DeviceScalar &alpha, const DeviceVector &x,
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

  // encoder->setBytes(&alpha, sizeof(float), 2);
  encoder->setBuffer(alpha.getNativeBuffer(), 0, 2);

  dispatch1D(encoder, impl_->axpyPipeline, sizeOfx);
}

void MetalBackend::encodeAxpy(MTL::ComputeCommandEncoder *encoder, float alpha,
                              const DeviceVector &x, DeviceVector &y) {

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

void MetalBackend::encodeSpmv(MTL::ComputeCommandEncoder *encoder,
                              const DeviceCSRMatrix &A, const DeviceVector &x,
                              DeviceVector &Ax) {
  encoder->setComputePipelineState(impl_->spmvPipeline);

  encoder->setBuffer(A.getRowPtrBuffer(), 0, 0);

  encoder->setBuffer(A.getColPtrBuffer(), 0, 1);

  encoder->setBuffer(A.getValPtrBuffer(), 0, 2);

  encoder->setBuffer(x.getNativeBuffer(), 0, 3);

  encoder->setBuffer(Ax.getNativeBuffer(), 0, 4);

  dispatch1D(encoder, impl_->spmvPipeline, A.rows());
}

void MetalBackend::encodeSpmvELL(MTL::ComputeCommandEncoder *encoder,
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

void MetalBackend::encodeDot(MTL::ComputeCommandEncoder *encoder,
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

// carry out the actual operations
void MetalBackend::scale(DeviceVector &x, float scalar) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->scalePipeline);

  encodeScale(encoder, x, scalar);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}


void MetalBackend::scale(DeviceVector &x, const DeviceScalar& scalar) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(impl_->scalePipeline);

  encodeScale(encoder, x, scalar);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::axpy(float alpha, const DeviceVector &x, DeviceVector &y) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encodeAxpy(encoder, alpha, x, y);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::axpy(const DeviceScalar& alpha, const DeviceVector &x, DeviceVector &y) {

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encodeAxpy(encoder, alpha, x, y);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
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

  encodeSpmv(encoder, A, x, Ax);

  encoder->endEncoding();

  commandBuffer->commit();

  commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmvRepeated(const DeviceCSRMatrix &A, const DeviceVector &x,
                                DeviceVector &Ax, std::size_t repetitions) {
  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    encodeSpmv(encoder, A, x, Ax);
  }

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
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

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encodeSpmvELL(encoder, A, x, y);

  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void MetalBackend::spmvRepeated(const DeviceELLMatrix &A, const DeviceVector &x,
                                DeviceVector &y, int repetitions) {
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

  MTL::CommandBuffer *commandBuffer = impl_->context.queue()->commandBuffer();

  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  for (int iteration = 0; iteration < repetitions; ++iteration) {
    encodeSpmvELL(encoder, A, x, y);
  }

  encoder->endEncoding();

  commandBuffer->commit();

  // One synchronization for the entire batch.
  commandBuffer->waitUntilCompleted();
}


void MetalBackend::dot(const DeviceVector& x, const DeviceVector& y, DeviceScalar& result){

  MTL::CommandBuffer* commandBuffer = impl_->context.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

  encodeDot(encoder,x,y,result);

  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

}

} // namespace gpuSolver
