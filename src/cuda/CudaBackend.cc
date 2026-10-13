#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceCSRMatrix.h>
#include <GPUSolver/cuda/CudaDeviceIndexVector.h>
#include <GPUSolver/cuda/CudaDeviceScalar.h>
#include <GPUSolver/cuda/CudaDeviceVector.h>
#include <GPUSolver/cuda/CudaEncoder.h>
#include <GPUSolver/cuda/CudaHelper.h>
#include <GPUSolver/cuda/CudaKernels.h>
#include <GPUSolver/cuda/GaussSeidelKernels.h>
#include <GPUSolver/cuda/ScalarKernels.h>
#include <GPUSolver/cuda/VectorKernels.h>
#include <GPUSolver/cuda/BlockKernels.h>

#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

namespace gpuSolver {

CudaBackend::CudaBackend()
    : cusparseHandle_(nullptr)
    , cublasHandle_(nullptr)
{
    if (cublasCreate_v2(&cublasHandle_) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle.");
    }
    const cusparseStatus_t status = cusparseCreate(&cusparseHandle_);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseCreate failed.");
    }
    int deviceCount = 0;

    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    if (deviceCount <= 0) {
        throw std::runtime_error("No CUDA device found.");
    }
}

CudaBackend::~CudaBackend()
{
    if (cusparseHandle_ != nullptr) {
        cusparseDestroy(cusparseHandle_);
    }
    if (cublasHandle_ != nullptr) {
        cublasDestroy_v2(cublasHandle_);
    }
}

void CudaBackend::timesTwo(const std::vector<float>& input, std::vector<float>& output) const
{
    if (input.empty()) {
        output.clear();
        return;
    }

    output.resize(input.size());

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;

    const std::size_t bytes = input.size() * sizeof(float);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), bytes), "cudaMalloc input failed");

    try {
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&deviceOutput), bytes),
            "cudaMalloc output failed"
        );

        checkCuda(
            cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device failed"
        );

        cuda::launchTimesTwo(deviceInput, deviceOutput, input.size());

        checkCuda(cudaDeviceSynchronize(), "CUDA kernel execution failed");

        checkCuda(
            cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host failed"
        );

        cudaFree(deviceOutput);
        cudaFree(deviceInput);

    } catch (...) {
        if (deviceOutput != nullptr) {
            cudaFree(deviceOutput);
        }

        if (deviceInput != nullptr) {
            cudaFree(deviceInput);
        }

        throw;
    }
}

BackendEncoder* CudaBackend::createEncoder()
{
    return new CudaEncoder();
}

void CudaBackend::submitAndWait(BackendEncoder& encoder)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    cuda::check(cudaStreamSynchronize(cudaEncoder.stream()), "CUDA stream execution failed");
}

void CudaBackend::submit(BackendEncoder& encoder)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    cuda::check(cudaStreamSynchronize(cudaEncoder.stream()), "CUDA stream execution failed");
}

DeviceVector* CudaBackend::createVector(std::size_t size)
{
    return new CudaDeviceVector(size);
}

DeviceVector* CudaBackend::createVector(std::size_t size, const float* values)
{
    DeviceVector* devVec = new CudaDeviceVector(size);
    devVec->updateValues(values, size);
    return devVec;
}

DeviceIndexVector* CudaBackend::createIndexVector(std::size_t size)
{
    return new CudaDeviceIndexVector(size);
}

DeviceIndexVector* CudaBackend::createIndexVector(std::size_t size, const int* values)
{
    DeviceIndexVector* devVec = new CudaDeviceIndexVector(size);
    devVec->updateValues(values, size);
    return devVec;
}

DeviceScalar* CudaBackend::createScalar()
{
    return new CudaDeviceScalar();
}

DeviceScalar* CudaBackend::createScalar(float value)
{
    return new CudaDeviceScalar(value);
}

DeviceCSRMatrix* CudaBackend::createCSRMatrix(const HostCSRMatrix& matrix)
{
    // throw std::runtime_error("not implemented yet");
    std::size_t nRows = matrix.rows();
    std::size_t nCols = matrix.cols();
    std::size_t nnz = matrix.nnz();
    const int* rowPtr = matrix.activeRowPtr();
    const int* colPtr = matrix.activeColPtr();
    const float* valPtr = matrix.activeValPtr();

    DeviceCSRMatrix* devMat = new CudaDeviceCSRMatrix(nRows, nCols, nnz, rowPtr, colPtr, valPtr);

    return devMat;
}

DeviceELLMatrix* CudaBackend::createELLMatrix(const HostELLMatrix& matrix)
{
    throw std::runtime_error("not implemented yet");
}

//-------------------------------------------------------------
//-------------------------------------------------------------
//-------------------------------------------------------------

void CudaBackend::encodeScale(BackendEncoder& encoder, DeviceVector& x, float alpha)
{
    if (x.size() == 0) {
        return;
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceVector& cudaX = static_cast<CudaDeviceVector&>(x);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_HOST);

    const cublasStatus_t status =
        cublasSscal(cublasHandle_, static_cast<int>(x.size()), &alpha, cudaX.data(), 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSscal failed.");
    }
}

void CudaBackend::encodeScale(BackendEncoder& encoder, DeviceVector& x, const DeviceScalar& alpha)
{
    if (x.size() == 0) {
        return;
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceVector& cudaX = static_cast<CudaDeviceVector&>(x);

    const CudaDeviceScalar& cudaAlpha = static_cast<const CudaDeviceScalar&>(alpha);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_DEVICE);

    const cublasStatus_t status =
        cublasSscal(cublasHandle_, static_cast<int>(x.size()), cudaAlpha.data(), cudaX.data(), 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSscal with device alpha failed.");
    }
}

void CudaBackend::encodeCopy(
    BackendEncoder& encoder,
    const DeviceVector& source,
    DeviceVector& destination
)
{
    if (source.size() != destination.size()) {
        throw std::runtime_error("CudaBackend::encodeCopy: vector size mismatch.");
    }

    if (source.size() == 0) {
        return;
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaSource = static_cast<const CudaDeviceVector&>(source);

    CudaDeviceVector& cudaDestination = static_cast<CudaDeviceVector&>(destination);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    const cublasStatus_t status = cublasScopy(
        cublasHandle_,
        static_cast<int>(source.size()),
        cudaSource.data(),
        1,
        cudaDestination.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasScopy failed.");
    }
}

void CudaBackend::encodeScaleVector(
    BackendEncoder& encoder,
    const DeviceVector& scaleVector,
    DeviceVector& output
)
{
    if (scaleVector.size() != output.size()) {
        throw std::runtime_error("CudaBackend::encodeScaleVector: vector size mismatch.");
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaScaleVector = static_cast<const CudaDeviceVector&>(scaleVector);

    CudaDeviceVector& cudaOutput = static_cast<CudaDeviceVector&>(output);

    cuda::launchScaleVector(
        cudaScaleVector.data(),
        cudaOutput.data(),
        output.size(),
        cudaEncoder.stream()
    );
}

void CudaBackend::encodeSetZero(BackendEncoder& encoder, DeviceVector& input)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceVector& cudaIn = static_cast<CudaDeviceVector&>(input);

    cuda::launchSetZero(cudaIn.data(), cudaIn.size(), cudaEncoder.stream());
}

void CudaBackend::encodeAxpy(
    BackendEncoder& encoder,
    float alpha,
    const DeviceVector& x,
    DeviceVector& y
)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("CudaBackend::encodeAxpy: vector size mismatch.");
    }

    if (x.size() == 0) {
        return;
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);

    CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(y);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_HOST);

    const cublasStatus_t status = cublasSaxpy(
        cublasHandle_,
        static_cast<int>(x.size()),
        &alpha,
        cudaX.data(),
        1,
        cudaY.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSaxpy failed.");
    }
}

void CudaBackend::encodeAxpy(
    BackendEncoder& encoder,
    const DeviceScalar& alpha,
    const DeviceVector& x,
    DeviceVector& y
)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("CudaBackend::encodeAxpy: vector size mismatch.");
    }

    if (x.size() == 0) {
        return;
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaAlpha = static_cast<const CudaDeviceScalar&>(alpha);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);

    CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(y);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_DEVICE);

    const cublasStatus_t status = cublasSaxpy(
        cublasHandle_,
        static_cast<int>(x.size()),
        cudaAlpha.data(),
        cudaX.data(),
        1,
        cudaY.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSaxpy with device alpha failed.");
    }
}


void CudaBackend::encodeSpmv(
    BackendEncoder& encoder,
    const DeviceCSRMatrix& A,
    const DeviceVector& x,
    DeviceVector& y
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceCSRMatrix& cudaMatrix = static_cast<const CudaDeviceCSRMatrix&>(A);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);

    CudaDeviceVector& cudaY = static_cast<CudaDeviceVector&>(y);

    cusparseSetStream(cusparseHandle_, cudaEncoder.stream());

    std::size_t bufferSize = 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::size_t requiredSize = 0;
    cusparseStatus_t status = cusparseSpMV_bufferSize(
        cusparseHandle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        cudaMatrix.descriptor(),
        cudaX.descriptor(),
        &beta,
        cudaY.descriptor(),
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &requiredSize
    );

    cudaMatrix.ensureSpmvBuffer(requiredSize);

    status = cusparseSpMV(
        cusparseHandle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        cudaMatrix.descriptor(),
        cudaX.descriptor(),
        &beta,
        cudaY.descriptor(),
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        cudaMatrix.spmvBuffer()
    );
}

void CudaBackend::encodeGaussSeidelColor(
    BackendEncoder& encoder,
    const DeviceCSRMatrix& matrix,
    const DeviceIndexVector& colorVertices,
    std::size_t colorStart,
    std::size_t colorCount,
    const DeviceVector& rhs,
    DeviceVector& solution,
    float omega
)
{
    if (colorCount == 0) {
        return;
    }

    if (matrix.rows() != rhs.size()) {
        throw std::runtime_error(
            "CudaBackend::encodeGaussSeidelColor: "
            "matrix/rhs size mismatch."
        );
    }

    if (matrix.cols() != solution.size()) {
        throw std::runtime_error(
            "CudaBackend::encodeGaussSeidelColor: "
            "matrix/solution size mismatch."
        );
    }

    if (colorStart + colorCount > colorVertices.size()) {
        throw std::runtime_error(
            "CudaBackend::encodeGaussSeidelColor: "
            "color range exceeds colorVertices size."
        );
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceCSRMatrix& cudaMatrix = static_cast<const CudaDeviceCSRMatrix&>(matrix);

    const CudaDeviceIndexVector& cudaColorVertices =
        static_cast<const CudaDeviceIndexVector&>(colorVertices);

    const CudaDeviceVector& cudaRhs = static_cast<const CudaDeviceVector&>(rhs);

    CudaDeviceVector& cudaSolution = static_cast<CudaDeviceVector&>(solution);

    cuda::launchGaussSeidelColor(
        cudaMatrix.rowPtr(),
        cudaMatrix.colIdx(),
        cudaMatrix.values(),
        cudaColorVertices.data(),
        colorStart,
        colorCount,
        cudaRhs.data(),
        cudaSolution.data(),
        omega,
        cudaEncoder.stream()
    );
}

void CudaBackend::encodeDot(
    BackendEncoder& encoder,
    const DeviceVector& x,
    const DeviceVector& y,
    DeviceScalar& result
)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("CudaBackend::encodeDot: vector size mismatch.");
    }

    if (x.size() == 0) {
        throw std::runtime_error("CudaBackend::encodeDot: zero-length vector.");
    }

    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceVector& cudaX = static_cast<const CudaDeviceVector&>(x);

    const CudaDeviceVector& cudaY = static_cast<const CudaDeviceVector&>(y);

    CudaDeviceScalar& cudaResult = static_cast<CudaDeviceScalar&>(result);

    cublasSetStream(cublasHandle_, cudaEncoder.stream());

    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_DEVICE);

    const cublasStatus_t status = cublasSdot(
        cublasHandle_,
        static_cast<int>(x.size()),
        cudaX.data(),
        1,
        cudaY.data(),
        1,
        cudaResult.data()
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSdot failed.");
    }
}

void CudaBackend::encodeScalarDivide(
    BackendEncoder& encoder,
    const DeviceScalar& numerator,
    const DeviceScalar& denominator,
    DeviceScalar& result
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaNumerator = static_cast<const CudaDeviceScalar&>(numerator);

    const CudaDeviceScalar& cudaDenominator = static_cast<const CudaDeviceScalar&>(denominator);

    CudaDeviceScalar& cudaResult = static_cast<CudaDeviceScalar&>(result);

    cuda::launchScalarDivide(
        cudaNumerator.data(),
        cudaDenominator.data(),
        cudaResult.data(),
        cudaEncoder.stream()
    );
}

void CudaBackend::encodeScalarMultiply(
    BackendEncoder& encoder,
    const DeviceScalar& a,
    const DeviceScalar& b,
    DeviceScalar& result
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaA = static_cast<const CudaDeviceScalar&>(a);

    const CudaDeviceScalar& cudaB = static_cast<const CudaDeviceScalar&>(b);

    CudaDeviceScalar& cudaResult = static_cast<CudaDeviceScalar&>(result);

    cuda::launchScalarMultiply(cudaA.data(), cudaB.data(), cudaResult.data(), cudaEncoder.stream());
}

void CudaBackend::encodeScalarSqrt(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& result
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaInput = static_cast<const CudaDeviceScalar&>(input);

    CudaDeviceScalar& cudaResult = static_cast<CudaDeviceScalar&>(result);

    cuda::launchScalarSqrt(cudaInput.data(), cudaResult.data(), cudaEncoder.stream());
}

void CudaBackend::encodeScalarSet(BackendEncoder& encoder, float value, DeviceScalar& output)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    CudaDeviceScalar& cudaOutput = static_cast<CudaDeviceScalar&>(output);

    cuda::launchScalarSet(value, cudaOutput.data(), cudaEncoder.stream());
}
void CudaBackend::encodeScalarCopy(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& output
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaInput = static_cast<const CudaDeviceScalar&>(input);

    CudaDeviceScalar& cudaOutput = static_cast<CudaDeviceScalar&>(output);

    cuda::check(
        cudaMemcpyAsync(
            cudaOutput.data(),
            cudaInput.data(),
            sizeof(float),
            cudaMemcpyDeviceToDevice,
            cudaEncoder.stream()
        ),
        "CudaBackend::encodeScalarCopy failed"
    );
}

void CudaBackend::encodeScalarNegate(
    BackendEncoder& encoder,
    const DeviceScalar& input,
    DeviceScalar& output
)
{
    CudaEncoder& cudaEncoder = static_cast<CudaEncoder&>(encoder);

    const CudaDeviceScalar& cudaInput = static_cast<const CudaDeviceScalar&>(input);

    CudaDeviceScalar& cudaOutput = static_cast<CudaDeviceScalar&>(output);

    cuda::launchScalarNegate(cudaInput.data(), cudaOutput.data(), cudaEncoder.stream());
}
void CudaBackend::encodeApplyBlockInverses3x3(
    BackendEncoder &encoder,
    const DeviceVector &inverseDiagBlocks,
    const DeviceVector &residual,
    DeviceVector &correction)
{
  if (residual.size() != correction.size()) {
    throw std::runtime_error(
        "CudaBackend::encodeApplyBlockInverses3x3: "
        "residual/correction size mismatch.");
  }

  if (residual.size() % 3 != 0) {
    throw std::runtime_error(
        "CudaBackend::encodeApplyBlockInverses3x3: "
        "vector size must be divisible by 3.");
  }

  const std::size_t numberOfBlocks =
      residual.size() / 3;

  if (inverseDiagBlocks.size()
      != 9 * numberOfBlocks) {
    throw std::runtime_error(
        "CudaBackend::encodeApplyBlockInverses3x3: "
        "wrong inverse block array size.");
  }

  CudaEncoder &cudaEncoder =
      static_cast<CudaEncoder &>(encoder);

  const CudaDeviceVector &cudaInverseDiagBlocks =
      static_cast<const CudaDeviceVector &>(
          inverseDiagBlocks);

  const CudaDeviceVector &cudaResidual =
      static_cast<const CudaDeviceVector &>(
          residual);

  CudaDeviceVector &cudaCorrection =
      static_cast<CudaDeviceVector &>(
          correction);

  cuda::launchApplyBlockInverses3x3(
      cudaInverseDiagBlocks.data(),
      cudaResidual.data(),
      cudaCorrection.data(),
      numberOfBlocks,
      cudaEncoder.stream());
}
void CudaBackend::encodeBlockGaussSeidelColor3x3(
    BackendEncoder &encoder,
    const DeviceCSRMatrix &matrix,
    const DeviceIndexVector &colorBlocks,
    const DeviceVector &diagonalInverses,
    std::size_t begin,
    std::size_t end,
    const DeviceVector &rhs,
    DeviceVector &solution,
    float omega)
{
  if (end < begin) {
    throw std::runtime_error(
        "CudaBackend::encodeBlockGaussSeidelColor3x3: "
        "invalid color range.");
  }

  if (end > colorBlocks.size()) {
    throw std::runtime_error(
        "CudaBackend::encodeBlockGaussSeidelColor3x3: "
        "color range exceeds colorBlocks size.");
  }

  if (rhs.size() != solution.size()) {
    throw std::runtime_error(
        "CudaBackend::encodeBlockGaussSeidelColor3x3: "
        "rhs/solution size mismatch.");
  }

  if (rhs.size() % 3 != 0) {
    throw std::runtime_error(
        "CudaBackend::encodeBlockGaussSeidelColor3x3: "
        "vector size must be divisible by 3.");
  }

  const std::size_t numberOfBlocks =
      rhs.size() / 3;

  if (diagonalInverses.size()
      != 9 * numberOfBlocks) {
    throw std::runtime_error(
        "CudaBackend::encodeBlockGaussSeidelColor3x3: "
        "wrong diagonal inverse array size.");
  }

  CudaEncoder &cudaEncoder =
      static_cast<CudaEncoder &>(
          encoder);

  const CudaDeviceCSRMatrix &cudaMatrix =
      static_cast<const CudaDeviceCSRMatrix &>(
          matrix);

  const CudaDeviceIndexVector &cudaColorBlocks =
      static_cast<const CudaDeviceIndexVector &>(
          colorBlocks);

  const CudaDeviceVector &cudaDiagonalInverses =
      static_cast<const CudaDeviceVector &>(
          diagonalInverses);

  const CudaDeviceVector &cudaRhs =
      static_cast<const CudaDeviceVector &>(
          rhs);

  CudaDeviceVector &cudaSolution =
      static_cast<CudaDeviceVector &>(
          solution);

  cuda::launchBlockGaussSeidelColor3x3(
      cudaMatrix.rowPtr(),
      cudaMatrix.colIdx(),
      cudaMatrix.values(),
      cudaColorBlocks.data(),
      cudaDiagonalInverses.data(),
      begin,
      end,
      cudaRhs.data(),
      cudaSolution.data(),
      omega,
      cudaEncoder.stream());
}


void CudaBackend::scale(DeviceVector& x, float alpha)
{
    BackendEncoder* encoder = this->createEncoder();
    encodeScale(*encoder, x, alpha);
    delete encoder;
}

void CudaBackend::scale(DeviceVector& x, const DeviceScalar& alpha)
{

    BackendEncoder* encoder = this->createEncoder();
    encodeScale(*encoder, x, alpha);
    delete encoder;
}

void CudaBackend::scaleVectorByVector(const DeviceVector& scaleVector, DeviceVector& output)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeScale(*encoder, x, alpha);
    encodeScaleVector(*encoder,scaleVector, output);
    delete encoder;
}

void CudaBackend::axpy(float alpha, const DeviceVector& x, DeviceVector& y)
{
    BackendEncoder* encoder = this->createEncoder();
    encodeAxpy(*encoder, alpha, x, y);
    delete encoder;
}

void CudaBackend::axpy(const DeviceScalar& alpha, const DeviceVector& x, DeviceVector& y)
{

    BackendEncoder* encoder = this->createEncoder();
    encodeAxpy(*encoder, alpha, x, y);
    delete encoder;
}

void CudaBackend::spmv(const DeviceCSRMatrix& A, const DeviceVector& x, DeviceVector& y)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    encodeSpmv(*encoder,A, x, y);
    delete encoder;
}

void CudaBackend::dot(const DeviceVector& x, const DeviceVector& y, DeviceScalar& result)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    encodeDot(*encoder, x, y,result);
    delete encoder;
}

void CudaBackend::scalarSet(float value, DeviceScalar& result)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarSet(*encoder,value,result);
    delete encoder;
    throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarCopy(const DeviceScalar& input, DeviceScalar& output)
{
    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarCopy(*encoder,input,output);
    delete encoder;
    // throw std::runtime_error("not implemented yet");
}

void CudaBackend::scalarDivide(
    const DeviceScalar& numerator,
    const DeviceScalar& denominator,
    DeviceScalar& result
)
{
    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarDivide(*encoder,numerator,denominator,result);
    delete encoder;
}

void CudaBackend::scalarMultiply(
    const DeviceScalar& factor1,
    const DeviceScalar& factor2,
    DeviceScalar& result
)
{
    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarMultiply(*encoder,factor1,factor2,result);
    delete encoder;
}

void CudaBackend::scalarNegate(const DeviceScalar& input, DeviceScalar& output)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarNegate(*encoder,input,output);
    delete encoder;
}

void CudaBackend::scalarSqrt(const DeviceScalar& input, DeviceScalar& output)
{

    BackendEncoder* encoder = this->createEncoder();
    // encodeAxpy(*encoder, alpha, x, y);
    // encodeDot(*encoder, x, y,result);
    encodeScalarSqrt(*encoder,input,output);
    delete encoder;
}


} // namespace gpuSolver
