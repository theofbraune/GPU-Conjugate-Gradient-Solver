#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceScalar.h>
#include <GPUSolver/BackendEncoder.h>

#include <gtest/gtest.h>

#include <memory>


TEST(CudaScalarOps, Set)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar value(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarSet(
      *encoder,
      7.25f,
      value);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      value.download(),
      7.25f,
      1.0e-6f);
}


TEST(CudaScalarOps, Copy)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar input(
      -8.5f);

  gpuSolver::CudaDeviceScalar output(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarCopy(
      *encoder,
      input,
      output);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      output.download(),
      -8.5f,
      1.0e-6f);
}


TEST(CudaScalarOps, Negate)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar input(
      3.75f);

  gpuSolver::CudaDeviceScalar output(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarNegate(
      *encoder,
      input,
      output);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      output.download(),
      -3.75f,
      1.0e-6f);
}



TEST(CudaScalarOps, Multiply)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar a(
      3.0f);

  gpuSolver::CudaDeviceScalar b(
      -4.0f);

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarMultiply(
      *encoder,
      a,
      b,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      -12.0f,
      1.0e-6f);
}


TEST(CudaScalarOps, Divide)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar numerator(
      15.0f);

  gpuSolver::CudaDeviceScalar denominator(
      4.0f);

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarDivide(
      *encoder,
      numerator,
      denominator,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      3.75f,
      1.0e-6f);
}


TEST(CudaScalarOps, Sqrt)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar input(
      20.25f);

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScalarSqrt(
      *encoder,
      input,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      4.5f,
      1.0e-6f);
}

TEST(CudaScalarOps, EntireScalarChainStaysOnGpu)
{
  gpuSolver::CudaBackend backend;

  gpuSolver::CudaDeviceScalar a(
      12.0f);

  gpuSolver::CudaDeviceScalar b(
      3.0f);

  gpuSolver::CudaDeviceScalar quotient(
      0.0f);

  gpuSolver::CudaDeviceScalar product(
      0.0f);

  gpuSolver::CudaDeviceScalar root(
      0.0f);

  gpuSolver::CudaDeviceScalar result(
      0.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  // quotient = 12 / 3 = 4
  backend.encodeScalarDivide(
      *encoder,
      a,
      b,
      quotient);

  // product = 4 * 3 = 12
  backend.encodeScalarMultiply(
      *encoder,
      quotient,
      b,
      product);

  // root = sqrt(12)
  backend.encodeScalarSqrt(
      *encoder,
      product,
      root);

  // result = -sqrt(12)
  backend.encodeScalarNegate(
      *encoder,
      root,
      result);

  backend.submitAndWait(
      *encoder);

  EXPECT_NEAR(
      result.download(),
      -std::sqrt(12.0f),
      1.0e-5f);
}
