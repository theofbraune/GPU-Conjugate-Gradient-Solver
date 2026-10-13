#include "CudaTestHelpers.h"

#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaDeviceScalar.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>


TEST(CudaVectorOps, Copy)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> values =
      {1.0f, -2.0f, 3.5f, 7.0f};

  std::unique_ptr<gpuSolver::DeviceVector> source(
      backend.createVector(
          values.size(),
          values.data()));

  std::unique_ptr<gpuSolver::DeviceVector> destination(
      backend.createVector(
          values.size()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeCopy(
      *encoder,
      *source,
      *destination);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(
          *destination),
      values);
}


TEST(CudaVectorOps, SetZero)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> values =
      {1.0f, -2.0f, 3.5f, 7.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          values.size(),
          values.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeSetZero(
      *encoder,
      *x);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*x),
      std::vector<float>(
          values.size(),
          0.0f));
}


TEST(CudaVectorOps, ScaleHostScalar)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> values =
      {1.0f, -2.0f, 3.0f, 4.0f};

  const std::vector<float> expected =
      {2.5f, -5.0f, 7.5f, 10.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          values.size(),
          values.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScale(
      *encoder,
      *x,
      2.5f);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*x),
      expected);
}


TEST(CudaVectorOps, ScaleDeviceScalar)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> values =
      {1.0f, -2.0f, 3.0f};

  const std::vector<float> expected =
      {-3.0f, 6.0f, -9.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          values.size(),
          values.data()));

  gpuSolver::CudaDeviceScalar alpha(
      -3.0f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScale(
      *encoder,
      *x,
      alpha);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*x),
      expected);
}


TEST(CudaVectorOps, AxpyHostScalar)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> xHost =
      {1.0f, 2.0f, 3.0f};

  const std::vector<float> yHost =
      {4.0f, 5.0f, 6.0f};

  // y <- 2*x + y
  const std::vector<float> expected =
      {6.0f, 9.0f, 12.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          xHost.size(),
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          yHost.size(),
          yHost.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeAxpy(
      *encoder,
      2.0f,
      *x,
      *y);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*y),
      expected);
}


TEST(CudaVectorOps, AxpyDeviceScalar)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> xHost =
      {1.0f, 2.0f, 3.0f};

  const std::vector<float> yHost =
      {4.0f, 5.0f, 6.0f};

  const std::vector<float> expected =
      {3.5f, 4.0f, 4.5f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          xHost.size(),
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          yHost.size(),
          yHost.data()));

  gpuSolver::CudaDeviceScalar alpha(
      -0.5f);

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeAxpy(
      *encoder,
      alpha,
      *x,
      *y);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*y),
      expected);
}


TEST(CudaVectorOps, ScaleVector)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> scale =
      {2.0f, -1.0f, 0.5f, 4.0f};

  const std::vector<float> initial =
      {3.0f, 5.0f, 8.0f, -2.0f};

  const std::vector<float> expected =
      {6.0f, -5.0f, 4.0f, -8.0f};

  std::unique_ptr<gpuSolver::DeviceVector> scaleVector(
      backend.createVector(
          scale.size(),
          scale.data()));

  std::unique_ptr<gpuSolver::DeviceVector> output(
      backend.createVector(
          initial.size(),
          initial.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  backend.encodeScaleVector(
      *encoder,
      *scaleVector,
      *output);

  backend.submitAndWait(
      *encoder);

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*output),
      expected);
}


TEST(CudaVectorOps, SeveralOperationsWithoutIntermediateSynchronization)
{
  gpuSolver::CudaBackend backend;

  const std::vector<float> xHost =
      {1.0f, 2.0f, 3.0f};

  const std::vector<float> yHost =
      {10.0f, 20.0f, 30.0f};

  std::unique_ptr<gpuSolver::DeviceVector> x(
      backend.createVector(
          xHost.size(),
          xHost.data()));

  std::unique_ptr<gpuSolver::DeviceVector> y(
      backend.createVector(
          yHost.size(),
          yHost.data()));

  std::unique_ptr<gpuSolver::BackendEncoder> encoder(
      backend.createEncoder());

  // x = 2x
  backend.encodeScale(
      *encoder,
      *x,
      2.0f);

  // y = y - x
  backend.encodeAxpy(
      *encoder,
      -1.0f,
      *x,
      *y);

  // y = 0.5 y
  backend.encodeScale(
      *encoder,
      *y,
      0.5f);

  // Exactly one wait here.
  backend.submitAndWait(
      *encoder);

  const std::vector<float> expected =
      {4.0f, 8.0f, 12.0f};

  gpuSolverTest::expectVectorNear(
      gpuSolverTest::downloadVector(*y),
      expected);
}
