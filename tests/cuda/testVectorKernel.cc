#include <GPUSolver/cuda/CudaBackend.h>
#include <GPUSolver/cuda/CudaEncoder.h>
#include <GPUSolver/cuda/CudaDeviceVector.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expectNear(
    float value,
    float expected,
    float tolerance,
    const char *message)
{
  if (std::abs(value - expected) > tolerance) {
    throw std::runtime_error(message);
  }
}


void checkVector(
    const float *values,
    const std::vector<float> &expected,
    const char *message)
{
  constexpr float tolerance = 1.0e-6f;

  for (std::size_t i = 0;
       i < expected.size();
       ++i) {

    if (std::abs(values[i] - expected[i])
        > tolerance) {

      std::cerr
          << message
          << "\n"
          << "  index    = " << i << "\n"
          << "  value    = " << values[i] << "\n"
          << "  expected = " << expected[i] << "\n";

      throw std::runtime_error(message);
    }
  }
}

} // namespace


int main()
{
  try {

    gpuSolver::CudaBackend backend;

    constexpr std::size_t size = 5;

    const std::vector<float> xValues = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f};

    const std::vector<float> yValues = {
        10.0f,
        20.0f,
        30.0f,
        40.0f,
        50.0f};

    gpuSolver::DeviceVector *x =
        backend.createVector(
            size,
            xValues.data());

    gpuSolver::DeviceVector *y =
        backend.createVector(
            size,
            yValues.data());

    // ========================================================
    // 1. SCALE
    //
    // x <- 2 x
    // ========================================================

    {
      gpuSolver::BackendEncoder *encoder =
          backend.createEncoder();

      backend.encodeScale(
          *encoder,
          *x,
          2.0f);

      backend.submitAndWait(
          *encoder);

      delete encoder;

      float *result =
          x->download();

      checkVector(
          result,
          {
              2.0f,
              4.0f,
              6.0f,
              8.0f,
              10.0f
          },
          "scale test failed");

      std::cout
          << "scale: PASS\n";
    }

    // ========================================================
    // 2. COPY
    //
    // y <- x
    // ========================================================

    {
      gpuSolver::BackendEncoder *encoder =
          backend.createEncoder();

      backend.encodeCopy(
          *encoder,
          *x,
          *y);

      backend.submitAndWait(
          *encoder);

      delete encoder;

      float *result =
          y->download();

      checkVector(
          result,
          {
              2.0f,
              4.0f,
              6.0f,
              8.0f,
              10.0f
          },
          "copy test failed");

      std::cout
          << "copy: PASS\n";
    }

    // ========================================================
    // 3. AXPY
    //
    // y <- y + 3 x
    //
    // Since y == x before this operation:
    //
    // y <- x + 3x = 4x
    // ========================================================

    {
      gpuSolver::BackendEncoder *encoder =
          backend.createEncoder();

      backend.encodeAxpy(
          *encoder,
          3.0f,
          *x,
          *y
          );

      backend.submitAndWait(
          *encoder);

      delete encoder;

      float *result =
          y->download();

      checkVector(
          result,
          {
              8.0f,
              16.0f,
              24.0f,
              32.0f,
              40.0f
          },
          "axpy test failed");

      std::cout
          << "axpy: PASS\n";
    }

    // ========================================================
    // 4. SET ZERO
    // ========================================================

    {
      gpuSolver::BackendEncoder *encoder =
          backend.createEncoder();

      backend.encodeSetZero(
          *encoder,
          *y);

      backend.submitAndWait(
          *encoder);

      delete encoder;

      float *result =
          y->download();

      checkVector(
          result,
          {
              0.0f,
              0.0f,
              0.0f,
              0.0f,
              0.0f
          },
          "setZero test failed");

      std::cout
          << "setZero: PASS\n";
    }

    // ========================================================
    // 5. MULTIPLE OPERATIONS IN ONE ENCODER
    //
    // Reset x to [1,2,3,4,5].
    //
    // x <- 2x
    // y <- x
    // y <- y - 0.5 x
    //
    // Expected:
    // y = 0.5 x
    //   = [1,2,3,4,5]
    //
    // This is important because it checks that kernels submitted
    // to the same CUDA stream execute in order.
    // ========================================================

    x->updateValues(xValues.data(), xValues.size());

    {
      gpuSolver::BackendEncoder *encoder =
          backend.createEncoder();

      backend.encodeScale(
          *encoder,
          *x,
          2.0f);

      backend.encodeCopy(
          *encoder,
          *x,
          *y);

      backend.encodeAxpy(
          *encoder,
          -0.5f,
          *x,
          *y);

      backend.submitAndWait(
          *encoder);

      delete encoder;

      float *result =
          y->download();

      checkVector(
          result,
          {
              1.0f,
              2.0f,
              3.0f,
              4.0f,
              5.0f
          },
          "multiple-kernel stream test failed");

      std::cout
          << "multiple kernels in one encoder: PASS\n";
    }

    delete x;
    delete y;

    std::cout
        << "\nCUDA vector kernel tests: PASS\n";

    return 0;

  } catch (const std::exception &exception) {

    std::cerr
        << "\nCUDA vector kernel tests: FAIL\n"
        << exception.what()
        << "\n";

    return 1;
  }
}
