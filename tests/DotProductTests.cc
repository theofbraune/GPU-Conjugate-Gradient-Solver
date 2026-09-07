
#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <cstdlib>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>
// #error TESTING_CORRECT_FILE
namespace {

double cpuDot(const std::vector<float> &x, const std::vector<float> &y) {
  double result = 0.0;

  for (std::size_t i = 0; i < x.size(); ++i) {
    result += static_cast<double>(x[i]) * static_cast<double>(y[i]);
  }

  return result;
}

void testDotForSize(std::size_t n) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  std::vector<float> x(n);
  std::vector<float> y(n);

  for (std::size_t i = 0; i < n; ++i) {
    const float index = static_cast<float>(i);

    x[i] = std::sin(0.013f * index) + 0.25f;

    y[i] = std::cos(0.007f * index) - 0.15f;
  }

  const float expected = cpuDot(x, y);

  gpuSolver::DeviceVector deviceX(context, x);

  gpuSolver::DeviceVector deviceY(context, y);

  gpuSolver::DeviceScalar result(context);

  backend.dot(deviceX, deviceY, result);

  const float gpuResult = result.download();
  // const float tolerance = 1e-4 * std::max(1.0, std::abs(expected));
  const float tolerance = 1e-4*(std::max(1.0f, std::abs(expected)));

  EXPECT_NEAR(static_cast<double>(gpuResult), expected, tolerance);

}

} // namespace

TEST(MetalDot, SingleElement) { testDotForSize(1); }

TEST(MetalDot, SmallNonPowerOfTwo) { testDotForSize(17); }

TEST(MetalDot, JustBelowThreadgroupSize) { testDotForSize(255); }

TEST(MetalDot, ExactlyThreadgroupSize) { testDotForSize(256); }

TEST(MetalDot, JustAboveThreadgroupSize) { testDotForSize(257); }

TEST(MetalDot, ThousandElements) { testDotForSize(1000); }

TEST(MetalDot, LargeVector) { testDotForSize(100000); }

TEST(MetalDot, Ones) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  const std::size_t n = 1000;

  std::vector<float> x(n, 1.0f);

  std::vector<float> y(n, 1.0f);

  gpuSolver::DeviceVector deviceX(context, x);

  gpuSolver::DeviceVector deviceY(context, y);

  gpuSolver::DeviceScalar result(context);

  backend.dot(deviceX, deviceY, result);

  EXPECT_FLOAT_EQ(result.download(), static_cast<float>(n));
}

TEST(MetalDot, ZeroResult) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  std::vector<float> x = {1.0f, 1.0f, -1.0f, -1.0f};

  std::vector<float> y = {1.0f, -1.0f, 1.0f, -1.0f};

  gpuSolver::DeviceVector deviceX(context, x);

  gpuSolver::DeviceVector deviceY(context, y);

  gpuSolver::DeviceScalar result(context);

  backend.dot(deviceX, deviceY, result);

  EXPECT_NEAR(result.download(), 0.0f, 1e-6f);
}

TEST(MetalDot, DifferentVectorSizesThrow) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  std::vector<float> x(10, 1.0f);

  std::vector<float> y(11, 1.0f);

  gpuSolver::DeviceVector deviceX(context, x);

  gpuSolver::DeviceVector deviceY(context, y);

  gpuSolver::DeviceScalar result(context);

  EXPECT_THROW(backend.dot(deviceX, deviceY, result), std::runtime_error);
}
