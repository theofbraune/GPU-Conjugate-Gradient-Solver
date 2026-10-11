#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <gtest/gtest.h>

#include <cmath>

TEST(MetalScalar, Set) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar *result = backend.createScalar();

  backend.scalarSet(3.25f, *result);

  EXPECT_FLOAT_EQ(result->download(), 3.25f);
  delete result;
}

TEST(MetalScalar, Copy) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar *input = backend.createScalar(4.75f);

  gpuSolver::DeviceScalar *output = backend.createScalar();

  backend.scalarCopy(*input, *output);

  EXPECT_FLOAT_EQ(output->download(), 4.75f);
  delete input;
  delete output;
}

TEST(MetalScalar, Divide) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar *numerator = backend.createScalar(7.5f);

  gpuSolver::DeviceScalar *denominator = backend.createScalar(2.5f);

  gpuSolver::DeviceScalar *result = backend.createScalar();

  backend.scalarDivide(*numerator, *denominator, *result);

  EXPECT_FLOAT_EQ(result->download(), 3.0f);
  delete numerator;
  delete denominator;
  delete result;
}

TEST(MetalScalar, DivideNonExact) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar *numerator = backend.createScalar(1.0f);

  gpuSolver::DeviceScalar *denominator = backend.createScalar(3.0f);

  gpuSolver::DeviceScalar *result = backend.createScalar();

  backend.scalarDivide(*numerator, *denominator, *result);

  EXPECT_NEAR(result->download(), 1.0f / 3.0f, 1e-6f);

  delete numerator;
  delete denominator;
  delete result;
}

TEST(MetalScalar, Multiply) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar* factor1 = backend.createScalar( 2.5f);

  gpuSolver::DeviceScalar* factor2 = backend.createScalar( -4.0f);

  gpuSolver::DeviceScalar* result = backend.createScalar();

  backend.scalarMultiply(*factor1, *factor2, *result);

  EXPECT_FLOAT_EQ(result->download(), -10.0f);

  delete factor1;
  delete factor2;
  delete result;
}

TEST(MetalScalar, NegatePositive) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar* input = backend.createScalar( 12.5f);

  gpuSolver::DeviceScalar* result = backend.createScalar();

  backend.scalarNegate(*input, *result);

  EXPECT_FLOAT_EQ(result->download(), -12.5f);

  delete input;
  delete result;
}

TEST(MetalScalar, NegateNegative) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar* input = backend.createScalar( -3.25f);

  gpuSolver::DeviceScalar* result = backend.createScalar();

  backend.scalarNegate(*input, *result);

  EXPECT_FLOAT_EQ(result->download(), 3.25f);

  delete input;
  delete result;
}

TEST(MetalScalar, Sqrt) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar* input = backend.createScalar( 25.0f);

  gpuSolver::DeviceScalar* result = backend.createScalar();

  backend.scalarSqrt(*input, *result);

  EXPECT_FLOAT_EQ(result->download(), 5.0f);

  delete input;
  delete result;
}

TEST(MetalScalar, SqrtNonExact) {
  gpuSolver::MetalContext context;
  gpuSolver::MetalBackend backend(context);

  gpuSolver::DeviceScalar *input = backend.createScalar(2.0f);

  gpuSolver::DeviceScalar *result = backend.createScalar();

  backend.scalarSqrt(*input, *result);

  EXPECT_NEAR(result->download(), std::sqrt(2.0f), 1e-6f);
  delete input;
  delete result;
}
