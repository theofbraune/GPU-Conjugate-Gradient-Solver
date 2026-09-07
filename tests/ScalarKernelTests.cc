#include <GPUSolver/DeviceScalar.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <gtest/gtest.h>

#include <cmath>


TEST(MetalScalar, Set)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarSet(
        3.25f,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        3.25f
    );
}


TEST(MetalScalar, Copy)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar input(
        context,
        4.75f
    );

    gpuSolver::DeviceScalar output(
        context
    );

    backend.scalarCopy(
        input,
        output
    );

    EXPECT_FLOAT_EQ(
        output.download(),
        4.75f
    );
}


TEST(MetalScalar, Divide)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar numerator(
        context,
        7.5f
    );

    gpuSolver::DeviceScalar denominator(
        context,
        2.5f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarDivide(
        numerator,
        denominator,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        3.0f
    );
}


TEST(MetalScalar, DivideNonExact)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar numerator(
        context,
        1.0f
    );

    gpuSolver::DeviceScalar denominator(
        context,
        3.0f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarDivide(
        numerator,
        denominator,
        result
    );

    EXPECT_NEAR(
        result.download(),
        1.0f / 3.0f,
        1e-6f
    );
}


TEST(MetalScalar, Multiply)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar factor1(
        context,
        2.5f
    );

    gpuSolver::DeviceScalar factor2(
        context,
        -4.0f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarMultiply(
        factor1,
        factor2,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        -10.0f
    );
}


TEST(MetalScalar, NegatePositive)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar input(
        context,
        12.5f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarNegate(
        input,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        -12.5f
    );
}


TEST(MetalScalar, NegateNegative)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar input(
        context,
        -3.25f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarNegate(
        input,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        3.25f
    );
}


TEST(MetalScalar, Sqrt)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar input(
        context,
        25.0f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarSqrt(
        input,
        result
    );

    EXPECT_FLOAT_EQ(
        result.download(),
        5.0f
    );
}


TEST(MetalScalar, SqrtNonExact)
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceScalar input(
        context,
        2.0f
    );

    gpuSolver::DeviceScalar result(
        context
    );

    backend.scalarSqrt(
        input,
        result
    );

    EXPECT_NEAR(
        result.download(),
        std::sqrt(2.0f),
        1e-6f
    );
}
