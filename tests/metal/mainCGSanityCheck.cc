#include <GPUSolver/CGSolver.h>
#include <GPUSolver/Preconditioners/IdentityPreconditioner.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(CGSolverTest, SolvesSmallSPDSystem)
{
    using Matrix =
        Eigen::SparseMatrix<float, Eigen::RowMajor>;

    Matrix A(5, 5);

    A.insert(0, 0) = 4.0f;
    A.insert(0, 1) = -1.0f;

    A.insert(1, 0) = -1.0f;
    A.insert(1, 1) = 4.0f;
    A.insert(1, 2) = -1.0f;

    A.insert(2, 1) = -1.0f;
    A.insert(2, 2) = 4.0f;
    A.insert(2, 3) = -1.0f;

    A.insert(3, 2) = -1.0f;
    A.insert(3, 3) = 4.0f;
    A.insert(3, 4) = -1.0f;

    A.insert(4, 3) = -1.0f;
    A.insert(4, 4) = 4.0f;

    A.makeCompressed();

    Eigen::VectorXf xTrue(5);

    xTrue <<
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f;

    Eigen::VectorXf b =
        A * xTrue;

    Eigen::VectorXf x =
        Eigen::VectorXf::Zero(5);

    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);
    gpuSolver::IdentityPreconditioner preconditioner;

    gpuSolver::CGSolver solver(
        backend,
        preconditioner,
        A
    );

    solver.setMaxIterations(100);
    solver.setTolerance(1e-6f);

    solver.solve(
        b,
        x
    );

    EXPECT_LT(
        (x - xTrue).norm(),
        1e-4f
    );

    EXPECT_LT(
        (A * x - b).norm(),
        1e-4f
    );
}
