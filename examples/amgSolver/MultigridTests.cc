#include <gtest/gtest.h>

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/SparseMatrixUtils.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>


Eigen::SparseMatrix<float, Eigen::RowMajor>
buildPoisson2D(std::size_t nx, std::size_t ny)
{
    using Matrix =
        Eigen::SparseMatrix<float, Eigen::RowMajor>;

    using Triplet =
        Eigen::Triplet<float>;

    const std::size_t n = nx * ny;

    std::vector<Triplet> entries;
    entries.reserve(5 * n);

    auto index = [nx](std::size_t x, std::size_t y)
    {
        return static_cast<int>(y * nx + x);
    };

    for (std::size_t y = 0; y < ny; ++y)
    {
        for (std::size_t x = 0; x < nx; ++x)
        {
            const int i = index(x, y);

            entries.emplace_back(i, i, 4.0f);

            if (x > 0)
                entries.emplace_back(i, index(x - 1, y), -1.0f);

            if (x + 1 < nx)
                entries.emplace_back(i, index(x + 1, y), -1.0f);

            if (y > 0)
                entries.emplace_back(i, index(x, y - 1), -1.0f);

            if (y + 1 < ny)
                entries.emplace_back(i, index(x, y + 1), -1.0f);
        }
    }

    Matrix A(
        static_cast<int>(n),
        static_cast<int>(n)
    );

    A.setFromTriplets(
        entries.begin(),
        entries.end()
    );

    A.makeCompressed();

    return A;
}


TEST(MultigridDampedJacobi, VCycleReducesResidual)
{
    
    using namespace gpuSolver;
    // --------------------------------------------------
    // Build test problem
    // --------------------------------------------------

    AMGHierarchy::Matrix A =
        buildPoisson2D(32, 32);

    // --------------------------------------------------
    // AMGCL hierarchy
    // --------------------------------------------------

    AMGHierarchy hierarchy =
        buildAMGCLHierarchy(A);

    ASSERT_GE(
        hierarchy.A.size(),
        2
    );

    ASSERT_EQ(
        hierarchy.P.size() + 1,
        hierarchy.A.size()
    );

    ASSERT_EQ(
        hierarchy.R.size() + 1,
        hierarchy.A.size()
    );

    // --------------------------------------------------
    // Verify Galerkin hierarchy
    // --------------------------------------------------

    for (std::size_t level = 0;
         level + 1 < hierarchy.A.size();
         ++level)
    {
        AMGHierarchy::Matrix projected =
            hierarchy.R[level]
            * hierarchy.A[level]
            * hierarchy.P[level];

        AMGHierarchy::Matrix difference =
            hierarchy.A[level + 1]
            - projected;

        const float relativeError =
            difference.norm()
            / hierarchy.A[level + 1].norm();

        EXPECT_LT(
            relativeError,
            1e-5f
        );
    }

    // --------------------------------------------------
    // Hierarchical permutations
    // --------------------------------------------------

    std::vector<Permutation> permutations =
        buildHierarchicalPermutations(
            hierarchy
        );

    ASSERT_EQ(
        permutations.size(),
        hierarchy.A.size()
    );

    // --------------------------------------------------
    // Permute finest matrix
    // --------------------------------------------------

    AMGHierarchy::Matrix APermuted =
        permuteMatrix(
            hierarchy.A[0],
            permutations[0].oldToNew(),
            permutations[0].oldToNew()
        );

    // --------------------------------------------------
    // Deterministic RHS
    // --------------------------------------------------

    Eigen::VectorXf rhs(
        A.rows()
    );

    for (int i = 0;
         i < rhs.size();
         ++i)
    {
        rhs[i] =
            std::sin(
                0.017f * static_cast<float>(i)
            )
            +
            0.3f * std::cos(
                0.043f * static_cast<float>(i)
            );
    }

    Eigen::VectorXf rhsPermuted(
        rhs.size()
    );

    const int* oldToNew =
        permutations[0].oldToNew();

    for (int oldIndex = 0;
         oldIndex < rhs.size();
         ++oldIndex)
    {
        rhsPermuted[
            oldToNew[oldIndex]
        ] = rhs[oldIndex];
    }

    // --------------------------------------------------
    // Backend
    // --------------------------------------------------

    // Replace with however your tests currently create
    // MetalContext + MetalBackend.
    MetalContext context;
    MetalBackend backend(context);

    // --------------------------------------------------
    // MG preconditioner
    // --------------------------------------------------

    MultigridDampedJacobiPreconditioner mg(
        backend,
        hierarchy,
        permutations,
        2,      // pre smoothing
        2,      // post smoothing
        1,      // V-cycle repetitions
        0.67f   // Jacobi damping
    );

    DeviceVector* rhsDevice =
        backend.createVector(
            rhsPermuted.size(),
            rhsPermuted.data()
        );

    DeviceVector* zDevice =
        backend.createVector(
            rhsPermuted.size()
        );

    BackendEncoder* encoder =
        backend.createEncoder();

    mg.apply(
        backend,
        *encoder,
        *rhsDevice,
        *zDevice
    );

    backend.submitAndWait(
        *encoder
    );

    // --------------------------------------------------
    // Download result
    // --------------------------------------------------

    const float* zData =
        zDevice->download();

    Eigen::VectorXf z(
        rhsPermuted.size()
    );

    for (int i = 0;
         i < z.size();
         ++i)
    {
        z[i] = zData[i];
    }

    // --------------------------------------------------
    // Check true residual
    // --------------------------------------------------

    const float residualBefore =
        rhsPermuted.norm();

    const Eigen::VectorXf finalResidual =
        rhsPermuted
        - APermuted * z;

    const float residualAfter =
        finalResidual.norm();

    const float reduction =
        residualAfter / residualBefore;

    std::cout
        << "\nMultigrid V-cycle test\n"
        << "----------------------\n"
        << "Levels:          "
        << hierarchy.A.size() << "\n"
        << "Initial residual: "
        << residualBefore << "\n"
        << "Final residual:   "
        << residualAfter << "\n"
        << "Reduction factor: "
        << reduction << "\n";

    EXPECT_TRUE(
        std::isfinite(residualAfter)
    );

    EXPECT_LT(
        residualAfter,
        residualBefore
    );

    delete encoder;
    delete rhsDevice;
    delete zDevice;
}
