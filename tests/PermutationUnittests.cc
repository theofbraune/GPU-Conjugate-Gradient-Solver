#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>


namespace
{

gpuSolver::Permutation* makeRandomPermutation(
    std::size_t size,
    unsigned int seed)
{
    int* newToOld = new int[size];
    int* oldToNew = new int[size];

    for (std::size_t i = 0; i < size; ++i)
    {
        newToOld[i] =
            static_cast<int>(i);
    }

    std::mt19937 generator(seed);

    std::shuffle(
        newToOld,
        newToOld + size,
        generator
    );

    // newToOld[newIndex] = oldIndex
    //
    // Therefore:
    //
    // oldToNew[oldIndex] = newIndex
    //
    for (std::size_t newIndex = 0;
         newIndex < size;
         ++newIndex)
    {
        const int oldIndex =
            newToOld[newIndex];

        oldToNew[oldIndex] =
            static_cast<int>(newIndex);
    }

    gpuSolver::Permutation* permutation =
        new gpuSolver::Permutation(
            size,
            oldToNew,
            newToOld
        );

    delete[] oldToNew;
    delete[] newToOld;

    return permutation;
}


void permuteVector(
    const float* original,
    float* permuted,
    std::size_t size,
    const gpuSolver::Permutation& permutation)
{
    const int* oldToNew =
        permutation.oldToNew();

    for (std::size_t oldIndex = 0;
         oldIndex < size;
         ++oldIndex)
    {
        const int newIndex =
            oldToNew[oldIndex];

        permuted[newIndex] =
            original[oldIndex];
    }
}


void inversePermuteVector(
    const float* permuted,
    float* original,
    std::size_t size,
    const gpuSolver::Permutation& permutation)
{
    const int* newToOld =
        permutation.newToOld();

    for (std::size_t newIndex = 0;
         newIndex < size;
         ++newIndex)
    {
        const int oldIndex =
            newToOld[newIndex];

        original[oldIndex] =
            permuted[newIndex];
    }
}

}


// ============================================================
// First test only the permutation itself.
// ============================================================

TEST(PermutationTest, RandomPermutationIsInvertible)
{
    constexpr std::size_t size = 100;

    gpuSolver::Permutation* permutation =
        makeRandomPermutation(
            size,
            12345
        );

    const int* oldToNew =
        permutation->oldToNew();

    const int* newToOld =
        permutation->newToOld();

    for (std::size_t oldIndex = 0;
         oldIndex < size;
         ++oldIndex)
    {
        const int newIndex =
            oldToNew[oldIndex];

        EXPECT_GE(newIndex, 0);
        EXPECT_LT(
            newIndex,
            static_cast<int>(size)
        );

        EXPECT_EQ(
            newToOld[newIndex],
            static_cast<int>(oldIndex)
        );
    }

    delete permutation;
}


// ============================================================
// Test the complete:
//
// A
//   -> random permutation
//   -> HostCSRMatrix
//   -> DeviceCSRMatrix
//   -> Metal SpMV
//   -> inverse permutation
//
// pipeline.
// ============================================================

TEST(PermutationTest, MetalSpMVWithRandomPermutation)
{
    constexpr int n = 8;

    // --------------------------------------------------------
    // Construct a small sparse matrix.
    //
    // It does not need to be a Laplacian here. We mainly want
    // a nontrivial sparsity pattern and nontrivial values.
    // --------------------------------------------------------

    std::vector<Eigen::Triplet<float>> triplets;

    triplets.emplace_back(0, 0, 4.0f);
    triplets.emplace_back(0, 1, -1.0f);

    triplets.emplace_back(1, 0, -1.0f);
    triplets.emplace_back(1, 1, 5.0f);
    triplets.emplace_back(1, 3, -2.0f);

    triplets.emplace_back(2, 2, 3.0f);
    triplets.emplace_back(2, 4, -1.0f);

    triplets.emplace_back(3, 1, -2.0f);
    triplets.emplace_back(3, 3, 6.0f);
    triplets.emplace_back(3, 5, -1.0f);

    triplets.emplace_back(4, 2, -1.0f);
    triplets.emplace_back(4, 4, 4.0f);
    triplets.emplace_back(4, 6, -1.0f);

    triplets.emplace_back(5, 3, -1.0f);
    triplets.emplace_back(5, 5, 5.0f);
    triplets.emplace_back(5, 7, -2.0f);

    triplets.emplace_back(6, 4, -1.0f);
    triplets.emplace_back(6, 6, 3.0f);

    triplets.emplace_back(7, 5, -2.0f);
    triplets.emplace_back(7, 7, 4.0f);

    Eigen::SparseMatrix<
        float,
        Eigen::RowMajor
    > A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    // --------------------------------------------------------
    // Original vector and Eigen reference.
    // --------------------------------------------------------

    Eigen::VectorXf x(n);

    x <<
        1.0f,
        2.0f,
        -1.0f,
        4.0f,
        0.5f,
        -2.0f,
        3.0f,
        1.5f;

    Eigen::VectorXf expectedY =
        A * x;

    // --------------------------------------------------------
    // Construct a deterministic random permutation.
    // --------------------------------------------------------

    gpuSolver::Permutation* permutation =
        makeRandomPermutation(
            static_cast<std::size_t>(n),
            12345
        );

    // --------------------------------------------------------
    // HostCSRMatrix builds and activates P A P^T.
    // --------------------------------------------------------

    gpuSolver::HostCSRMatrix hostA(
        static_cast<std::size_t>(A.rows()),
        static_cast<std::size_t>(A.cols()),
        static_cast<std::size_t>(A.nonZeros()),
        A.outerIndexPtr(),
        A.innerIndexPtr(),
        A.valuePtr(),
        *permutation
    );

    EXPECT_TRUE(
        hostA.isPermuted()
    );

    // --------------------------------------------------------
    // Build x' = P x.
    // --------------------------------------------------------

    float* xPermuted =
        new float[n];

    permuteVector(
        x.data(),
        xPermuted,
        static_cast<std::size_t>(n),
        *permutation
    );

    // --------------------------------------------------------
    // GPU.
    // --------------------------------------------------------

    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    gpuSolver::DeviceCSRMatrix deviceA(
        context,
        hostA
    );

    std::vector<float> xPermutedVector(
        xPermuted,
        xPermuted + n
    );

    gpuSolver::DeviceVector deviceX(
        context,
        xPermutedVector
    );

    gpuSolver::DeviceVector deviceY(
        context,
        static_cast<std::size_t>(n)
    );

    backend.spmv(
        deviceA,
        deviceX,
        deviceY
    );

    // --------------------------------------------------------
    // GPU gives y' = P y.
    //
    // Convert it back to the original ordering.
    // --------------------------------------------------------

    float* yPermuted =
        deviceY.download();

    float* yOriginal =
        new float[n];

    inversePermuteVector(
        yPermuted,
        yOriginal,
        static_cast<std::size_t>(n),
        *permutation
    );

    // --------------------------------------------------------
    // Compare against ordinary Eigen A*x.
    // --------------------------------------------------------

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(
            yOriginal[i],
            expectedY[i],
            1e-5f
        );
    }

    delete[] xPermuted;
    delete[] yOriginal;
    delete permutation;
}
