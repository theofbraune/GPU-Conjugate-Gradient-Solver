#include <gtest/gtest.h>

// #include <GPUSolver/GraphColoring.h>
#include <GPUSolver/GraphColoring.h>

#include <Eigen/SparseCore>

#include <cstddef>
#include <vector>

namespace
{

using Matrix =
    Eigen::SparseMatrix<float, Eigen::RowMajor>;


// ------------------------------------------------------------
// Generic validation helper.
//
// Checks:
//   1. every vertex has a valid color
//   2. adjacent vertices have different colors
//   3. colorVertices contains every vertex exactly once
//   4. colorOffsets correctly describe the color groups
// ------------------------------------------------------------

void validateColoring(
    const Matrix& A,
    const gpuSolver::GraphColoring& coloring)
{
    const std::size_t n =
        static_cast<std::size_t>(A.rows());

    const std::vector<int>& colors =
        coloring.colors();

    const std::vector<int>& colorVertices =
        coloring.colorVertices();

    const std::vector<int>& colorOffsets =
        coloring.colorOffsets();

    const std::size_t numberOfColors =
        coloring.numberOfColors();

    // --------------------------------------------------------
    // Correct sizes
    // --------------------------------------------------------

    ASSERT_EQ(colors.size(), n);

    ASSERT_EQ(colorVertices.size(), n);

    ASSERT_EQ(
        colorOffsets.size(),
        numberOfColors + 1
    );

    ASSERT_EQ(colorOffsets.front(), 0);

    ASSERT_EQ(
        colorOffsets.back(),
        static_cast<int>(n)
    );

    // --------------------------------------------------------
    // Every vertex has a valid color.
    // --------------------------------------------------------

    for (std::size_t vertex = 0;
         vertex < n;
         ++vertex)
    {
        EXPECT_GE(
            colors[vertex],
            0
        );

        EXPECT_LT(
            static_cast<std::size_t>(
                colors[vertex]
            ),
            numberOfColors
        );
    }

    // --------------------------------------------------------
    // Graph-coloring condition:
    //
    //     A_ij != 0, i != j
    //          =>
    //     color(i) != color(j)
    // --------------------------------------------------------

    for (Eigen::Index row = 0;
         row < A.outerSize();
         ++row)
    {
        for (Matrix::InnerIterator entry(A, row);
             entry;
             ++entry)
        {
            const Eigen::Index col =
                entry.col();

            if (row == col)
            {
                continue;
            }

            EXPECT_NE(
                colors[static_cast<std::size_t>(row)],
                colors[static_cast<std::size_t>(col)]
            )
                << "Vertices "
                << row
                << " and "
                << col
                << " are adjacent but have the same color.";
        }
    }

    // --------------------------------------------------------
    // colorVertices must contain each vertex exactly once.
    // --------------------------------------------------------

    std::vector<bool> seen(n, false);

    for (std::size_t index = 0;
         index < colorVertices.size();
         ++index)
    {
        const int vertex =
            colorVertices[index];

        ASSERT_GE(vertex, 0);

        ASSERT_LT(
            static_cast<std::size_t>(vertex),
            n
        );

        EXPECT_FALSE(
            seen[static_cast<std::size_t>(vertex)]
        )
            << "Vertex "
            << vertex
            << " occurs more than once in colorVertices.";

        seen[static_cast<std::size_t>(vertex)] =
            true;
    }

    for (std::size_t vertex = 0;
         vertex < n;
         ++vertex)
    {
        EXPECT_TRUE(seen[vertex])
            << "Vertex "
            << vertex
            << " is missing from colorVertices.";
    }

    // --------------------------------------------------------
    // Verify colorOffsets / colorVertices agreement.
    //
    // All vertices between offsets[c] and offsets[c+1]
    // must actually have color c.
    // --------------------------------------------------------

    for (std::size_t color = 0;
         color < numberOfColors;
         ++color)
    {
        const int begin =
            colorOffsets[color];

        const int end =
            colorOffsets[color + 1];

        ASSERT_LE(begin, end);

        for (int index = begin;
             index < end;
             ++index)
        {
            const int vertex =
                colorVertices[
                    static_cast<std::size_t>(index)
                ];

            EXPECT_EQ(
                colors[
                    static_cast<std::size_t>(vertex)
                ],
                static_cast<int>(color)
            );
        }
    }
}


// ------------------------------------------------------------
// Helper to run coloring on an Eigen CSR matrix.
// ------------------------------------------------------------

gpuSolver::GraphColoring computeColoring(
    const Matrix& A)
{
    gpuSolver::GraphColoring coloring;

    coloring.compute(
        static_cast<std::size_t>(A.rows()),
        A.outerIndexPtr(),
        A.innerIndexPtr()
    );

    return coloring;
}

TEST(GraphColoring, PathGraph)
{
    // 0 -- 1 -- 2 -- 3 -- 4 -- 5

    constexpr int n = 6;

    std::vector<Eigen::Triplet<float>> triplets;

    for (int i = 0; i < n; ++i)
    {
        triplets.emplace_back(i, i, 1.0f);

        if (i + 1 < n)
        {
            triplets.emplace_back(i, i + 1, -1.0f);
            triplets.emplace_back(i + 1, i, -1.0f);
        }
    }

    Matrix A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    gpuSolver::GraphColoring coloring =
        computeColoring(A);

    validateColoring(
        A,
        coloring
    );

    EXPECT_EQ(
        coloring.numberOfColors(),
        2
    );
}

TEST(GraphColoring, TriangleRequiresThreeColors)
{
    //       0
    //      / \
    //     1---2

    constexpr int n = 3;

    std::vector<Eigen::Triplet<float>> triplets = {
        {0, 0, 1.0f},
        {1, 1, 1.0f},
        {2, 2, 1.0f},

        {0, 1, -1.0f},
        {1, 0, -1.0f},

        {1, 2, -1.0f},
        {2, 1, -1.0f},

        {2, 0, -1.0f},
        {0, 2, -1.0f}
    };

    Matrix A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    gpuSolver::GraphColoring coloring =
        computeColoring(A);

    validateColoring(
        A,
        coloring
    );

    EXPECT_EQ(
        coloring.numberOfColors(),
        3
    );
}

TEST(GraphColoring, EvenCycleUsesTwoColors)
{
    // 0 -- 1
    // |    |
    // 3 -- 2

    constexpr int n = 4;

    std::vector<Eigen::Triplet<float>> triplets = {
        {0, 0, 1.0f},
        {1, 1, 1.0f},
        {2, 2, 1.0f},
        {3, 3, 1.0f},

        {0, 1, -1.0f},
        {1, 0, -1.0f},

        {1, 2, -1.0f},
        {2, 1, -1.0f},

        {2, 3, -1.0f},
        {3, 2, -1.0f},

        {3, 0, -1.0f},
        {0, 3, -1.0f}
    };

    Matrix A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    gpuSolver::GraphColoring coloring =
        computeColoring(A);

    validateColoring(
        A,
        coloring
    );

    EXPECT_EQ(
        coloring.numberOfColors(),
        2
    );
}

TEST(GraphColoring, CompleteGraph)
{
    constexpr int n = 5;

    std::vector<Eigen::Triplet<float>> triplets;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                triplets.emplace_back(
                    i,
                    j,
                    1.0f
                );
            }
            else
            {
                triplets.emplace_back(
                    i,
                    j,
                    -1.0f
                );
            }
        }
    }

    Matrix A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    gpuSolver::GraphColoring coloring =
        computeColoring(A);

    validateColoring(
        A,
        coloring
    );

    EXPECT_EQ(
        coloring.numberOfColors(),
        5
    );
}

TEST(GraphColoring, Regular2DGrid)
{
    constexpr int nx = 10;
    constexpr int ny = 10;

    constexpr int n =
        nx * ny;

    std::vector<Eigen::Triplet<float>> triplets;

    auto index =
        [](int x, int y)
        {
            return y * nx + x;
        };

    for (int y = 0;
         y < ny;
         ++y)
    {
        for (int x = 0;
             x < nx;
             ++x)
        {
            const int i =
                index(x, y);

            triplets.emplace_back(
                i,
                i,
                4.0f
            );

            if (x > 0)
            {
                triplets.emplace_back(
                    i,
                    index(x - 1, y),
                    -1.0f
                );
            }

            if (x + 1 < nx)
            {
                triplets.emplace_back(
                    i,
                    index(x + 1, y),
                    -1.0f
                );
            }

            if (y > 0)
            {
                triplets.emplace_back(
                    i,
                    index(x, y - 1),
                    -1.0f
                );
            }

            if (y + 1 < ny)
            {
                triplets.emplace_back(
                    i,
                    index(x, y + 1),
                    -1.0f
                );
            }
        }
    }

    Matrix A(n, n);

    A.setFromTriplets(
        triplets.begin(),
        triplets.end()
    );

    A.makeCompressed();

    gpuSolver::GraphColoring coloring =
        computeColoring(A);

    validateColoring(
        A,
        coloring
    );

    // The grid graph is bipartite.
    EXPECT_EQ(
        coloring.numberOfColors(),
        2
    );
}

} // namespace
