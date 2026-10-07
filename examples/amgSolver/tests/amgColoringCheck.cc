#include <gtest/gtest.h>

#include <AMGUtils/AMGHierarchyBuilder.h>
#include <AMGUtils/SmoothedAggregationCoarsener.h>

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/GraphColoring.h>

#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

// ============================================================
// Build pinned cotangent Laplacian
// ============================================================

Matrix buildReducedCotangentLaplacian(const Eigen::MatrixXd &V,
                                      const Eigen::MatrixXi &F) {
  Eigen::SparseMatrix<double> LDouble;

  igl::cotmatrix(V, F, LDouble);

  Matrix K = (-LDouble).cast<float>();

  K.makeCompressed();

  const Eigen::Index n = K.rows();

  const Eigen::Index pinnedVertex = 0;

  std::vector<Eigen::Index> oldToReduced(static_cast<std::size_t>(n), -1);

  Eigen::Index reducedIndex = 0;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    oldToReduced[static_cast<std::size_t>(i)] = reducedIndex;

    ++reducedIndex;
  }

  std::vector<Eigen::Triplet<float>> triplets;

  triplets.reserve(static_cast<std::size_t>(K.nonZeros()));

  for (Eigen::Index row = 0; row < K.outerSize(); ++row) {
    for (Matrix::InnerIterator entry(K, row); entry; ++entry) {
      const Eigen::Index i = entry.row();

      const Eigen::Index j = entry.col();

      if (i == pinnedVertex || j == pinnedVertex) {
        continue;
      }

      triplets.emplace_back(oldToReduced[static_cast<std::size_t>(i)],
                            oldToReduced[static_cast<std::size_t>(j)],
                            entry.value());
    }
  }

  Matrix A(n - 1, n - 1);

  A.setFromTriplets(triplets.begin(), triplets.end());

  A.makeCompressed();

  return A;
}

// ============================================================
// Verify coloring
// ============================================================

void validateColoring(const Matrix &A,
                      const gpuSolver::GraphColoring &coloring) {
  const std::size_t n = static_cast<std::size_t>(A.rows());

  const std::vector<int> &colors = coloring.colors();

  const std::vector<int> &colorVertices = coloring.colorVertices();

  const std::vector<int> &colorOffsets = coloring.colorOffsets();

  const std::size_t numberOfColors = coloring.numberOfColors();

  ASSERT_EQ(colors.size(), n);

  ASSERT_EQ(colorVertices.size(), n);

  ASSERT_EQ(colorOffsets.size(), numberOfColors + 1);

  ASSERT_EQ(colorOffsets.front(), 0);

  ASSERT_EQ(colorOffsets.back(), static_cast<int>(n));

  // --------------------------------------------------------
  // Every vertex must have a valid color.
  // --------------------------------------------------------

  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    ASSERT_GE(colors[vertex], 0);

    ASSERT_LT(static_cast<std::size_t>(colors[vertex]), numberOfColors);
  }

  // --------------------------------------------------------
  // Core graph-coloring condition:
  //
  // A_ij != 0 and i != j
  //     =>
  // color(i) != color(j)
  // --------------------------------------------------------

  for (Eigen::Index row = 0; row < A.outerSize(); ++row) {
    for (Matrix::InnerIterator entry(A, row); entry; ++entry) {
      const Eigen::Index col = entry.col();

      if (row == col) {
        continue;
      }

      ASSERT_NE(colors[static_cast<std::size_t>(row)],
                colors[static_cast<std::size_t>(col)])
          << "Invalid coloring at edge (" << row << ", " << col << ")";
    }
  }

  // --------------------------------------------------------
  // Every vertex must appear exactly once in colorVertices.
  // --------------------------------------------------------

  std::vector<bool> seen(n, false);

  for (std::size_t i = 0; i < colorVertices.size(); ++i) {
    const int vertex = colorVertices[i];

    ASSERT_GE(vertex, 0);

    ASSERT_LT(static_cast<std::size_t>(vertex), n);

    ASSERT_FALSE(seen[static_cast<std::size_t>(vertex)]);

    seen[static_cast<std::size_t>(vertex)] = true;
  }

  // --------------------------------------------------------
  // colorOffsets must describe correct contiguous groups.
  // --------------------------------------------------------

  for (std::size_t color = 0; color < numberOfColors; ++color) {
    const int begin = colorOffsets[color];

    const int end = colorOffsets[color + 1];

    for (int i = begin; i < end; ++i) {
      const int vertex = colorVertices[static_cast<std::size_t>(i)];

      ASSERT_EQ(colors[static_cast<std::size_t>(vertex)],
                static_cast<int>(color));
    }
  }
}

// ============================================================
// Print diagnostics for one coloring
// ============================================================

void printColoringStatistics(std::size_t level, const Matrix &A,
                             const gpuSolver::GraphColoring &coloring) {
  const std::vector<int> &offsets = coloring.colorOffsets();

  const std::size_t numberOfColors = coloring.numberOfColors();

  std::vector<std::size_t> colorSizes(numberOfColors);

  for (std::size_t color = 0; color < numberOfColors; ++color) {
    colorSizes[color] =
        static_cast<std::size_t>(offsets[color + 1] - offsets[color]);
  }

  const std::size_t smallestColor =
      *std::min_element(colorSizes.begin(), colorSizes.end());

  const std::size_t largestColor =
      *std::max_element(colorSizes.begin(), colorSizes.end());

  const double averageColorSize =
      static_cast<double>(A.rows()) / static_cast<double>(numberOfColors);

  const double averageNnzPerRow =
      static_cast<double>(A.nonZeros()) / static_cast<double>(A.rows());

  std::cout << "Level " << level << ":\n"
            << "  DOFs               = " << A.rows() << "\n"
            << "  nnz                 = " << A.nonZeros() << "\n"
            << "  nnz / row           = " << averageNnzPerRow << "\n"
            << "  number of colors    = " << numberOfColors << "\n"
            << "  smallest color      = " << smallestColor << "\n"
            << "  largest color       = " << largestColor << "\n"
            << "  average color size  = " << averageColorSize << "\n";
}

} // namespace

// ============================================================
// AMG integration test
// ============================================================

TEST(AMGGraphColoring, ColoringIsValidOnAllHierarchyLevels) {
  // --------------------------------------------------------
  // Hard-coded mesh path for now.
  // --------------------------------------------------------

  const std::string meshPath = "../data/surfaceMeshes/david140k.obj";

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  ASSERT_TRUE(igl::read_triangle_mesh(meshPath, V, F));

  std::cout << "\nLoaded mesh with " << V.rows() << " vertices and " << F.rows()
            << " triangles.\n";

  // --------------------------------------------------------
  // Fine SPD system.
  // --------------------------------------------------------

  Matrix A = buildReducedCotangentLaplacian(V, F);

  std::cout << "\nFine matrix:\n"
            << "  DOFs = " << A.rows() << "\n"
            << "  nnz  = " << A.nonZeros() << "\n";

  // --------------------------------------------------------
  // Build scalar smoothed-aggregation AMG hierarchy.
  // --------------------------------------------------------
  gpuSolver::SmoothedAggregationCoarsener coarsener(1, 0.1f);

  gpuSolver::AMGHierarchyBuilder builder(coarsener, 10, 1000);

  gpuSolver::AMGHierarchy hierarchy = builder.build(A);
  // gpuSolver::AMGHierarchy hierarchy =
  //     MGBuilder::buildAmgclScalarSmoothedAggregationHierarchy(A, 10, 1000);

  ASSERT_FALSE(hierarchy.A.empty());

  ASSERT_EQ(hierarchy.P.size() + 1, hierarchy.A.size());

  ASSERT_EQ(hierarchy.R.size() + 1, hierarchy.A.size());

  std::cout << "\nHierarchy contains " << hierarchy.A.size() << " levels.\n\n";

  // --------------------------------------------------------
  // Color every Galerkin matrix.
  // --------------------------------------------------------

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    const Matrix &levelMatrix = hierarchy.A[level];

    gpuSolver::GraphColoring coloring;

    coloring.compute(static_cast<std::size_t>(levelMatrix.rows()),
                     levelMatrix.outerIndexPtr(), levelMatrix.innerIndexPtr());

    // Actual correctness test.
    validateColoring(levelMatrix, coloring);

    // Performance-relevant diagnostics.
    printColoringStatistics(level, levelMatrix, coloring);

    std::cout << "\n";
  }
}
