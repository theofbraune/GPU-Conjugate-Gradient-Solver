#include <gtest/gtest.h>

#include <AMGUtils/AMGHierarchyBuilder.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Permutation.h>
#include <GPUSolver/Preconditioners/MultigridDampedJacobiPreconditioner.h>
#include <GPUSolver/SparseMatrixUtils.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace gpuSolver {

std::vector<Permutation>
buildIdentityPermutations(const AMGHierarchy &hierarchy) {
  std::vector<Permutation> permutations;

  permutations.reserve(hierarchy.A.size());

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    const std::size_t n = static_cast<std::size_t>(hierarchy.A[level].rows());

    std::vector<int> oldToNew(n);
    std::vector<int> newToOld(n);

    for (std::size_t i = 0; i < n; ++i) {
      oldToNew[i] = static_cast<int>(i);

      newToOld[i] = static_cast<int>(i);
    }

    permutations.emplace_back(n, oldToNew.data(), newToOld.data());
  }

  return permutations;
}

namespace {

using Matrix = AMGHierarchy::Matrix;

// ============================================================
// Small deterministic 3D test geometry
// ============================================================

Eigen::MatrixXf buildGridPositions(std::size_t nx, std::size_t ny,
                                   std::size_t nz) {
  const std::size_t nVertices = nx * ny * nz;

  Eigen::MatrixXf V(static_cast<Eigen::Index>(nVertices), 3);

  std::size_t index = 0;

  for (std::size_t z = 0; z < nz; ++z) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nx; ++x) {
        V(static_cast<Eigen::Index>(index), 0) = static_cast<float>(x);

        V(static_cast<Eigen::Index>(index), 1) = static_cast<float>(y);

        V(static_cast<Eigen::Index>(index), 2) = static_cast<float>(z);

        ++index;
      }
    }
  }

  return V;
}

// ============================================================
// Build simple 3-DoF SPD graph operator
//
// This is not a full FEM elasticity stiffness matrix.
// It is a deterministic SPD block graph Laplacian with
// three independent components per vertex.
//
// That is sufficient for exercising the AMG hierarchy,
// permutations and V-cycle.
// ============================================================

Matrix buildVectorLaplacian(std::size_t nx, std::size_t ny, std::size_t nz) {
  using Triplet = Eigen::Triplet<float>;

  const std::size_t nVertices = nx * ny * nz;

  const std::size_t nDofs = 3 * nVertices;

  std::vector<Triplet> entries;

  auto vertexIndex = [nx, ny](std::size_t x, std::size_t y, std::size_t z) {
    return z * nx * ny + y * nx + x;
  };

  auto addEdge = [&entries](std::size_t vertexA, std::size_t vertexB) {
    for (std::size_t component = 0; component < 3; ++component) {
      const int i = static_cast<int>(3 * vertexA + component);

      const int j = static_cast<int>(3 * vertexB + component);

      entries.emplace_back(i, i, 1.0f);

      entries.emplace_back(j, j, 1.0f);

      entries.emplace_back(i, j, -1.0f);

      entries.emplace_back(j, i, -1.0f);
    }
  };

  for (std::size_t z = 0; z < nz; ++z) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nx; ++x) {
        const std::size_t v = vertexIndex(x, y, z);

        if (x + 1 < nx) {
          addEdge(v, vertexIndex(x + 1, y, z));
        }

        if (y + 1 < ny) {
          addEdge(v, vertexIndex(x, y + 1, z));
        }

        if (z + 1 < nz) {
          addEdge(v, vertexIndex(x, y, z + 1));
        }
      }
    }
  }

  // Make the matrix strictly SPD.
  //
  // The pure graph Laplacian has translational null modes.
  // A small diagonal shift avoids singularity in this unit test.
  for (std::size_t dof = 0; dof < nDofs; ++dof) {
    entries.emplace_back(static_cast<int>(dof), static_cast<int>(dof), 0.1f);
  }

  Matrix A(static_cast<Eigen::Index>(nDofs), static_cast<Eigen::Index>(nDofs));

  A.setFromTriplets(entries.begin(), entries.end());

  A.makeCompressed();

  return A;
}

// ============================================================
// Simple diagonal mass matrix
// ============================================================

Matrix buildMassMatrix(std::size_t numberOfDofs) {
  Matrix M(static_cast<Eigen::Index>(numberOfDofs),
           static_cast<Eigen::Index>(numberOfDofs));

  M.reserve(static_cast<Eigen::Index>(numberOfDofs));

  for (std::size_t i = 0; i < numberOfDofs; ++i) {
    M.insert(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 1.0f;
  }

  M.makeCompressed();

  return M;
}

// ============================================================
// Deterministic RHS
// ============================================================

Eigen::VectorXf buildDeterministicRhs(std::size_t size) {
  Eigen::VectorXf rhs(static_cast<Eigen::Index>(size));

  for (Eigen::Index i = 0; i < rhs.size(); ++i) {
    const float index = static_cast<float>(i);

    rhs[i] = std::sin(0.017f * index) + 0.3f * std::cos(0.043f * index);
  }

  return rhs;
}

// ============================================================
// Permute vector
// ============================================================

Eigen::VectorXf permuteVector(const Eigen::VectorXf &vector,
                              const Permutation &permutation) {
  Eigen::VectorXf permuted(vector.size());

  const int *oldToNew = permutation.oldToNew();

  for (Eigen::Index oldIndex = 0; oldIndex < vector.size(); ++oldIndex) {
    const int newIndex = oldToNew[oldIndex];

    permuted[newIndex] = vector[oldIndex];
  }

  return permuted;
}

// ============================================================
// Build actual hierarchy using your AMGCL builder
// ============================================================

AMGHierarchy buildHierarchy() {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 8;
  constexpr std::size_t nz = 8;

  const Eigen::MatrixXf V = buildGridPositions(nx, ny, nz);

  const Matrix A = buildVectorLaplacian(nx, ny, nz);

  const Matrix M = buildMassMatrix(static_cast<std::size_t>(A.rows()));

  return MGBuilder::buildAmgclSmoothedAggregationHierarchy(A, M, V, 10, 50);
}

// ============================================================
// Test 1: hierarchy is Galerkin-consistent
// ============================================================

TEST(MultigridHierarchy, GalerkinOperatorsAreConsistent) {
  const AMGHierarchy hierarchy = buildHierarchy();

  ASSERT_GE(hierarchy.A.size(), 2);

  ASSERT_EQ(hierarchy.P.size() + 1, hierarchy.A.size());

  ASSERT_EQ(hierarchy.R.size() + 1, hierarchy.A.size());

  for (std::size_t level = 0; level + 1 < hierarchy.A.size(); ++level) {
    const Matrix projected =
        hierarchy.R[level] * hierarchy.A[level] * hierarchy.P[level];

    const Matrix difference = hierarchy.A[level + 1] - projected;

    const float denominator = hierarchy.A[level + 1].norm();

    ASSERT_GT(denominator, 0.0f);

    const float relativeError = difference.norm() / denominator;

    EXPECT_LT(relativeError, 1e-4f) << "Galerkin mismatch at level " << level;
  }
}

// ============================================================
// Test 2: every hierarchy permutation is bijective
// ============================================================

TEST(MultigridHierarchy, HierarchicalPermutationsAreValid) {
  const AMGHierarchy hierarchy = buildHierarchy();

  const std::vector<Permutation> permutations =
      buildIdentityPermutations(hierarchy);

  ASSERT_EQ(permutations.size(), hierarchy.A.size());

  for (std::size_t level = 0; level < hierarchy.A.size(); ++level) {
    const std::size_t n = static_cast<std::size_t>(hierarchy.A[level].rows());

    const int *oldToNew = permutations[level].oldToNew();

    std::vector<bool> seen(n, false);

    for (std::size_t oldIndex = 0; oldIndex < n; ++oldIndex) {
      const int newIndex = oldToNew[oldIndex];

      ASSERT_GE(newIndex, 0);

      ASSERT_LT(static_cast<std::size_t>(newIndex), n);

      ASSERT_FALSE(seen[newIndex]);

      seen[newIndex] = true;
    }
  }
}

// ============================================================
// Test 3: GPU V-cycle reduces true residual
// ============================================================

TEST(MultigridDampedJacobi, VCycleReducesResidual) {
  const AMGHierarchy hierarchy = buildHierarchy();

  const std::vector<Permutation> permutations =
      buildIdentityPermutations(hierarchy);

  ASSERT_EQ(permutations.size(), hierarchy.A.size());

  const Matrix APermuted = permuteMatrix(
      hierarchy.A[0], permutations[0].oldToNew(), permutations[0].oldToNew());

  const Eigen::VectorXf rhs =
      buildDeterministicRhs(static_cast<std::size_t>(hierarchy.A[0].rows()));

  const Eigen::VectorXf rhsPermuted = permuteVector(rhs, permutations[0]);

  // --------------------------------------------------------
  // Backend
  // --------------------------------------------------------

  MetalContext context;
  MetalBackend backend(context);

  // --------------------------------------------------------
  // Multigrid preconditioner
  // --------------------------------------------------------

  constexpr std::size_t preSmoothingSteps = 2;
  constexpr std::size_t postSmoothingSteps = 2;
  constexpr std::size_t numberOfVCycles = 1;

  constexpr float omega = 0.67f;

  MultigridDampedJacobiPreconditioner preconditioner(
      backend, hierarchy, permutations, preSmoothingSteps, postSmoothingSteps,
      numberOfVCycles, omega);

  DeviceVector *rhsDevice = backend.createVector(
      static_cast<std::size_t>(rhsPermuted.size()), rhsPermuted.data());

  DeviceVector *zDevice =
      backend.createVector(static_cast<std::size_t>(rhsPermuted.size()));

  BackendEncoder *encoder = backend.createEncoder();

  // --------------------------------------------------------
  // Apply one V-cycle
  // --------------------------------------------------------

  preconditioner.apply(backend, *encoder, *rhsDevice, *zDevice);

  backend.submitAndWait(*encoder);

  // --------------------------------------------------------
  // Download correction
  // --------------------------------------------------------

  const float *zData = zDevice->download();

  Eigen::VectorXf z(rhsPermuted.size());

  for (Eigen::Index i = 0; i < z.size(); ++i) {
    z[i] = zData[i];
  }

  // --------------------------------------------------------
  // True residual
  // --------------------------------------------------------

  const float residualBefore = rhsPermuted.norm();

  const Eigen::VectorXf finalResidual = rhsPermuted - APermuted * z;

  const float residualAfter = finalResidual.norm();

  const float reductionFactor = residualAfter / residualBefore;

  std::cout << "\nMultigrid V-cycle test\n"
            << "----------------------\n"
            << "Levels:            " << hierarchy.A.size() << "\n"
            << "Initial residual:  " << residualBefore << "\n"
            << "Final residual:    " << residualAfter << "\n"
            << "Reduction factor:  " << reductionFactor << "\n";

  EXPECT_TRUE(std::isfinite(residualAfter));

  EXPECT_LT(residualAfter, residualBefore);

  delete encoder;
  delete rhsDevice;
  delete zDevice;
}

} // namespace
} // namespace gpuSolver
