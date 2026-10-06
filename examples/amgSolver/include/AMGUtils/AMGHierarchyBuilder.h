#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <GPUSolver/AMGHierarchy.h>


namespace MGBuilder {

// A multigrid hierarchy: matrices, mass matrices, transfer operators, and
// the per-level near-null space (rigid body modes), finest level first.
// struct AMGHierarchy {
//     std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> A;
//     std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> M;
//     std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> P;
//     std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> R;
//     std::vector<Eigen::MatrixXf> nullspaces;
// };

// Builds the 6 analytical rigid body modes (3 translations + 3 infinitesimal
// rotations) for a set of 3D vertex positions, orthonormalized via QR.
// V_rest: nVerts x 3 rest-pose positions. modes (out): (3*nVerts) x cols.
void buildRigidBodyModes(const Eigen::MatrixXf& V_rest, Eigen::MatrixXf& modes, int& cols);

// Full AMG hierarchy via amgcl's Ruge-Stuben coarsening. Ruge-Stuben does
// not use near-null-space information for coarsening itself; the rigid
// body modes are still tracked per level for consumers that want them.
gpuSolver::AMGHierarchy buildAmgclRugeStubenHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M0,
    const Eigen::MatrixXf& V,
    int maxLevels = 10,
    int minDofs = 1000);

gpuSolver::AMGHierarchy buildAmgclScalarSmoothedAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    int maxLevels = 10,
    int minDofs = 1000);

// Full AMG hierarchy via amgcl's plain (non-smoothed) aggregation,
// driven by the analytical rigid body modes as near-null space.
gpuSolver::AMGHierarchy buildAmgclAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M0,
    const Eigen::MatrixXf& V,
    int maxLevels = 10,
    int minDofs = 1000);

// Full AMG hierarchy via amgcl's smoothed aggregation.
gpuSolver::AMGHierarchy buildAmgclSmoothedAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M0,
    const Eigen::MatrixXf& V,
    int maxLevels = 10,
    int minDofs = 1000);

gpuSolver::AMGHierarchy buildAmgclBlockSmoothedAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    int blockSize,
    int maxLevels = 10,
    int minDofs = 1000);

// Just the chain of restriction operators (finest -> coarsest) for a single
// matrix, via smoothed aggregation, stopping once at most maxVertices rows
// remain.
std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> buildRestrictionMatricesSmoothedAggregation(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    int blockSize = 3,
    float epsStrong = 0.08f,
    int maxVertices = 1000);

// Same, via Ruge-Stuben coarsening.
std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> buildRestrictionMatricesRugeStuben(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    float epsStrong = 0.08f,
    int maxVertices = 1000);

} // namespace MGBuilder



