#pragma once
#include <Eigen/Sparse>
#include <GPUSolver/HostSparseMatrix.h>
#include <GPUSolver/Permutation.h>
#include <vector>

namespace gpuSolver {

struct BlockGraph {
  std::vector<int> rowPtr;
  std::vector<int> colPtr;
};

Permutation
expandBlockPermutation(const gpuSolver::Permutation &nodePermutation,
                       std::size_t numberOfNodes, std::size_t blockSize);

BlockGraph
buildBlockGraph(const Eigen::SparseMatrix<float, Eigen::RowMajor> &matrix,
                std::size_t blockSize);

BlockGraph buildBlockGraph(const HostCSRMatrix &matrix, std::size_t blockSize);

std::vector<float> extractInverseDiagonalBlocks3x3(const HostCSRMatrix &matrix);

} // namespace gpuSolver
