#include "GPUSolver/HostSparseMatrix.h"
#include <Eigen/Cholesky>
#include <GPUSolver/BlockUtils.h>

namespace gpuSolver {

Permutation
expandBlockPermutation(const gpuSolver::Permutation &nodePermutation,
                       std::size_t numberOfNodes, std::size_t blockSize) {
  const std::size_t numberOfDofs = numberOfNodes * blockSize;

  int *oldToNew = new int[numberOfDofs];
  int *newToOld = new int[numberOfDofs];

  const int *nodeOldToNew = nodePermutation.oldToNew();

  const int *nodeNewToOld = nodePermutation.newToOld();

  for (std::size_t oldNode = 0; oldNode < numberOfNodes; ++oldNode) {

    const int newNode = nodeOldToNew[oldNode];

    for (std::size_t component = 0; component < blockSize; ++component) {

      const std::size_t oldDof = blockSize * oldNode + component;

      const std::size_t newDof =
          blockSize * static_cast<std::size_t>(newNode) + component;

      oldToNew[oldDof] = static_cast<int>(newDof);
    }
  }

  for (std::size_t newNode = 0; newNode < numberOfNodes; ++newNode) {

    const int oldNode = nodeNewToOld[newNode];

    for (std::size_t component = 0; component < blockSize; ++component) {

      const std::size_t newDof = blockSize * newNode + component;

      const std::size_t oldDof =
          blockSize * static_cast<std::size_t>(oldNode) + component;

      newToOld[newDof] = static_cast<int>(oldDof);
    }
  }

  Permutation permutation(numberOfDofs, oldToNew, newToOld);

  delete[] oldToNew;
  delete[] newToOld;

  return permutation;
}

BlockGraph
buildBlockGraph(const Eigen::SparseMatrix<float, Eigen::RowMajor> &matrix,
                std::size_t blockSize) {
  if (matrix.rows() != matrix.cols()) {
    throw std::runtime_error("buildBlockGraph: matrix must be square.");
  }

  if (matrix.rows() % static_cast<int>(blockSize) != 0) {
    throw std::runtime_error("buildBlockGraph: matrix size is not "
                             "divisible by block size.");
  }

  const std::size_t numberOfBlocks =
      static_cast<std::size_t>(matrix.rows()) / blockSize;

  std::vector<std::vector<int>> adjacency(numberOfBlocks);

  for (int outer = 0; outer < matrix.outerSize(); ++outer) {

    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(matrix,
                                                                       outer);
         it; ++it) {

      const std::size_t blockRow =
          static_cast<std::size_t>(it.row()) / blockSize;

      const std::size_t blockCol =
          static_cast<std::size_t>(it.col()) / blockSize;

      if (blockRow == blockCol) {
        continue;
      }

      adjacency[blockRow].push_back(static_cast<int>(blockCol));

      adjacency[blockCol].push_back(static_cast<int>(blockRow));
    }
  }

  BlockGraph graph;

  graph.rowPtr.resize(numberOfBlocks + 1);
  graph.rowPtr[0] = 0;

  for (std::size_t block = 0; block < numberOfBlocks; ++block) {

    std::vector<int> &neighbors = adjacency[block];

    std::sort(neighbors.begin(), neighbors.end());

    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());

    graph.rowPtr[block + 1] =
        graph.rowPtr[block] + static_cast<int>(neighbors.size());
  }

  graph.colPtr.reserve(static_cast<std::size_t>(graph.rowPtr.back()));

  for (const std::vector<int> &neighbors : adjacency) {

    for (int neighbor : neighbors) {
      graph.colPtr.push_back(neighbor);
    }
  }

  return graph;
}

BlockGraph buildBlockGraph(const HostCSRMatrix &matrix, std::size_t blockSize) {
  if (matrix.rows() != matrix.cols()) {
    throw std::runtime_error("buildBlockGraph: matrix must be square.");
  }

  if (matrix.rows() % blockSize != 0) {
    throw std::runtime_error("buildBlockGraph: matrix size is not "
                             "divisible by block size.");
  }

  const std::size_t numberOfBlocks = matrix.rows() / blockSize;

  std::vector<std::vector<int>> adjacency(numberOfBlocks);

  const int *rowPtr = matrix.activeRowPtr();

  const int *colPtr = matrix.activeColPtr();

  // ------------------------------------------------------
  // Traverse the scalar CSR matrix.
  //
  // Every scalar row belongs to one block:
  //
  //   blockRow = row / blockSize
  //
  // Every scalar column belongs to one block:
  //
  //   blockCol = col / blockSize
  //
  // If blockRow != blockCol, the two blocks are coupled.
  // ------------------------------------------------------

  for (std::size_t row = 0; row < matrix.rows(); ++row) {

    const std::size_t blockRow = row / blockSize;

    for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {

      const int col = colPtr[entry];

      const std::size_t blockCol = static_cast<std::size_t>(col) / blockSize;

      // Couplings within the same diagonal block do not
      // create graph edges.
      if (blockRow == blockCol) {
        continue;
      }

      adjacency[blockRow].push_back(static_cast<int>(blockCol));

      // Make the block graph explicitly undirected.
      adjacency[blockCol].push_back(static_cast<int>(blockRow));
    }
  }

  // ------------------------------------------------------
  // Convert adjacency lists to CSR.
  // ------------------------------------------------------

  BlockGraph graph;

  graph.rowPtr.resize(numberOfBlocks + 1);

  graph.rowPtr[0] = 0;

  for (std::size_t block = 0; block < numberOfBlocks; ++block) {

    std::vector<int> &neighbors = adjacency[block];

    // Many scalar entries can belong to the same 3x3
    // block coupling, so duplicates are expected.
    std::sort(neighbors.begin(), neighbors.end());

    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());

    graph.rowPtr[block + 1] =
        graph.rowPtr[block] + static_cast<int>(neighbors.size());
  }

  graph.colPtr.reserve(static_cast<std::size_t>(graph.rowPtr.back()));

  for (const std::vector<int> &neighbors : adjacency) {

    for (int neighbor : neighbors) {
      graph.colPtr.push_back(neighbor);
    }
  }

  return graph;
}

std::vector<float>
extractInverseDiagonalBlocks3x3(const HostCSRMatrix &matrix) {
  const std::size_t n = matrix.rows();

  if (n % 3 != 0) {
    throw std::runtime_error("3x3 block smoother requires "
                             "number of DOFs divisible by 3.");
  }

  const std::size_t numberOfBlocks = n / 3;

  std::vector<float> inverseBlocks(9 * numberOfBlocks);

  const int *rowPtr = matrix.activeRowPtr();

  const int *colPtr = matrix.activeColPtr();

  const float *values = matrix.activeValPtr();

  for (std::size_t block = 0; block < numberOfBlocks; ++block) {

    Eigen::Matrix3f diagonalBlock = Eigen::Matrix3f::Zero();

    for (int localRow = 0; localRow < 3; ++localRow) {

      const int row = static_cast<int>(3 * block) + localRow;

      for (int entry = rowPtr[row]; entry < rowPtr[row + 1]; ++entry) {

        const int col = colPtr[entry];

        const int firstBlockColumn = static_cast<int>(3 * block);

        if (col >= firstBlockColumn && col < firstBlockColumn + 3) {

          const int localCol = col - firstBlockColumn;

          diagonalBlock(localRow, localCol) = values[entry];
        }
      }
    }

    Eigen::LLT<Eigen::Matrix3f> llt(diagonalBlock);

    if (llt.info() != Eigen::Success) {
      throw std::runtime_error("Block diagonal is not SPD.");
    }

    const Eigen::Matrix3f inverse = llt.solve(Eigen::Matrix3f::Identity());

    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {

        inverseBlocks[9 * block + 3 * row + col] = inverse(row, col);
      }
    }
  }

  return inverseBlocks;
}

} // namespace gpuSolver
