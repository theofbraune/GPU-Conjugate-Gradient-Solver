#pragma once

#include <Eigen/SparseCore>

namespace gpuSolver {
struct AMGHierarchy {
  using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

  std::vector<Matrix> A; // size L+1
  std::vector<Matrix> P; // size L, P[k]: level k+1 -> level k
  std::vector<Matrix> R; // size L, R[k]: level k   -> level k+1
};
} // namespace gpuSolver
