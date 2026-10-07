#pragma once

#include <Eigen/SparseCore>

namespace gpuSolver {
// struct AMGHierarchy {
//
//   std::vector<Matrix> A; // size L+1
//   std::vector<Matrix> P; // size L, P[k]: level k+1 -> level k
//   std::vector<Matrix> R; // size L, R[k]: level k   -> level k+1
// };

struct AMGHierarchy {
  using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;
  std::vector<Matrix> A;
  std::vector<Matrix> M;
  std::vector<Matrix> P;
  std::vector<Matrix> R;
  std::vector<Eigen::MatrixXf> nullspaces;
  std::size_t levels() const { return A.size(); }
};

} // namespace gpuSolver
