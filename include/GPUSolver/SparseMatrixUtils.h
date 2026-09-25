#pragma once 
#include <Eigen/Sparse>


namespace gpuSolver{

// Remaps a sparse matrix's row indices via rowOldToNew and column indices
// via colOldToNew. Pass the same array for both when permuting a square
// matrix symmetrically (e.g. A_k); pass different arrays for one-sided
// remaps (e.g. P_k, R_k, which touch two different levels' permutations).
Eigen::SparseMatrix<float, Eigen::RowMajor> permuteMatrix(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M,
    const int* rowOldToNew,
    const int* colOldToNew);

}
