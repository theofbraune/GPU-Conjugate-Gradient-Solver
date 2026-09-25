#include <GPUSolver/SparseMatrixUtils.h>
#include <vector>
#include <cstdlib>


namespace gpuSolver {

Eigen::SparseMatrix<float, Eigen::RowMajor> permuteMatrix(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M,
    const int* rowOldToNew,
    const int* colOldToNew)
{
    using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(static_cast<std::size_t>(M.nonZeros()));

    for (Eigen::Index row = 0; row < M.outerSize(); ++row) {
        for (Matrix::InnerIterator it(M, row); it; ++it) {
            triplets.emplace_back(
                rowOldToNew[it.row()],
                colOldToNew[it.col()],
                it.value());
        }
    }

    Matrix result(M.rows(), M.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}
}
