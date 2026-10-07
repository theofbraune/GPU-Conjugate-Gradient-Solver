#include <AMGUtils/SmoothedAggregationCoarsener.h>

#include <AMGUtils/TypeConverter.h>
#include <GPUSolver/AMGHierarchy.h>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace gpuSolver {

namespace {

using Backend = amgcl::backend::builtin<float>;

using CrsMatrix = amgcl::backend::crs<float>;

using Coarsening = amgcl::coarsening::smoothed_aggregation<Backend>;

AMGHierarchy::Matrix toEigen(const CrsMatrix &matrix) {
  return AMGUtils::crs_to_eigen_triplets(matrix);
}

std::shared_ptr<CrsMatrix> toBackend(const AMGHierarchy::Matrix &matrix) {
  ptrdiff_t rows;

  std::vector<ptrdiff_t> rowPtr;
  std::vector<ptrdiff_t> colPtr;
  std::vector<float> values;

  AMGUtils::eigen_to_crs(matrix, rows, rowPtr, colPtr, values);

  return std::make_shared<CrsMatrix>(rows, rows, rowPtr, colPtr, values);
}

void setNearNullspace(Coarsening::params &parameters,
                      const Eigen::MatrixXf &modes) {
  const int numberOfRows = static_cast<int>(modes.rows());

  const int numberOfModes = static_cast<int>(modes.cols());

  parameters.nullspace.cols = numberOfModes;

  parameters.nullspace.B.resize(
      static_cast<std::size_t>(numberOfRows * numberOfModes));

  for (int col = 0; col < numberOfModes; ++col) {

    for (int row = 0; row < numberOfRows; ++row) {

      parameters.nullspace
          .B[static_cast<std::size_t>(col * numberOfRows + row)] =
          modes(row, col);
    }
  }
}

} // namespace


AMGCoarsener::Result SmoothedAggregationCoarsener::coarsen(
    const Matrix &matrix, const Eigen::MatrixXf *nearNullspace) const {
  if (matrix.rows() % blockSize_ != 0) {

    throw std::runtime_error("SmoothedAggregationCoarsener: "
                             "matrix size is not divisible by block size.");
  }

  std::shared_ptr<CrsMatrix> backendMatrix = toBackend(matrix);

  Coarsening::params parameters;

  parameters.aggr.block_size = blockSize_;

  parameters.aggr.eps_strong = epsStrong_;

  if (nearNullspace != nullptr) {

    if (nearNullspace->rows() != matrix.rows()) {

      throw std::runtime_error("SmoothedAggregationCoarsener: "
                               "near-nullspace row count mismatch.");
    }

    setNearNullspace(parameters, *nearNullspace);
  }

  Coarsening coarsening(parameters);

  auto [prolongationPtr, restrictionPtr] =
      coarsening.transfer_operators(*backendMatrix);

  std::shared_ptr<CrsMatrix> coarseMatrixPtr = coarsening.coarse_operator(
      *backendMatrix, *prolongationPtr, *restrictionPtr);

  Result result;

  result.prolongation = toEigen(*prolongationPtr);

  result.restriction = toEigen(*restrictionPtr);

  result.coarseMatrix = toEigen(*coarseMatrixPtr);

  return result;
}

} // namespace gpuSolver
