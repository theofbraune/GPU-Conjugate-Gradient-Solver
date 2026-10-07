#include <AMGUtils/AMGHierarchyBuilder.h>

#include <Eigen/QR>

#include <stdexcept>

namespace gpuSolver {

AMGHierarchyBuilder::AMGHierarchyBuilder(
    const AMGCoarsener &coarsener,
    int maxLevels,
    int minDofs)
    : coarsener_(coarsener),
      maxLevels_(maxLevels),
      minDofs_(minDofs)
{
  if (maxLevels_ <= 0) {
    throw std::runtime_error(
        "AMGHierarchyBuilder: "
        "maxLevels must be positive.");
  }

  if (minDofs_ <= 0) {
    throw std::runtime_error(
        "AMGHierarchyBuilder: "
        "minDofs must be positive.");
  }
}


Eigen::MatrixXf
AMGHierarchyBuilder::orthonormalize(
    const Eigen::MatrixXf &modes) const
{
  if (modes.cols() == 0) {
    return modes;
  }

  if (modes.rows()
      < modes.cols()) {

    throw std::runtime_error(
        "AMGHierarchyBuilder: "
        "more nullspace modes than DOFs.");
  }

  Eigen::HouseholderQR<Eigen::MatrixXf>
      qr(modes);

  return
      qr.householderQ()
      * Eigen::MatrixXf::Identity(
            modes.rows(),
            modes.cols());
}


AMGHierarchy
AMGHierarchyBuilder::build(
    const AMGHierarchy::Matrix &matrix,
    const Eigen::MatrixXf *nearNullspace) const
{
  if (matrix.rows() != matrix.cols()) {
    throw std::runtime_error(
        "AMGHierarchyBuilder: "
        "matrix must be square.");
  }

  AMGHierarchy hierarchy;

  hierarchy.A.push_back(
      matrix);

  AMGHierarchy::Matrix currentMatrix =
      matrix;

  Eigen::MatrixXf currentNullspace;

  const bool hasNullspace =
      nearNullspace != nullptr;

  if (hasNullspace) {

    if (nearNullspace->rows()
        != matrix.rows()) {

      throw std::runtime_error(
          "AMGHierarchyBuilder: "
          "near-nullspace row count mismatch.");
    }

    currentNullspace =
        orthonormalize(
            *nearNullspace);

    hierarchy.nullspaces.push_back(
        currentNullspace);
  }

  for (int level = 0;
       level < maxLevels_;
       ++level) {

    if (currentMatrix.rows()
        <= minDofs_) {

      break;
    }

    const Eigen::MatrixXf *nullspace =
        hasNullspace
            ? &currentNullspace
            : nullptr;

    AMGCoarsener::Result result =
        coarsener_.coarsen(
            currentMatrix,
            nullspace);

    hierarchy.P.push_back(
        result.prolongation);

    hierarchy.R.push_back(
        result.restriction);

    hierarchy.A.push_back(
        result.coarseMatrix);

    if (hasNullspace) {

      currentNullspace =
          result.restriction
          * currentNullspace;

      currentNullspace =
          orthonormalize(
              currentNullspace);

      hierarchy.nullspaces.push_back(
          currentNullspace);
    }

    currentMatrix =
        result.coarseMatrix;
  }

  return hierarchy;
}

} // namespace gpuSolver

