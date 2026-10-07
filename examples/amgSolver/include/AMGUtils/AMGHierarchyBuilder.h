#pragma once

#include <AMGUtils/AMGCoarsener.h>
#include <GPUSolver/AMGHierarchy.h>

namespace gpuSolver {

class AMGHierarchyBuilder {
public:
  AMGHierarchyBuilder(
      const AMGCoarsener &coarsener,
      int maxLevels,
      int minDofs);

  AMGHierarchy build(
      const AMGHierarchy::Matrix &matrix,
      const Eigen::MatrixXf *nearNullspace = nullptr) const;

private:
  Eigen::MatrixXf orthonormalize(
      const Eigen::MatrixXf &modes) const;

  const AMGCoarsener &coarsener_;

  int maxLevels_;
  int minDofs_;
};

} // namespace gpuSolver
