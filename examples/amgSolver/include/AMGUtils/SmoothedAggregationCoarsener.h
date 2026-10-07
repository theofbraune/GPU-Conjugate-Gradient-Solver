#pragma once
#include <AMGUtils/AMGCoarsener.h>

namespace gpuSolver {

class SmoothedAggregationCoarsener : public AMGCoarsener {
public:
  SmoothedAggregationCoarsener(int blockSize, float epsStrong)
      : blockSize_(blockSize), epsStrong_(epsStrong) {
    if (blockSize_ <= 0) {
      throw std::runtime_error("block size must be positive.");
    }
  }

  Result coarsen(const Matrix &matrix,
                 const Eigen::MatrixXf *nearNullspace) const override;

  int blockSize() const override { return blockSize_; }

private:
  int blockSize_;
  float epsStrong_;
};
} // namespace gpuSolver
