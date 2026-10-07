#pragma once 
#include <Eigen/Sparse>


class AMGCoarsener {
public:
  using Matrix =
      Eigen::SparseMatrix<float, Eigen::RowMajor>;

  struct Result {
    Matrix prolongation;
    Matrix restriction;
    Matrix coarseMatrix;
  };

  virtual ~AMGCoarsener() = default;

  virtual Result coarsen(
      const Matrix &matrix,
      const Eigen::MatrixXf *nearNullspace) const = 0;

  virtual int blockSize() const = 0;
};
