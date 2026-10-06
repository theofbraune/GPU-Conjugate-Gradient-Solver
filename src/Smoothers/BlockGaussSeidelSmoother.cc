#include "GPUSolver/DeviceSparseMatrix.h"
#include "GPUSolver/HostSparseMatrix.h"
#include <GPUSolver/Smoothers/BlockGaussSeidelSmoother.h>
#include <GPUSolver/BlockUtils.h>
#include <GPUSolver/GraphColoring.h>


namespace gpuSolver {

BlockGaussSeidelSmoother::BlockGaussSeidelSmoother(
    float omega)
    : omega_(omega)
{
  if (omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error(
        "BlockGaussSeidelSmoother: "
        "omega must be in (0, 2)."
    );
  }
}

BlockGaussSeidelSmoother::~BlockGaussSeidelSmoother()
{
  delete inverseDiagonalBlocks_;
  delete colorBlocks_;
}


void BlockGaussSeidelSmoother::initialize(
    Backend &backend,
    const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix)
{
  if (matrix_ != nullptr) {
    throw std::runtime_error(
        "BlockGaussSeidelSmoother: "
        "already initialized."
    );
  }

  if (hostMatrix.rows() % 3 != 0) {
    throw std::runtime_error(
        "BlockGaussSeidelSmoother: "
        "number of DOFs must be divisible by 3."
    );
  }

  // Borrow the already permuted device matrix.
  matrix_ = &deviceMatrix;

  // ------------------------------------------------------
  // 1. Extract and invert the diagonal 3x3 blocks.
  // ------------------------------------------------------

  const std::vector<float> inverseBlocks =
      extractInverseDiagonalBlocks3x3(
          hostMatrix
      );

  inverseDiagonalBlocks_ =
      backend.createVector(
          inverseBlocks.size(),
          inverseBlocks.data()
      );
  // inverseDiagonalBlocks_ = backend.

  // ------------------------------------------------------
  // 2. Build node/block adjacency graph.
  //
  // One block = one node = 3 scalar DOFs.
  // ------------------------------------------------------

  BlockGraph blockGraph =
      buildBlockGraph(
          hostMatrix,
          3
      );

  const std::size_t numberOfBlocks =
      blockGraph.rowPtr.size() - 1;

  // ------------------------------------------------------
  // 3. Color the block graph.
  // ------------------------------------------------------

  GraphColoring coloring;

  coloring.compute(
      numberOfBlocks,
      blockGraph.rowPtr.data(),
      blockGraph.colPtr.data()
  );

  numberOfColors_ =
      coloring.numberOfColors();

  colorOffsets_ =
      coloring.colorOffsets();

  // ------------------------------------------------------
  // 4. Upload the block indices grouped by color.
  // ------------------------------------------------------

  colorBlocks_ =
      backend.createIndexVector(
          coloring.colorVertices().size(),
          coloring.colorVertices().data()
      );
}

void BlockGaussSeidelSmoother::smooth(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &rhs,
    DeviceVector &solution,
    std::size_t iterations)
{
  if (matrix_ == nullptr ||
      inverseDiagonalBlocks_ == nullptr ||
      colorBlocks_ == nullptr) {
    throw std::runtime_error(
        "BlockGaussSeidelSmoother: "
        "not initialized."
    );
  }

  for (std::size_t iteration = 0;
       iteration < iterations;
       ++iteration) {

    // ----------------------------------------------------
    // Forward sweep.
    // ----------------------------------------------------

    for (std::size_t color = 0;
         color < numberOfColors_;
         ++color) {

      const std::size_t begin =
          static_cast<std::size_t>(
              colorOffsets_[color]
          );

      const std::size_t end =
          static_cast<std::size_t>(
              colorOffsets_[color + 1]
          );

      const std::size_t count =
          end - begin;

      if (count == 0) {
        continue;
      }

      backend.encodeBlockGaussSeidelColor3x3(
          encoder,
          *matrix_,
          *colorBlocks_,
          *inverseDiagonalBlocks_,
          begin,
          count,
          rhs,
          solution,
          omega_
      );
    }

    // ----------------------------------------------------
    // Backward sweep.
    // ----------------------------------------------------

    for (std::size_t color = numberOfColors_;
         color > 0;
         --color) {

      const std::size_t currentColor =
          color - 1;

      const std::size_t begin =
          static_cast<std::size_t>(
              colorOffsets_[currentColor]
          );

      const std::size_t end =
          static_cast<std::size_t>(
              colorOffsets_[currentColor + 1]
          );

      const std::size_t count =
          end - begin;

      if (count == 0) {
        continue;
      }

      backend.encodeBlockGaussSeidelColor3x3(
          encoder,
          *matrix_,
          *colorBlocks_,
          *inverseDiagonalBlocks_,
          begin,
          count,
          rhs,
          solution,
          omega_
      );
    }
  }
}

} // namespace gpuSolver
