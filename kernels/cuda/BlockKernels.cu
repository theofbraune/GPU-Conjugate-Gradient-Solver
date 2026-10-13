#include <GPUSolver/cuda/BlockKernels.h>
#include <GPUSolver/cuda/CudaHelper.h>

namespace {

constexpr int threadsPerBlock = 256;


__global__
void applyBlockInverses3x3Kernel(
    const float *inverseDiagBlocks,
    const float *residual,
    float *correction,
    std::size_t numberOfBlocks)
{
  const std::size_t blockIndex =
      static_cast<std::size_t>(blockIdx.x)
          * static_cast<std::size_t>(blockDim.x)
      + static_cast<std::size_t>(threadIdx.x);

  if (blockIndex >= numberOfBlocks) {
    return;
  }

  const std::size_t vectorOffset =
      3 * blockIndex;

  const std::size_t matrixOffset =
      9 * blockIndex;

  const float r0 =
      residual[vectorOffset + 0];

  const float r1 =
      residual[vectorOffset + 1];

  const float r2 =
      residual[vectorOffset + 2];

  const float *inverse =
      inverseDiagBlocks + matrixOffset;

  correction[vectorOffset + 0] =
      inverse[0] * r0
      + inverse[1] * r1
      + inverse[2] * r2;

  correction[vectorOffset + 1] =
      inverse[3] * r0
      + inverse[4] * r1
      + inverse[5] * r2;

  correction[vectorOffset + 2] =
      inverse[6] * r0
      + inverse[7] * r1
      + inverse[8] * r2;
}

__global__
void blockGaussSeidelColor3x3Kernel(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorBlocks,
    const float *diagonalInverses,
    std::size_t begin,
    std::size_t end,
    const float *rhs,
    float *solution,
    float omega)
{
  const std::size_t localIndex =
      static_cast<std::size_t>(blockIdx.x)
          * static_cast<std::size_t>(blockDim.x)
      + static_cast<std::size_t>(threadIdx.x);

  const std::size_t count =
      end - begin;

  if (localIndex >= count) {
    return;
  }

  const int blockRow =
      colorBlocks[begin + localIndex];

  const int firstScalarRow =
      3 * blockRow;

  float residual0 =
      rhs[firstScalarRow + 0];

  float residual1 =
      rhs[firstScalarRow + 1];

  float residual2 =
      rhs[firstScalarRow + 2];

  // --------------------------------------------------
  // Scan the three scalar CSR rows belonging to this
  // 3x3 block row.
  //
  // Contributions belonging to the diagonal block are
  // skipped because A_ii^{-1} is already supplied.
  // --------------------------------------------------

  for (int component = 0;
       component < 3;
       ++component) {

    const int scalarRow =
        firstScalarRow + component;

    float *residualComponent = nullptr;

    if (component == 0) {
      residualComponent = &residual0;
    } else if (component == 1) {
      residualComponent = &residual1;
    } else {
      residualComponent = &residual2;
    }

    for (int k = rowPtr[scalarRow];
         k < rowPtr[scalarRow + 1];
         ++k) {

      const int column =
          colIdx[k];

      const int columnBlock =
          column / 3;

      if (columnBlock == blockRow) {
        continue;
      }

      *residualComponent -=
          values[k] * solution[column];
    }
  }

  // --------------------------------------------------
  // Apply the precomputed inverse 3x3 diagonal block.
  // --------------------------------------------------

  const float *inverse =
      diagonalInverses
      + 9 * blockRow;

  const float new0 =
      inverse[0] * residual0
      + inverse[1] * residual1
      + inverse[2] * residual2;

  const float new1 =
      inverse[3] * residual0
      + inverse[4] * residual1
      + inverse[5] * residual2;

  const float new2 =
      inverse[6] * residual0
      + inverse[7] * residual1
      + inverse[8] * residual2;

  // --------------------------------------------------
  // Weighted Gauss-Seidel update.
  // --------------------------------------------------

  const float old0 =
      solution[firstScalarRow + 0];

  const float old1 =
      solution[firstScalarRow + 1];

  const float old2 =
      solution[firstScalarRow + 2];

  solution[firstScalarRow + 0] =
      (1.0f - omega) * old0
      + omega * new0;

  solution[firstScalarRow + 1] =
      (1.0f - omega) * old1
      + omega * new1;

  solution[firstScalarRow + 2] =
      (1.0f - omega) * old2
      + omega * new2;
}

}

namespace gpuSolver{
namespace cuda{
void launchApplyBlockInverses3x3(
    const float *inverseDiagBlocks,
    const float *residual,
    float *correction,
    std::size_t numberOfBlocks,
    cudaStream_t stream)
{
  if (numberOfBlocks == 0) {
    return;
  }

  const int numberOfThreadBlocks =
      static_cast<int>(
          (numberOfBlocks + threadsPerBlock - 1)
          / threadsPerBlock);

  applyBlockInverses3x3Kernel<<<
      numberOfThreadBlocks,
      threadsPerBlock,
      0,
      stream>>>(
          inverseDiagBlocks,
          residual,
          correction,
          numberOfBlocks);

  cuda::check(
      cudaGetLastError(),
      "applyBlockInverses3x3Kernel launch failed");
}

void launchBlockGaussSeidelColor3x3(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const int *colorBlocks,
    const float *diagonalInverses,
    std::size_t begin,
    std::size_t end,
    const float *rhs,
    float *solution,
    float omega,
    cudaStream_t stream)
{
  if (end <= begin) {
    return;
  }

  const std::size_t count =
      end - begin;

  const int numberOfThreadBlocks =
      static_cast<int>(
          (count + threadsPerBlock - 1)
          / threadsPerBlock);

  blockGaussSeidelColor3x3Kernel<<<
      numberOfThreadBlocks,
      threadsPerBlock,
      0,
      stream>>>(
          rowPtr,
          colIdx,
          values,
          colorBlocks,
          diagonalInverses,
          begin,
          end,
          rhs,
          solution,
          omega);

  cuda::check(
      cudaGetLastError(),
      "blockGaussSeidelColor3x3Kernel launch failed");
}

}
}
