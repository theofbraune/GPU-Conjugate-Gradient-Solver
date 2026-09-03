#include <GPUSolver/ReorderingStrategies/IdentityReordering.h>

namespace gpuSolver {

void IdentityReordering::compute(std::size_t nRows, const int *rowPtr,
                                 const int *colPtr, int *oldToNew,
                                 int *newToOld) const {

  (void)rowPtr;
  (void)colPtr;

  for (std::size_t i = 0; i < nRows; ++i) {
    oldToNew[i] = static_cast<int>(i);

    newToOld[i] = static_cast<int>(i);
  }
}
} // namespace gpuSolver
