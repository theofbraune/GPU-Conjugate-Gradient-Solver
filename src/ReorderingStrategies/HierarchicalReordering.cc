#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/ReorderingStrategies/HierarchicalReordering.h>
#include <GPUSolver/ReorderingStrategies/RCMReordering.h>
#include <algorithm>
#include <iostream>

/**
 * Given a restriction matrix n_coase x n_fine, we compute for each fine node
 * the parent node in the coarse setting that has the hightest weight, i.e the
 * one with the closest proximity.
 */
static std::vector<int>
computeStrongestParents(const gpuSolver::AMGHierarchy::Matrix &R) {
  const std::size_t nCoarse = static_cast<std::size_t>(R.rows());

  const std::size_t nFine = static_cast<std::size_t>(R.cols());

  std::vector<int> parent(nFine, -1);

  std::vector<float> strongestWeight(nFine, -1.0f);

  for (std::size_t coarse = 0; coarse < nCoarse; ++coarse) {
    for (gpuSolver::AMGHierarchy::Matrix::InnerIterator entry(
             R, static_cast<int>(coarse));
         entry; ++entry) {
      const std::size_t fine = static_cast<std::size_t>(entry.col());

      const float weight = std::abs(entry.value());

      if (weight > strongestWeight[fine]) {
        strongestWeight[fine] = weight;

        parent[fine] = static_cast<int>(coarse);
      }
    }
  }

  std::size_t orphanCount = 0;

  for (std::size_t fine = 0; fine < nFine; ++fine) {
    if (parent[fine] < 0) {
      ++orphanCount;
    }
  }

  if (orphanCount > 0) {
    std::cout << "HierarchicalReordering: " << orphanCount << " / " << nFine
              << " fine DOFs have no parent in restriction matrix.\n";
  }

  return parent;
}
static gpuSolver::Permutation
computeRCM(const gpuSolver::AMGHierarchy::Matrix &A) {
  const std::size_t n = static_cast<std::size_t>(A.rows());

  int *oldToNew = new int[n];

  int *newToOld = new int[n];

  gpuSolver::RCMReordering rcm;

  rcm.compute(n, A.outerIndexPtr(), A.innerIndexPtr(), oldToNew, newToOld);

  gpuSolver::Permutation permutation(n, oldToNew, newToOld);

  delete[] oldToNew;
  delete[] newToOld;

  return permutation;
}

static gpuSolver::Permutation buildFinePermutation(
    const gpuSolver::AMGHierarchy::Matrix& fineMatrix,
    const gpuSolver::AMGHierarchy::Matrix& restriction,
    const gpuSolver::Permutation& coarsePermutation)
{
    const std::size_t nFine =
        static_cast<std::size_t>(
            fineMatrix.rows()
        );

    const std::size_t nCoarse =
        static_cast<std::size_t>(
            restriction.rows()
        );

    if (fineMatrix.rows() != fineMatrix.cols())
    {
        throw std::runtime_error(
            "HierarchicalReordering: "
            "fine matrix must be square."
        );
    }

    if (restriction.cols()
        != fineMatrix.rows())
    {
        throw std::runtime_error(
            "HierarchicalReordering: "
            "restriction dimensions do not match fine matrix."
        );
    }

    const std::vector<int> parent =
        computeStrongestParents(
            restriction
        );

    // Independent RCM ordering on this fine level.
    //
    // We use this only as a secondary ordering inside
    // each coarse-parent group.
    const gpuSolver::Permutation fineRCM =
        computeRCM(
            fineMatrix
        );

    const int* coarseOldToNew =
        coarsePermutation.oldToNew();

    const int* fineRCMOldToNew =
        fineRCM.oldToNew();

    // Start with old indices.
    std::vector<int> newToOld(
        nFine
    );

    for (std::size_t i = 0;
         i < nFine;
         ++i)
    {
        newToOld[i] =
            static_cast<int>(i);
    }

    // --------------------------------------------------
    // Sort key:
    //
    //  1. rank of coarse parent
    //  2. fine-level RCM rank
    //
    // Fine DOFs without a parent are placed after all
    // parented DOFs, but still ordered by fine-level RCM.
    // --------------------------------------------------

    std::sort(
        newToOld.begin(),
        newToOld.end(),
        [&](int a, int b)
        {
            const int parentA =
                parent[
                    static_cast<std::size_t>(a)
                ];

            const int parentB =
                parent[
                    static_cast<std::size_t>(b)
                ];

            const int parentRankA =
                parentA >= 0
                    ? coarseOldToNew[parentA]
                    : static_cast<int>(nCoarse);

            const int parentRankB =
                parentB >= 0
                    ? coarseOldToNew[parentB]
                    : static_cast<int>(nCoarse);

            if (parentRankA != parentRankB)
            {
                return parentRankA
                     < parentRankB;
            }

            return fineRCMOldToNew[a]
                 < fineRCMOldToNew[b];
        }
    );

    // --------------------------------------------------
    // Construct inverse map.
    //
    // newToOld[new] = old
    // oldToNew[old] = new
    // --------------------------------------------------

    int* oldToNew =
        new int[nFine];

    for (std::size_t newIndex = 0;
         newIndex < nFine;
         ++newIndex)
    {
        const int oldIndex =
            newToOld[newIndex];

        oldToNew[oldIndex] =
            static_cast<int>(newIndex);
    }

    gpuSolver::Permutation permutation(
        nFine,
        oldToNew,
        newToOld.data()
    );

    delete[] oldToNew;

    return permutation;
}


std::vector<gpuSolver::Permutation> gpuSolver::HierarchicalReordering::compute(
    const AMGHierarchy &hierarchy) const {
  const std::size_t numberOfLevels = hierarchy.A.size();

  if (numberOfLevels == 0) {
    throw std::runtime_error("HierarchicalReordering: empty hierarchy.");
  }

  if (hierarchy.R.size() + 1 != numberOfLevels) {
    throw std::runtime_error("HierarchicalReordering: "
                             "expected one restriction matrix "
                             "between consecutive levels.");
  }

  // --------------------------------------------------
  // First construct in the natural order:
  //
  //     coarsest -> finest
  //
  // No default construction is needed.
  // --------------------------------------------------

  std::vector<Permutation> coarseToFine;

  coarseToFine.reserve(numberOfLevels);

  const std::size_t coarsestLevel = numberOfLevels - 1;

  coarseToFine.emplace_back(computeRCM(hierarchy.A[coarsestLevel]));

  // --------------------------------------------------
  // Propagate from coarse toward fine.
  //
  // coarseToFine.back() always contains the
  // permutation of the current coarse level.
  // --------------------------------------------------

  for (std::size_t coarseLevel = coarsestLevel; coarseLevel > 0;
       --coarseLevel) {
    const std::size_t fineLevel = coarseLevel - 1;

    coarseToFine.emplace_back(buildFinePermutation(
        hierarchy.A[fineLevel], hierarchy.R[fineLevel], coarseToFine.back()));
  }

  // --------------------------------------------------
  // Convert to our normal hierarchy convention:
  //
  //     permutations[0] = finest
  //     permutations[L] = coarsest
  // --------------------------------------------------

  std::vector<Permutation> permutations;

  permutations.reserve(numberOfLevels);

  for (std::size_t i = numberOfLevels; i > 0; --i) {
    permutations.emplace_back(std::move(coarseToFine[i - 1]));
  }

  return permutations;
}
