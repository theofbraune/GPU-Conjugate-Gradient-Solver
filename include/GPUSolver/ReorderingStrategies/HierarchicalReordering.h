#pragma once

#include <GPUSolver/AMGHierarchy.h>
#include <GPUSolver/Permutation.h>

#include <vector>

namespace gpuSolver{

struct HierarchicalReordering{

  std::vector<Permutation> compute(const AMGHierarchy& hierarchy) const;


};

}
