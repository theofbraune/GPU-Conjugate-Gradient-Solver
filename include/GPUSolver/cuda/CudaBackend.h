#pragma once

#include <cstddef>
#include <vector>

namespace gpuSolver {

class CudaBackend {
public:
  CudaBackend();

  void timesTwo(
      const std::vector<float> &input,
      std::vector<float> &output) const;
};

} // namespace gpuSolver
