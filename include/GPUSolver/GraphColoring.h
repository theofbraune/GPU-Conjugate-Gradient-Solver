#pragma once
#include <iostream>
#include <vector>

namespace gpuSolver {

class GraphColoring {
  int nbColors_=0;
  std::vector<int> colors_;
  std::vector<int> colorVertices_;
  std::vector<int> colorOffsets_;

  bool isInitialized = false;
public:
  void compute(std::size_t n, const int *rowPtr, const int *colPtr);

  std::size_t numberOfColors() const;

  const std::vector<int> &colors() const;
  const std::vector<int> &colorVertices() const;
  const std::vector<int> &colorOffsets() const;
};
} // namespace gpuSolver
