#include <GPUSolver/GraphColoring.h>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace gpuSolver {

void GraphColoring::compute(std::size_t n, const int *rowPtr,
                            const int *colPtr) {

  this->isInitialized = true;

  this->colors_ = std::vector<int>(n, -1);
  this->colorVertices_ = std::vector<int>(n, -1);
  this->colorOffsets_ = std::vector<int>(n, -1);
  // this->colors_.
  std::vector<int> forbidden(n + 1, -1);

  for (std::size_t i = 0; i < n; i++) {
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
      int j = colPtr[k];
      if (j != i && colors_[j] >= 0) {
        forbidden[colors_[j]] = i;
      }
    }
    int c = 0;
    while (forbidden[c] == i) {
      c++;
    }
    colors_[i] = c;
    this->nbColors_ = std::max(nbColors_, c + 1);
  }

  std::vector<int> colorCounts(nbColors_, 0);

  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    const int color = colors_[vertex];

    ++colorCounts[static_cast<std::size_t>(color)];
  }

  // --------------------------------------------------------
  // Build offsets.
  //
  // colorOffsets_[c]     = first vertex of color c
  // colorOffsets_[c + 1] = one past last vertex of color c
  // --------------------------------------------------------

  colorOffsets_.assign(nbColors_ + 1, 0);

  for (std::size_t color = 0; color < nbColors_; ++color) {
    colorOffsets_[color + 1] = colorOffsets_[color] + colorCounts[color];
  }

  // --------------------------------------------------------
  // Fill grouped vertex list.
  // --------------------------------------------------------

  colorVertices_.assign(n, -1);

  std::vector<int> nextOffset = colorOffsets_;

  for (std::size_t vertex = 0; vertex < n; ++vertex) {
    const std::size_t color = static_cast<std::size_t>(colors_[vertex]);

    const int destination = nextOffset[color]++;

    colorVertices_[static_cast<std::size_t>(destination)] =
        static_cast<int>(vertex);
  }
}

std::size_t GraphColoring::numberOfColors() const { return this->nbColors_; }

const std::vector<int> &GraphColoring::colors() const {

  if (!this->isInitialized) {
    throw std::runtime_error(" the coloring was not initialized");
  }

  return this->colors_;
}

const std::vector<int> &GraphColoring::colorVertices() const {

  if (!this->isInitialized) {
    throw std::runtime_error(" the coloring was not initialized");
  }
  return this->colorVertices_;
}

const std::vector<int> &GraphColoring::colorOffsets() const {

  if (!this->isInitialized) {
    throw std::runtime_error(" the coloring was not initialized");
  }

  return this->colorOffsets_;
}

} // namespace gpuSolver
