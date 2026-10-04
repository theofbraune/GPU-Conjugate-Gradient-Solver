#include <GPUSolver/Backend.h>
#include <GPUSolver/GraphColoring.h>
#include <GPUSolver/Smoothers/SymmetricGaussSeidelSmoother.h>

#include <cmath>
#include <stdexcept>

namespace gpuSolver {

SymmetricGaussSeidelSmoother::SymmetricGaussSeidelSmoother(
    float omega)
    : omega_(omega)
{
  if (!std::isfinite(omega_) ||
      omega_ <= 0.0f ||
      omega_ >= 2.0f) {
    throw std::runtime_error(
        "SymmetricGaussSeidelSmoother: invalid omega.");
  }
}

SymmetricGaussSeidelSmoother::~SymmetricGaussSeidelSmoother()
{
  delete colorVertices_;
}

void SymmetricGaussSeidelSmoother::initialize(
    Backend &backend,
    const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix)
{
  if (colorVertices_ != nullptr) {
    throw std::runtime_error(
        "SymmetricGaussSeidelSmoother: already initialized.");
  }

  matrix_ = &deviceMatrix;

  GraphColoring coloring;

  coloring.compute(
      hostMatrix.rows(),
      hostMatrix.activeRowPtr(),
      hostMatrix.activeColPtr()
  );

  numberOfColors_ = coloring.numberOfColors();

  colorOffsets_ = coloring.colorOffsets();

  colorVertices_ = backend.createIndexVector(
      coloring.colorVertices().size(),
      coloring.colorVertices().data()
  );
}

void SymmetricGaussSeidelSmoother::smooth(
    Backend &backend,
    BackendEncoder &encoder,
    const DeviceVector &rhs,
    DeviceVector &solution,
    std::size_t iterations)
{
  if (matrix_ == nullptr || colorVertices_ == nullptr) {
    throw std::runtime_error(
        "SymmetricGaussSeidelSmoother: not initialized.");
  }

  for (std::size_t iteration = 0;
       iteration < iterations;
       ++iteration) {

    // Forward sweep.
    for (std::size_t color = 0;
         color < numberOfColors_;
         ++color) {

      const std::size_t begin = colorOffsets_[color];

      const std::size_t count =
          colorOffsets_[color + 1] - begin;

      backend.encodeGaussSeidelColor(
          encoder,
          *matrix_,
          *colorVertices_,
          begin,
          count,
          rhs,
          solution,
          omega_
      );
    }

    // Backward sweep.
    for (std::size_t color = numberOfColors_;
         color > 0;
         --color) {

      const std::size_t currentColor = color - 1;

      const std::size_t begin =
          colorOffsets_[currentColor];

      const std::size_t count =
          colorOffsets_[currentColor + 1] - begin;

      backend.encodeGaussSeidelColor(
          encoder,
          *matrix_,
          *colorVertices_,
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
