#include <GPUSolver/Backend.h>
#include <GPUSolver/GraphColoring.h>
#include <GPUSolver/Preconditioners/SymmetricGaussSeidelPreconditioner.h>
#include <GPUSolver/DeviceIndexVector.h>

namespace gpuSolver {

void SymmetricGaussSeidelPreconditioner::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {
  if (colorVertices_ != nullptr) {
    throw std::runtime_error("SymmetricGaussSeidelPreconditioner: "
                             "already initialized.");
  }

  // Borrow the matrix.
  matrix_ = &deviceMatrix;

  // Compute coloring of the ACTIVE CSR representation.
  GraphColoring coloring;

  coloring.compute(hostMatrix.rows(), hostMatrix.activeRowPtr(),
                   hostMatrix.activeColPtr());

  numberOfColors_ = coloring.numberOfColors();

  colorOffsets_ = coloring.colorOffsets();

  // Upload the grouped vertex indices.
  colorVertices_ = backend.createIndexVector(coloring.colorVertices().size(),
                                             coloring.colorVertices().data());
}

SymmetricGaussSeidelPreconditioner::SymmetricGaussSeidelPreconditioner(
    float omega)
    : matrix_(nullptr), colorVertices_(nullptr), numberOfColors_(0),
      omega_(omega) {
  if (omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error("SymmetricGaussSeidelPreconditioner: "
                             "omega must be between 0 and 2.");
  }
}

void SymmetricGaussSeidelPreconditioner::smooth(Backend &backend,
                                                BackendEncoder &encoder,
                                                const DeviceVector &rhs,
                                                DeviceVector &solution,
                                                std::size_t iterations) {
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t color = 0; color < numberOfColors_; ++color) {
      const std::size_t begin = colorOffsets_[color];

      const std::size_t count = colorOffsets_[color + 1] - begin;

      backend.encodeGaussSeidelColor(encoder, *matrix_, *colorVertices_, begin,
                                     count, rhs, solution, omega_);
    }

    for (std::size_t color = numberOfColors_; color > 0; --color) {
      const std::size_t currentColor = color - 1;

      const std::size_t begin = colorOffsets_[currentColor];

      const std::size_t count = colorOffsets_[currentColor + 1] - begin;

      backend.encodeGaussSeidelColor(encoder, *matrix_, *colorVertices_, begin,
                                     count, rhs, solution, omega_);
    }
  }
}

void SymmetricGaussSeidelPreconditioner::apply(Backend &backend,
                                               BackendEncoder &encoder,
                                               const DeviceVector &input,
                                               DeviceVector &output) {

  backend.encodeSetZero(encoder, output);

  this->smooth(backend, encoder, input, output, 1);
}

SymmetricGaussSeidelPreconditioner::~SymmetricGaussSeidelPreconditioner() {
  delete colorVertices_;
}
} // namespace gpuSolver
