#include <GPUSolver/Backend.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/GraphColoring.h>
#include <GPUSolver/Smoothers/SymmetricGaussSeidelSmoother.h>
#include <glog/logging.h>

#include <cmath>
#include <stdexcept>

namespace gpuSolver {

SymmetricGaussSeidelSmoother::SymmetricGaussSeidelSmoother(float omega)
    : omega_(omega) {
  if (!std::isfinite(omega_) || omega_ <= 0.0f || omega_ >= 2.0f) {
    throw std::runtime_error("SymmetricGaussSeidelSmoother: invalid omega.");
  }
}

SymmetricGaussSeidelSmoother::~SymmetricGaussSeidelSmoother() {
  delete colorVertices_;
}

void SymmetricGaussSeidelSmoother::initialize(
    Backend &backend, const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix) {
  if (colorVertices_ != nullptr) {
    throw std::runtime_error(
        "SymmetricGaussSeidelSmoother: already initialized.");
  }

  matrix_ = &deviceMatrix;

  GraphColoring coloring;

  coloring.compute(hostMatrix.rows(), hostMatrix.activeRowPtr(),
                   hostMatrix.activeColPtr());

  numberOfColors_ = coloring.numberOfColors();

  colorOffsets_ = coloring.colorOffsets();

  colorVertices_ = backend.createIndexVector(coloring.colorVertices().size(),
                                             coloring.colorVertices().data());
  // LOG(INFO) << "SGS initialize: rows = " << hostMatrix.rows();
  //
  // LOG(INFO) << "SGS colors = " << numberOfColors_;
  //
  // LOG(INFO) << "SGS colorVertices = " << colorVertices_->size();
  //
  // LOG(INFO) << "SGS colorOffsets size = " << colorOffsets_.size();
  const std::vector<int> cpuColors = coloring.colorVertices();

  colorVertices_ =
      backend.createIndexVector(cpuColors.size(), cpuColors.data());

  const int *gpuColors = colorVertices_->download();

  for (std::size_t i = 0; i < cpuColors.size(); ++i) {
    // LOG(INFO)<<" i : color_cpu: "<<cpuColors[i]<<" gpu color "<<gpuColors[i];

    if (gpuColors[i] != cpuColors[i]) {

      LOG(ERROR) << "Color upload mismatch at " << i << ": CPU=" << cpuColors[i]
                 << ", GPU=" << gpuColors[i];

      throw std::runtime_error("DeviceIndexVector upload is broken.");
    }
  }
  for (std::size_t color = 0;
       color < std::min<std::size_t>(numberOfColors_, 10); ++color) {

    const int begin = colorOffsets_[color];

    const int end = colorOffsets_[color + 1];

    // LOG(INFO) << "color " << color << ": " << end - begin << " rows";
  }
}

void SymmetricGaussSeidelSmoother::smooth(Backend &backend,
                                          BackendEncoder &encoder,
                                          const DeviceVector &rhs,
                                          DeviceVector &solution,
                                          std::size_t iterations) {
  if (matrix_ == nullptr || colorVertices_ == nullptr) {
    throw std::runtime_error("SymmetricGaussSeidelSmoother: not initialized.");
  }

  if (iterations == 0) {
    throw std::runtime_error(
        "Preconditioner requires at least one smoothing step.");
  }
  // LOG(INFO) << " smoothing itereations are " << iterations;
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {

    // Forward sweep.
    for (std::size_t color = 0; color < numberOfColors_; ++color) {

      const std::size_t begin = colorOffsets_[color];

      const std::size_t count = colorOffsets_[color + 1] - begin;
      // LOG(INFO) << "Encoding SGS color " << color << ", begin = " << begin
      //           << ", count = " << count;
      backend.encodeGaussSeidelColor(encoder, *matrix_, *colorVertices_, begin,
                                     count, rhs, solution, omega_);

      // backend.submitAndWait(encoder);
      // float *checkSolution = solution.download();
      //
      // Eigen::Map<Eigen::VectorXf> zCheck(checkSolution,solution.size());
      // LOG(INFO) << " norm of the vector after first color in GS sweep is " <<
      // zCheck.norm();
    }

    // Backward sweep.
    for (std::size_t color = numberOfColors_; color > 0; --color) {

      const std::size_t currentColor = color - 1;

      const std::size_t begin = colorOffsets_[currentColor];

      const std::size_t count = colorOffsets_[currentColor + 1] - begin;

      backend.encodeGaussSeidelColor(encoder, *matrix_, *colorVertices_, begin,
                                     count, rhs, solution, omega_);
    }
  }
}

} // namespace gpuSolver
