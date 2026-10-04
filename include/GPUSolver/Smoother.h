#pragma once
#include "GPUSolver/HostSparseMatrix.h"
#include <cstddef>

namespace gpuSolver {
class Backend;
class DeviceVector;
class BackendEncoder;
class DeviceCSRMatrix;
class HostCSRMatrix;

class Smoother {
public:
  virtual ~Smoother() = default;

  virtual void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                  const DeviceCSRMatrix &deviceMatrix) =0;

  virtual void smooth(Backend &backend, BackendEncoder &encoder,
                      const DeviceVector &rhs, DeviceVector &solution,
                      std::size_t iterations) = 0;
};
} // namespace gpuSolver
