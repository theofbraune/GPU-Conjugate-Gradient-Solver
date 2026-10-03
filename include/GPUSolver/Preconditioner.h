#pragma once

#include "GPUSolver/HostSparseMatrix.h"

namespace gpuSolver {

class Backend;
class BackendEncoder;
class DeviceVector;
class HostCSRMatrix;
class DeviceCSRMatrix;

class Preconditioner {
public:
  virtual ~Preconditioner() = default;
  virtual void initialize(Backend &backend, const HostCSRMatrix &hostMatrix,
                          const DeviceCSRMatrix &deviceMatrix) = 0;
  virtual void apply(Backend &backend, BackendEncoder &encoder,
                     const DeviceVector &input, DeviceVector &output) = 0;
};

} // namespace gpuSolver
