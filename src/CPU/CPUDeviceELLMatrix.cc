#include <GPUSolver/CPU/CPUDeviceELLMatrix.h>

#include <GPUSolver/HostELLMatrix.h>

namespace gpuSolver {

CPUDeviceELLMatrix::CPUDeviceELLMatrix(const HostELLMatrix &hostMatrix)
    : nRows_(hostMatrix.rows()), nCols_(hostMatrix.cols()),
      ellWidth_(hostMatrix.ellWidth()),
      overflowNnz_(hostMatrix.overflowNnz()),
      ellColIdx_(hostMatrix.ellColIdx(),
                hostMatrix.ellColIdx() + hostMatrix.ellWidth() * hostMatrix.rows()),
      ellValues_(hostMatrix.ellValues(),
                hostMatrix.ellValues() + hostMatrix.ellWidth() * hostMatrix.rows()),
      overflowRowPtr_(hostMatrix.overflowRowPtr(),
                     hostMatrix.overflowRowPtr() + hostMatrix.rows() + 1),
      overflowColIdx_(hostMatrix.overflowColIdx(),
                      hostMatrix.overflowColIdx() + hostMatrix.overflowNnz()),
      overflowValues_(hostMatrix.overflowValues(),
                      hostMatrix.overflowValues() + hostMatrix.overflowNnz()) {
}

std::size_t CPUDeviceELLMatrix::rows() const { return nRows_; }
std::size_t CPUDeviceELLMatrix::cols() const { return nCols_; }
std::size_t CPUDeviceELLMatrix::ellWidth() const { return ellWidth_; }
std::size_t CPUDeviceELLMatrix::overflowNnz() const { return overflowNnz_; }

const int *CPUDeviceELLMatrix::ellColIdx() const { return ellColIdx_.data(); }
const float *CPUDeviceELLMatrix::ellValues() const {
  return ellValues_.data();
}
const int *CPUDeviceELLMatrix::overflowRowPtr() const {
  return overflowRowPtr_.data();
}
const int *CPUDeviceELLMatrix::overflowColIdx() const {
  return overflowColIdx_.data();
}
const float *CPUDeviceELLMatrix::overflowValues() const {
  return overflowValues_.data();
}

} // namespace gpuSolver
