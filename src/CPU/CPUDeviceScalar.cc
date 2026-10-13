#include <GPUSolver/CPU/CPUDeviceScalar.h>

namespace gpuSolver {

CPUDeviceScalar::CPUDeviceScalar() : value_(0.0f) {}

CPUDeviceScalar::CPUDeviceScalar(float value) : value_(value) {}

float CPUDeviceScalar::download() const { return value_; }

float CPUDeviceScalar::value() const { return value_; }

void CPUDeviceScalar::setValue(float value) { value_ = value; }

} // namespace gpuSolver
