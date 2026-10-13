#include <GPUSolver/CPU/CPUDeviceVector.h>

#include <algorithm>
#include <stdexcept>

namespace gpuSolver {

CPUDeviceVector::CPUDeviceVector(std::size_t size) : data_(size, 0.0f) {}

CPUDeviceVector::CPUDeviceVector(std::size_t size, const float *values)
    : data_(values, values + size) {
  if (values == nullptr && size != 0) {
    throw std::runtime_error(
        "CPUDeviceVector: nullptr values with nonzero size.");
  }
}

std::size_t CPUDeviceVector::size() const { return data_.size(); }

void CPUDeviceVector::updateValues(const float *values,
                                   std::size_t sizeOfValues) {
  if (sizeOfValues != data_.size()) {
    throw std::runtime_error(
        "CPUDeviceVector::updateValues: size mismatch.");
  }
  std::copy(values, values + sizeOfValues, data_.begin());
}

float *CPUDeviceVector::download() const { return data_.data(); }

float *CPUDeviceVector::data() { return data_.data(); }

const float *CPUDeviceVector::data() const { return data_.data(); }

} // namespace gpuSolver
