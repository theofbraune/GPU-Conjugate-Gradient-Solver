#include <GPUSolver/CPU/CPUDeviceIndexVector.h>

#include <algorithm>
#include <stdexcept>

namespace gpuSolver {

CPUDeviceIndexVector::CPUDeviceIndexVector(std::size_t size)
    : data_(size, 0) {}

CPUDeviceIndexVector::CPUDeviceIndexVector(std::size_t size,
                                           const int *values)
    : data_(values, values + size) {
  if (values == nullptr && size != 0) {
    throw std::runtime_error(
        "CPUDeviceIndexVector: nullptr values with nonzero size.");
  }
}

std::size_t CPUDeviceIndexVector::size() const { return data_.size(); }
std::size_t CPUDeviceIndexVector::size() { return data_.size(); }

void CPUDeviceIndexVector::updateValues(const int *values,
                                        std::size_t sizeOfValues) {
  if (sizeOfValues != data_.size()) {
    throw std::runtime_error(
        "CPUDeviceIndexVector::updateValues: size mismatch.");
  }
  std::copy(values, values + sizeOfValues, data_.begin());
}

int *CPUDeviceIndexVector::download() const { return data_.data(); }

int *CPUDeviceIndexVector::data() { return data_.data(); }

const int *CPUDeviceIndexVector::data() const { return data_.data(); }

} // namespace gpuSolver
