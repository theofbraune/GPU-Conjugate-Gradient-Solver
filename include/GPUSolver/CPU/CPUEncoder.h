#pragma once

#include <GPUSolver/BackendEncoder.h>

namespace gpuSolver {

class CPUEncoder final : public BackendEncoder {
public:
  CPUEncoder() = default;
  ~CPUEncoder() override = default;

  CPUEncoder(const CPUEncoder &) = delete;
  CPUEncoder &operator=(const CPUEncoder &) = delete;
};

} // namespace gpuSolver
