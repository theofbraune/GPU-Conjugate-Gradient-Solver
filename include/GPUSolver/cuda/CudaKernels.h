#pragma once

#include <cstddef>

namespace gpuSolver {
namespace cuda {

void launchTimesTwo(
    const float *input,
    float *output,
    std::size_t size);

} // namespace cuda
} // namespace gpuSolver
