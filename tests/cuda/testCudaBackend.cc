#include <GPUSolver/cuda/CudaBackend.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
  try {

    gpuSolver::CudaBackend backend;

    const std::vector<float> input = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f};

    std::vector<float> output;

    backend.timesTwo(
        input,
        output);

    if (output.size()
        != input.size()) {

      throw std::runtime_error(
          "Output has wrong size.");
    }

    for (std::size_t i = 0;
         i < input.size();
         ++i) {

      const float expected =
          2.0f * input[i];

      if (std::abs(
              output[i]
              - expected)
          > 1.0e-6f) {

        throw std::runtime_error(
            "CUDA result is incorrect.");
      }
    }

    std::cout
        << "CUDA backend smoke test: PASS\n";

    return 0;

  } catch (const std::exception &exception) {

    std::cerr
        << "CUDA backend smoke test: FAIL\n"
        << exception.what()
        << "\n";

    return 1;
  }
}
