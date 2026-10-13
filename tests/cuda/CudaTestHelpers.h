#pragma once

#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceSparseMatrix.h>
#include <GPUSolver/DeviceVector.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace gpuSolverTest {

inline std::vector<float>
downloadVector(
    gpuSolver::DeviceVector &deviceVector)
{
  float *data =
      deviceVector.download();

  return std::vector<float>(
      data,
      data + deviceVector.size());
}


inline void
expectVectorNear(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float tolerance = 1.0e-5f)
{
  ASSERT_EQ(
      actual.size(),
      expected.size());

  for (std::size_t i = 0;
       i < actual.size();
       ++i) {

    EXPECT_NEAR(
        actual[i],
        expected[i],
        tolerance)
        << "Mismatch at index "
        << i;
  }
}


inline float
relativeError(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
{
  EXPECT_EQ(
      actual.size(),
      expected.size());

  double numerator = 0.0;
  double denominator = 0.0;

  for (std::size_t i = 0;
       i < actual.size();
       ++i) {

    const double difference =
        static_cast<double>(actual[i])
        - static_cast<double>(expected[i]);

    numerator +=
        difference * difference;

    denominator +=
        static_cast<double>(expected[i])
        * static_cast<double>(expected[i]);
  }

  if (denominator == 0.0) {
    return static_cast<float>(
        std::sqrt(numerator));
  }

  return static_cast<float>(
      std::sqrt(
          numerator / denominator));
}

} // namespace gpuSolverTest
