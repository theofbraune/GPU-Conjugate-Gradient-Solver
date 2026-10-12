add_executable(
    testCudaBackend
    testCudaBackend.cc
)

target_link_libraries(
    testCudaBackend
    PRIVATE
        gpuSolverCuda
)

add_test(
    NAME cudaBackendSmokeTest
    COMMAND testCudaBackend
)

add_executable(
  vectorKernelTestCuda
  testVectorKernel.cc
)

target_link_libraries(
  vectorKernelTestCuda
  PRIVATE
  gpuSolverCuda
)

add_test(
  NAME vectorKernelTestCuda
  COMMAND testVectorKernelCuda
)
