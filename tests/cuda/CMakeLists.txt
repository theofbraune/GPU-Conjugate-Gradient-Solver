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
