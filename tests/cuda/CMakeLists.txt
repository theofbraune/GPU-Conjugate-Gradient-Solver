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
    COMMAND vectorKernelTestCuda
)


add_executable(
    spmvCusparseTest
    testSpmvCusparse.cc
)

target_link_libraries(
    spmvCusparseTest
    PRIVATE
        gpuSolverCuda
)

add_test(
    NAME spmvCusparseTest
    COMMAND spmvCusparseTest
)


add_executable(
    testCudaVectorOps
    testCudaVectorOps.cc
)

target_link_libraries(
    testCudaVectorOps
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
)

add_test(
    NAME testCudaVectorOps
    COMMAND testCudaVectorOps
)


add_executable(
    testCudaScalarOps
    testCudaScalarOps.cc
)

target_link_libraries(
    testCudaScalarOps
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
)

add_test(
    NAME testCudaScalarOps
    COMMAND testCudaScalarOps
)


add_executable(
    testCudaSpmv
    testCudaSpmv.cc
)

target_link_libraries(
    testCudaSpmv
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
)

add_test(
    NAME testCudaSpmv
    COMMAND testCudaSpmv
)


add_executable(
    testCudaGaussSeidel
    testCudaGaussSeidel.cc
)

target_link_libraries(
    testCudaGaussSeidel
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
)

add_test(
    NAME testCudaGaussSeidel
    COMMAND testCudaGaussSeidel
)


add_executable(
    testCudaBlockOps
    testCudaBlockOps.cc
)

target_link_libraries(
    testCudaBlockOps
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
)

add_test(
    NAME testCudaBlockOps
    COMMAND testCudaBlockOps
)


add_executable(
    testCudaLibigl
    testCudaLibigl.cc
)

target_link_libraries(
    testCudaLibigl
    PRIVATE
        gpuSolverCuda
        GTest::gtest_main
        igl::core
)

target_compile_definitions(
    testCudaLibigl
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(
    NAME testCudaLibigl
    COMMAND testCudaLibigl
)
