# ------------------------------------------------------------
# AMG solver tests
# ------------------------------------------------------------

include(GoogleTest)
add_executable(multigridUnitTests
    MultigridUnitTests.cc
)

target_link_libraries(multigridUnitTests
    PRIVATE
        gpuSolverAMG
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(multigridUnitTests
    PROPERTIES
        FOLDER "tests/amg"
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin/tests/amg"
)

include(libigl)

add_executable(benchmarkPermutations
  benchmarkDifferentPermutations.cc
)

target_link_libraries(benchmarkPermutations
    PRIVATE
        gpuSolverAMG
        gpuSolver::gpuSolver
        igl::core
)

set_target_properties(benchmarkPermutations
    PROPERTIES
        FOLDER "tests/amg"
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin/tests/amg"
)

add_executable(AMGgraphColoringUnittest
  amgColoringCheck.cc
)

target_link_libraries(AMGgraphColoringUnittest
    PRIVATE
        gpuSolver::gpuSolver
        gpuSolverAMG
        GTest::gtest_main
        igl::core
)

set_target_properties(AMGgraphColoringUnittest PROPERTIES
    FOLDER "tests/amg"
    RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/bin/tests/amg"
)
