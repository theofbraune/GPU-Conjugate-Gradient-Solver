# ------------------------------------------------------------
# AMG solver tests
# ------------------------------------------------------------

include(GoogleTest)
add_executable(multigridTests
    MultigridTests.cc
)

target_link_libraries(multigridTests
    PRIVATE
        gpuSolverAMG
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(multigridTests
    PROPERTIES
        FOLDER "tests/amg"
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin/tests/amg"
)

