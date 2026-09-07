include(googletest)

add_executable(permutationUnittests
    PermutationUnittests.cc
)

target_link_libraries(permutationUnittests
    PRIVATE
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(permutationUnittests PROPERTIES
    FOLDER tests
)

add_executable(dotProductTests
  DotProductTests.cc
)

target_link_libraries(dotProductTests
    PRIVATE
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(dotProductTests PROPERTIES
    FOLDER tests
)


add_executable(scalarKernelTests
  ScalarKernelTests.cc
)

target_link_libraries(scalarKernelTests
    PRIVATE
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(scalarKernelTests PROPERTIES
    FOLDER tests
)


if(APPLE)
  add_subdirectory(metal)
endif()
