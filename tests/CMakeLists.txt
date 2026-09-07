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

include(libigl)

add_executable(laplacianTestCG 
  mainCGSolverTest.cc
)
target_link_libraries(laplacianTestCG
  PRIVATE 
  gpuSolver::gpuSolver
  igl::core
)

target_compile_definitions(laplacianTestCG
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(APPLE)
  add_subdirectory(metal)
endif()
