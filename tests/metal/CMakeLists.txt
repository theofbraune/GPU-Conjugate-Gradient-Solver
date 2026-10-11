
include(googletest)
add_executable(testMetalDevice
  testMetalDevice.cc
)

target_link_libraries(testMetalDevice
  PRIVATE
    metal::metal
)


set_target_properties(testMetalDevice PROPERTIES 
  FOLDER tests
)


add_executable(testMetalTimesTwo
    testMetalTimesTwo.cc
)

target_link_libraries(testMetalTimesTwo
    PRIVATE
        gpuSolver::gpuSolver
)




include(libigl)

add_executable(benchmarkSpmv
    benchmarkSpmv.cc
)

target_link_libraries(benchmarkSpmv
    PRIVATE
        gpuSolver::gpuSolver
        igl::core
)
target_compile_definitions(benchmarkSpmv
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(benchmarkSpmv PROPERTIES
    FOLDER examples
)


add_executable(graphColoringUnittest
  graphColoringTest.cc
)

target_link_libraries(graphColoringUnittest
    PRIVATE
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(graphColoringUnittest PROPERTIES
    FOLDER tests
)


add_executable(gaussSeidelUnittest
  gaussSeidelSymmUnittest.cc
)

target_link_libraries(gaussSeidelUnittest
    PRIVATE
        gpuSolver::gpuSolver
        GTest::gtest_main
)

set_target_properties(gaussSeidelUnittest PROPERTIES
    FOLDER tests
)


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

add_executable(sanityCheckCG
  mainCGSanityCheck.cc
)
target_link_libraries(sanityCheckCG
  PRIVATE 
  gpuSolver::gpuSolver
  GTest::gtest_main
)

target_compile_definitions(sanityCheckCG
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")


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



include(polyscope)
add_executable(laplacianAMG
  mainMGCGCheck.cc
)
target_link_libraries(laplacianAMG
  PRIVATE 
  gpuSolver::gpuSolver
  igl::core
  polyscope::polyscope
)

target_compile_definitions(laplacianAMG
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
