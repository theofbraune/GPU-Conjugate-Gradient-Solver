
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
