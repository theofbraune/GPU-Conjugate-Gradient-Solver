
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
