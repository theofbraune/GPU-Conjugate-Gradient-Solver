# include(googletest)

# add_executable(gpuSolverTests
#     test_vector.cpp
#     test_cg.cpp
#     test_poisson.cpp
# )
#
# target_link_libraries(gpuSolverTests
#     PRIVATE
#         gpuSolver::gpuSolver
#         GTest::gtest_main
# )
#
# set_target_properties(gpuSolverTests PROPERTIES
#     FOLDER tests
# )
#
# include(GoogleTest)
#
# gtest_discover_tests(gpuSolverTests)

# add_executable(
#   metal/testsMetalDevice.cc
# )

if(APPLE)
  add_subdirectory(metal)
endif()
