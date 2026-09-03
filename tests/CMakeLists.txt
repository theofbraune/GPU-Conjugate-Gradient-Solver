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



if(APPLE)
  add_subdirectory(metal)
endif()
