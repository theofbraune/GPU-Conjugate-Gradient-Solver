set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin/tests"
)




if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")
    add_subdirectory(metal)
endif()

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")
    add_subdirectory(cuda)
endif()

# if(GPUSOLVER_SELECTED_BACKEND STREQUAL "CPU")
#     add_subdirectory(cpu)
# endif()
