set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin/tests"
)



if(APPLE)
  add_subdirectory(metal)
endif()
