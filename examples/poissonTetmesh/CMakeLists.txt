
# examples/poissonTetmesh/CMakeLists.txt

include(libigl)
include(polyscope)
include(glog)

add_executable(poissonTetmesh
    main.cc
)

target_link_libraries(poissonTetmesh
    PRIVATE
        igl::core
        polyscope::polyscope
        glog::glog
)

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")
  target_link_libraries(
    poissonTetmesh
    PRIVATE 
    gpuSolverMetal
  )
endif()

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    poissonTetmesh
    PRIVATE 
    gpuSolverCuda
  )
endif()

target_compile_definitions(poissonTetmesh
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(poissonTetmesh PROPERTIES
    FOLDER examples
)
