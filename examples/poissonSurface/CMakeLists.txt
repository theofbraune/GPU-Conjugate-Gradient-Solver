# examples/poissonSurface/CMakeLists.txt

include(libigl)
include(polyscope)
include(glog)

add_executable(poissonSurface
  mainSurfaceMesh.cc
)

target_link_libraries(poissonSurface
    PRIVATE
        gpuSolver::gpuSolver
        igl::core
        polyscope::polyscope
        glog::glog
)

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")
  target_link_libraries(
    poissonSurface
    PRIVATE 
    gpuSolverMetal
  )
  target_compile_definitions(
        poissonSurface
        PRIVATE
            GPUSOLVER_USE_METAL
  )

elseif(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    poissonSurface
    PRIVATE 
    gpuSolverCuda
  )
  target_compile_definitions(
        poissonSurface
        PRIVATE
            GPUSOLVER_USE_CUDA
  )
endif()

target_compile_definitions(poissonSurface
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(poissonSurface PROPERTIES
    FOLDER examples
)
