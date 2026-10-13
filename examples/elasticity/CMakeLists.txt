include(libigl)
include(polyscope)
include(glog)

add_executable(cantileverBeamExample
  cantileverBeam.cc
)

target_link_libraries(cantileverBeamExample
    PRIVATE
        gpuSolver::gpuSolver
        gpuSolverAMG::gpuSolverAMG
        igl::core
        polyscope::polyscope
)

add_executable(elasticityDebugger
  debugElasticitySolver.cc
)

target_link_libraries(elasticityDebugger
    PRIVATE
        gpuSolver::gpuSolver
        gpuSolverAMG::gpuSolverAMG
        igl::core
        polyscope::polyscope
        glog::glog
)

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")
  target_link_libraries(
    cantileverBeamExample
    PRIVATE 
    gpuSolverMetal
  )
  target_compile_definitions(
        cantileverBeamExample
        PRIVATE
            GPUSOLVER_USE_METAL
  )

  target_link_libraries(
    elasticityDebugger
    PRIVATE 
    gpuSolverMetal
  )
  target_compile_definitions(
        elasticityDebugger
        PRIVATE
            GPUSOLVER_USE_METAL
  )
elseif(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")

  target_link_libraries(
    cantileverBeamExample
    PRIVATE 
    gpuSolverCuda
  )
  target_compile_definitions(
        cantileverBeamExample
        PRIVATE
            GPUSOLVER_USE_CUDA
  )
  target_link_libraries(
    elasticityDebugger
    PRIVATE 
    gpuSolverCuda
  )
  target_compile_definitions(
        elasticityDebugger
        PRIVATE
            GPUSOLVER_USE_CUDA
  )
endif()
