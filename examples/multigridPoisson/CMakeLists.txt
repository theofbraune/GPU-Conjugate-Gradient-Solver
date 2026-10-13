
include(libigl)
include(polyscope)

add_executable(multigridPoissonExample
  mainAMGtest.cc
)

target_link_libraries(multigridPoissonExample
    PRIVATE
        gpuSolver::gpuSolver
        gpuSolverAMG::gpuSolverAMG
        igl::core
        polyscope::polyscope
)

add_executable(multigridPoissonExampleVolumetric
  mainTetmeshPoisson.cc
)



target_link_libraries(multigridPoissonExampleVolumetric
    PRIVATE
        gpuSolver::gpuSolver
        gpuSolverAMG::gpuSolverAMG
        igl::core
        polyscope::polyscope
)

#-----------------------------------
#-----------------------------------
#-----------------------------------

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")
  target_link_libraries(
    multigridPoissonExample
    PRIVATE 
    gpuSolverMetal
  )
  target_compile_definitions(
        multigridPoissonExample
        PRIVATE
            GPUSOLVER_USE_METAL
  )

  target_link_libraries(
    multigridPoissonExampleVolumetric
    PRIVATE 
    gpuSolverMetal
  )
  target_compile_definitions(
        multigridPoissonExampleVolumetric
        PRIVATE
            GPUSOLVER_USE_METAL
  )
elseif(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")

  target_link_libraries(
    multigridPoissonExample
    PRIVATE 
    gpuSolverCuda
  )
  target_compile_definitions(
        multigridPoissonExample
        PRIVATE
            GPUSOLVER_USE_CUDA
  )
  target_link_libraries(
    multigridPoissonExampleVolumetric
    PRIVATE 
    gpuSolverCuda
  )
  target_compile_definitions(
        multigridPoissonExampleVolumetric
        PRIVATE
            GPUSOLVER_USE_CUDA
  )
endif()
