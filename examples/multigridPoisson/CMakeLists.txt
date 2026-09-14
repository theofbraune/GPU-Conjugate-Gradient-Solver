
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
