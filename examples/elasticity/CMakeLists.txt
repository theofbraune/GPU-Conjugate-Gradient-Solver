include(libigl)
include(polyscope)

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
