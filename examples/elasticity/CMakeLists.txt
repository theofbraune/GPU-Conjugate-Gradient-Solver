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
