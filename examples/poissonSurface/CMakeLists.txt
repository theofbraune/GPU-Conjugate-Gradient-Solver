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

target_compile_definitions(poissonSurface
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(poissonSurface PROPERTIES
    FOLDER examples
)
