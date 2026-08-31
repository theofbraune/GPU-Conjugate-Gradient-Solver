
# examples/poissonTetmesh/CMakeLists.txt

include(libigl)
include(polyscope)
include(glog)

add_executable(poissonTetmesh
    main.cc
)

target_link_libraries(poissonTetmesh
    PRIVATE
        gpuSolver::gpuSolver
        igl::core
        polyscope::polyscope
        glog::glog
)
target_compile_definitions(poissonTetmesh
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(poissonTetmesh PROPERTIES
    FOLDER examples
)
