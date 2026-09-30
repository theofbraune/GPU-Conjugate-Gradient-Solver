# ------------------------------------------------------------
# AMG hierarchy support (amgcl-based).
#
# Deliberately its own library: the core gpuSolver library never
# depends on amgcl. Anything that wants to *build* a hierarchy links
# against this; the result is handed to gpuSolver as plain Eigen
# sparse matrices.
# ------------------------------------------------------------

include(amgcl)   

add_library(gpuSolverAMG STATIC
    src/AMGHierarchyBuilder.cc
)

add_library(gpuSolverAMG::gpuSolverAMG ALIAS gpuSolverAMG)

target_include_directories(gpuSolverAMG
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gpuSolverAMG
    PUBLIC
        gpuSolver::gpuSolver
        Eigen3::Eigen     # public header exposes Eigen types, so this must propagate
    PRIVATE
        amgcl::amgcl      # amgcl only ever appears in the .cc
)

target_compile_features(gpuSolverAMG PUBLIC cxx_std_17)

add_subdirectory(tests)
