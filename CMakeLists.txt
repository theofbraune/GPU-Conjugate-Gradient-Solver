cmake_minimum_required(VERSION 3.20)

project(
    gpuSolver
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------
# CPM
# ------------------------------------------------------------

list(PREPEND CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake/settings
    ${PROJECT_SOURCE_DIR}/cmake/recipes
    ${PROJECT_SOURCE_DIR}/cmake/backends
)

include(CPMConfig)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(
    GPUSOLVER_BUILD_EXAMPLES
    "Build gpuSolver examples"
    ON
)

option(
    GPUSOLVER_BUILD_TESTS
    "Build gpuSolver tests"
    ON
)

set(
    GPUSOLVER_BACKEND
    "AUTO"
    CACHE STRING
    "Backend to use: AUTO, METAL, CUDA, or CPU"
)

set_property(
    CACHE GPUSOLVER_BACKEND
    PROPERTY STRINGS
        AUTO
        METAL
        CUDA
        CPU
)

string(
    TOUPPER
    "${GPUSOLVER_BACKEND}"
    GPUSOLVER_BACKEND
)

# ------------------------------------------------------------
# Dependencies required by core solver
# ------------------------------------------------------------

include(eigen)
include(glog)

# ------------------------------------------------------------
# Core library
# ------------------------------------------------------------

add_library(
    gpuSolverCore

    src/CGSolver.cc
    src/HostSparseMatrix.cc
    src/Permutation.cc

    src/ReorderingStrategies/IdentityReordering.cc
    src/ReorderingStrategies/RCMReordering.cc
    src/ReorderingStrategies/HierarchicalReordering.cc

    src/AMGHierarchy.cc
    src/HostELLMatrix.cc
    src/GraphColoring.cc
    src/BlockUtils.cc
    src/SparseMatrixUtils.cc

    src/Preconditioners/IdentityPreconditioner.cc
    src/Preconditioners/JacobiPreconditioner.cc
    src/Preconditioners/DampedJacobiPreconditioner.cc
    src/Preconditioners/MultigridDampedJacobiPreconditioner.cc
    src/Preconditioners/SymmetricGaussSeidelPreconditioner.cc
    src/Preconditioners/MultigridGaussSeidelPreconditioner.cc
    src/Preconditioners/BlockJacobiPreconditioner.cc
    src/Preconditioners/BlockGaussSeidelPreconditioner.cc
    src/Preconditioners/MultigridBlockGaussSeidelPreconditioner.cc
    src/Preconditioners/MultigridBlockJacobiPreconditioner.cc
    src/Preconditioners/MultigridHybridBlockPreconditioner.cc

    src/Smoothers/DampedJacobiSmoother.cc
    src/Smoothers/BlockJacobiSmoother.cc
    src/Smoothers/BlockGaussSeidelSmoother.cc
    src/Smoothers/SymmetricGaussSeidelSmoother.cc
    src/Smoothers/MultigridSmoother.cc
)

add_library(
    gpuSolver::core
    ALIAS gpuSolverCore
)

target_include_directories(
    gpuSolverCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    gpuSolverCore
    PUBLIC
        Eigen3::Eigen
        glog::glog
)

target_compile_features(
    gpuSolverCore
    PUBLIC
        cxx_std_17
)

set_target_properties(
    gpuSolverCore
    PROPERTIES
        FOLDER core
)

# ------------------------------------------------------------
# Backend selection
# ------------------------------------------------------------

include(CheckLanguage)

if(GPUSOLVER_BACKEND STREQUAL "AUTO")

    if(APPLE)

        set(GPUSOLVER_SELECTED_BACKEND "METAL")

    else()

        check_language(CUDA)

        if(CMAKE_CUDA_COMPILER)
            message(" select cuda backend")

            set(GPUSOLVER_SELECTED_BACKEND "CUDA")

        else()

            set(GPUSOLVER_SELECTED_BACKEND "CPU")

        endif()

    endif()

else()

    set(
        GPUSOLVER_SELECTED_BACKEND
        "${GPUSOLVER_BACKEND}"
    )

endif()

message(
    STATUS
    "gpuSolver backend: ${GPUSOLVER_SELECTED_BACKEND}"
)

# ------------------------------------------------------------
# Backend implementation
# ------------------------------------------------------------

if(GPUSOLVER_SELECTED_BACKEND STREQUAL "METAL")

    include(MetalBackend)

elseif(GPUSOLVER_SELECTED_BACKEND STREQUAL "CUDA")

    include(CudaBackend)

elseif(GPUSOLVER_SELECTED_BACKEND STREQUAL "CPU")

    include(CpuBackend)

else()

    message(
        FATAL_ERROR
        "Unknown GPUSOLVER_BACKEND='${GPUSOLVER_SELECTED_BACKEND}'"
    )

endif()

# ------------------------------------------------------------
# Public gpuSolver target
# ------------------------------------------------------------

add_library(
    gpuSolver
    INTERFACE
)

target_link_libraries(
    gpuSolver
    INTERFACE
        gpuSolverCore
        ${GPUSOLVER_BACKEND_TARGET}
)

add_library(
    gpuSolver::gpuSolver
    ALIAS gpuSolver
)

# ------------------------------------------------------------
# Examples
# ------------------------------------------------------------

if(GPUSOLVER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

if(GPUSOLVER_BUILD_TESTS)

  enable_testing()

  include(CTest)

  add_subdirectory(tests)

endif()
