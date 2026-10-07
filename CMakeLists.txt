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
)

include(CPMConfig)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(GPUSOLVER_BUILD_EXAMPLES "Build gpuSolver examples" ON)
option(GPUSOLVER_BUILD_TESTS    "Build gpuSolver tests" ON)

# ------------------------------------------------------------
# Dependencies required by the actual solver
# ------------------------------------------------------------

include(eigen)
include(glog)

if(APPLE)
    include(metal)
endif()

# ------------------------------------------------------------
# gpuSolver library
# ------------------------------------------------------------

add_library(gpuSolver
  src/CGSolver.cc
)

add_library(gpuSolver::gpuSolver ALIAS gpuSolver)

target_include_directories(gpuSolver
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gpuSolver
    PUBLIC
        Eigen3::Eigen
)

target_compile_features(gpuSolver
    PUBLIC
        cxx_std_17
)

if(APPLE)
    target_sources(gpuSolver
        PRIVATE
            src/metal/MetalContext.cc
            src/metal/MetalImplementation.cc
            src/metal/MetalBackend.cc
            src/metal/MetalEncoder.cc
            src/DeviceVector.cc
            src/DeviceIndexVector.cc
            src/DeviceSparseMatrix.cc
            src/DeviceScalar.cc
            src/HostSparseMatrix.cc 
            src/Permutation.cc 
            src/ReorderingStrategies/IdentityReordering.cc
            src/ReorderingStrategies/RCMReordering.cc
            src/ReorderingStrategies/HierarchicalReordering.cc
            src/AMGHierarchy.cc
            src/DeviceELLMatrix.cc
            src/HostELLMatrix.cc
            src/GraphColoring.cc
            src/BlockUtils.cc
            src/Preconditioners/IdentityPreconditioner.cc
            src/Preconditioners/MetalJacobiPreconditioner.cc
            src/Preconditioners/DampedJacobiPreconditioner.cc
            src/Preconditioners/MultigridDampedJacobiPreconditioner.cc
            src/Preconditioners/SymmetricGaussSeidelPreconditioner.cc
            src/Preconditioners/MultigridGaussSeidelPreconditioner.cc
            src/Preconditioners/BlockJacobiPreconditioner.cc 
            src/Preconditioners/BlockGaussSeidelPreconditioner.cc
            src/Preconditioners/MultigridBlockGaussSeidelPreconditioner.cc
            src/Preconditioners/MultigridBlockJacobiPreconditioner.cc
            src/Preconditioners/MultigridHybridBlockPreconditioner.cc
            src/SparseMatrixUtils.cc
            src/Smoothers/DampedJacobiSmoother.cc
            src/Smoothers/BlockJacobiSmoother.cc
            src/Smoothers/BlockGaussSeidelSmoother.cc
            src/Smoothers/SymmetricGaussSeidelSmoother.cc
            src/Smoothers/MultigridSmoother.cc
    )

    target_link_libraries(gpuSolver
        PRIVATE
            metal::metal
            glog::glog
    )

    target_compile_definitions(gpuSolver
        PRIVATE
            GPUSOLVER_USE_METAL
    )
    # ------------------------------------------------------------
    # Metal kernel sources
    # ------------------------------------------------------------

    set(GPUSOLVER_METAL_VECTOR_SOURCE
        "${PROJECT_SOURCE_DIR}/kernels/VectorKernels.metal"
    )

    set(GPUSOLVER_METAL_SPMV_SOURCE
        "${PROJECT_SOURCE_DIR}/kernels/SpMVKernels.metal"
    )

    set(GPUSOLVER_METAL_REDUCTION_SOURCE
        "${PROJECT_SOURCE_DIR}/kernels/ReductionKernels.metal"
    )

    set(GPUSOLVER_METAL_SCALAR_SOURCE
        "${PROJECT_SOURCE_DIR}/kernels/ScalarKernels.metal"
    )

    set(GPUSOLVER_METAL_GAUSS_SEIDEL_SOURCE
      "${PROJECT_SOURCE_DIR}/kernels/GaussSeidelColoringKernels.metal"
    )

    set(GPUSOLVER_METAL_BLOCK_JACOBI_SOURCE
      "${PROJECT_SOURCE_DIR}/kernels/Invert3x3Diagonal.metal"
    )

    set(GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_SOURCE
      "${PROJECT_SOURCE_DIR}/kernels/GaussSeidelBlock3x3.metal"
    )

    set(GPUSOLVER_METAL_OUTPUT_DIR
        "${CMAKE_BINARY_DIR}/metal"
    )

    set(GPUSOLVER_METAL_VECTOR_AIR
        "${GPUSOLVER_METAL_OUTPUT_DIR}/vector.air"
    )

    set(GPUSOLVER_METAL_SPMV_AIR
        "${GPUSOLVER_METAL_OUTPUT_DIR}/spmv.air"
    )

    set(GPUSOLVER_METAL_REDUCTION_AIR
        "${GPUSOLVER_METAL_OUTPUT_DIR}/reduction.air"
    )

    set(GPUSOLVER_METAL_SCALAR_AIR
        "${GPUSOLVER_METAL_OUTPUT_DIR}/scalar.air"
    )
    set(GPUSOLVER_METAL_GAUSS_SEIDEL_AIR
      "${GPUSOLVER_METAL_OUTPUT_DIR}/gaussSeidel.air"
    )

    set(GPUSOLVER_METAL_BLOCK_JACOBI_AIR
      "${GPUSOLVER_METAL_OUTPUT_DIR}/blockJacobi.air"
    )
    set(GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_AIR
      "${GPUSOLVER_METAL_OUTPUT_DIR}/blockGaussSeidel.air"
    )

    set(GPUSOLVER_METAL_LIBRARY
        "${GPUSOLVER_METAL_OUTPUT_DIR}/gpuSolver.metallib"
    )

    file(MAKE_DIRECTORY
        "${GPUSOLVER_METAL_OUTPUT_DIR}"
    )


    # ------------------------------------------------------------
    # Compile VectorKernels.metal -> vector.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_VECTOR_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_VECTOR_SOURCE}
            -o
            ${GPUSOLVER_METAL_VECTOR_AIR}

        DEPENDS
            ${GPUSOLVER_METAL_VECTOR_SOURCE}

        COMMENT
            "Compiling Metal vector kernels"
    )


    # ------------------------------------------------------------
    # Compile SpMVKernels.metal -> spmv.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_SPMV_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_SPMV_SOURCE}
            -o
            ${GPUSOLVER_METAL_SPMV_AIR}

        DEPENDS
            ${GPUSOLVER_METAL_SPMV_SOURCE}

        COMMENT
            "Compiling Metal SpMV kernels"
    )


    # ------------------------------------------------------------
    # Compile ReductionKernels.metal -> reduction.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_REDUCTION_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_REDUCTION_SOURCE}
            -o
            ${GPUSOLVER_METAL_REDUCTION_AIR}

        DEPENDS
            ${GPUSOLVER_METAL_REDUCTION_SOURCE}

        COMMENT
            "Compiling Metal reduction kernels"
    )

    # ------------------------------------------------------------
    # Compile ScalarKernels.metal -> scalar.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
        ${GPUSOLVER_METAL_SCALAR_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_SCALAR_SOURCE}
            -o
            ${GPUSOLVER_METAL_SCALAR_AIR}

        DEPENDS
        ${GPUSOLVER_METAL_SCALAR_SOURCE}

        COMMENT
        "Compiling Metal scalar kernels"
    )

    # ------------------------------------------------------------
    # Compile GaussSeidelColoringKernels.metal -> gaussSeidel.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
        ${GPUSOLVER_METAL_GAUSS_SEIDEL_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_GAUSS_SEIDEL_SOURCE}
            -o
            ${GPUSOLVER_METAL_GAUSS_SEIDEL_AIR}

        DEPENDS
        ${GPUSOLVER_METAL_GAUSS_SEIDEL_SOURCE}

        COMMENT
        "Compiling Metal gauss seidel kernels"
    )

    # ------------------------------------------------------------
    # Compile BlockJacobiKernels.metal -> blockJacobi.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
        ${GPUSOLVER_METAL_BLOCK_JACOBI_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_BLOCK_JACOBI_SOURCE}
            -o
            ${GPUSOLVER_METAL_BLOCK_JACOBI_AIR}

        DEPENDS
        ${GPUSOLVER_METAL_BLOCK_JACOBI_SOURCE}

        COMMENT
        "Compiling Metal block jacobi kernels"
    )

    # ------------------------------------------------------------
    # Compile BlockGaussSeidel.metal -> blockGaussSeidel.air
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
        ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_SOURCE}
            -o
            ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_AIR}

        DEPENDS
        ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_SOURCE}

        COMMENT
        "Compiling Metal block gauss seidel kernels"
    )

    # ------------------------------------------------------------
    # Link all AIR files into one metallib
    # ------------------------------------------------------------

    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_LIBRARY}

        COMMAND
            xcrun
            -sdk macosx
            metallib
            ${GPUSOLVER_METAL_VECTOR_AIR}
            ${GPUSOLVER_METAL_SPMV_AIR}
            ${GPUSOLVER_METAL_REDUCTION_AIR}
            ${GPUSOLVER_METAL_SCALAR_AIR}
            ${GPUSOLVER_METAL_GAUSS_SEIDEL_AIR}
            ${GPUSOLVER_METAL_BLOCK_JACOBI_AIR}
            ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_AIR}
            -o
            ${GPUSOLVER_METAL_LIBRARY}

        DEPENDS
            ${GPUSOLVER_METAL_VECTOR_AIR}
            ${GPUSOLVER_METAL_SPMV_AIR}
            ${GPUSOLVER_METAL_REDUCTION_AIR}
            ${GPUSOLVER_METAL_SCALAR_AIR}
            ${GPUSOLVER_METAL_GAUSS_SEIDEL_AIR}
            ${GPUSOLVER_METAL_BLOCK_JACOBI_AIR}
            ${GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_AIR}


        COMMENT
            "Linking Metal library gpuSolver.metallib"
    )


    add_custom_target(gpuSolverMetalKernels
        DEPENDS
            ${GPUSOLVER_METAL_LIBRARY}
    )


    add_dependencies(
        gpuSolver
        gpuSolverMetalKernels
    )


    target_compile_definitions(
        gpuSolver
        PRIVATE
            GPUSOLVER_METAL_LIBRARY="${GPUSOLVER_METAL_LIBRARY}"
    )
endif()

set_target_properties(gpuSolver PROPERTIES
    FOLDER core
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
    include(CTest)
    add_subdirectory(tests)
endif()
