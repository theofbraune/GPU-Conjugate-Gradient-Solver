if(NOT APPLE)

    message(
        FATAL_ERROR
        "Metal backend requested on a non-Apple platform."
    )

endif()

include(metal)

add_library(
    gpuSolverMetal

    src/metal/MetalContext.cc
    src/metal/MetalImplementation.cc
    src/metal/MetalBackend.cc
    src/metal/MetalEncoder.cc
    src/metal/MetalDeviceVector.cc
    src/metal/MetalDeviceSparseMatrix.cc
    src/metal/MetalDeviceIndexVector.cc
    src/metal/MetalDeviceScalar.cc
    src/metal/MetalDeviceELLMatrix.cc
)

add_library(
    gpuSolver::metal
    ALIAS gpuSolverMetal
)

target_include_directories(
    gpuSolverMetal
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    gpuSolverMetal
    PUBLIC
        gpuSolverCore

    PRIVATE
        metal::metal
)

target_compile_definitions(
    gpuSolverMetal
    PUBLIC
        GPUSOLVER_USE_METAL
)

# ------------------------------------------------------------
# Metal kernel sources
# ------------------------------------------------------------

set(GPUSOLVER_METAL_VECTOR_SOURCE
    "${PROJECT_SOURCE_DIR}/kernels/metal/VectorKernels.metal"
)

set(GPUSOLVER_METAL_SPMV_SOURCE
    "${PROJECT_SOURCE_DIR}/kernels/metal/SpMVKernels.metal"
)

set(GPUSOLVER_METAL_REDUCTION_SOURCE
    "${PROJECT_SOURCE_DIR}/kernels/metal/ReductionKernels.metal"
)

set(GPUSOLVER_METAL_SCALAR_SOURCE
    "${PROJECT_SOURCE_DIR}/kernels/metal/ScalarKernels.metal"
)

set(GPUSOLVER_METAL_GAUSS_SEIDEL_SOURCE
  "${PROJECT_SOURCE_DIR}/kernels/metal/GaussSeidelColoringKernels.metal"
)

set(GPUSOLVER_METAL_BLOCK_JACOBI_SOURCE
  "${PROJECT_SOURCE_DIR}/kernels/metal/Invert3x3Diagonal.metal"
)

set(GPUSOLVER_METAL_BLOCK_GAUSS_SEIDEL_SOURCE
  "${PROJECT_SOURCE_DIR}/kernels/metal/GaussSeidelBlock3x3.metal"
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
    gpuSolverMetal
    gpuSolverMetalKernels
)


target_compile_definitions(
    gpuSolverMetal
    PRIVATE
        GPUSOLVER_METAL_LIBRARY="${GPUSOLVER_METAL_LIBRARY}"
)

set(
    GPUSOLVER_BACKEND_TARGET
    gpuSolverMetal
)
