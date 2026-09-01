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

if(APPLE)
    include(metal)
endif()

# ------------------------------------------------------------
# gpuSolver library
# ------------------------------------------------------------

add_library(gpuSolver
    src/ConjugateGradient.cc
    src/SparseMatrix.cc
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
            src/DeviceVector.cc
    )

    target_link_libraries(gpuSolver
        PRIVATE
            metal::metal
    )

    target_compile_definitions(gpuSolver
        PRIVATE
            GPUSOLVER_USE_METAL
    )

    # ------------------------------------------------------------
    # here do the compilation of the gpu files
    # ------------------------------------------------------------

    set(GPUSOLVER_METAL_KERNEL_SOURCE
      "${PROJECT_SOURCE_DIR}/kernels/VectorKernels.metal"
    )

    set(GPUSOLVER_METAL_OUTPUT_DIR
      "${CMAKE_BINARY_DIR}/metal"
    )

    set(GPUSOLVER_METAL_AIR
        "${GPUSOLVER_METAL_OUTPUT_DIR}/vector.air"
    )

    set(GPUSOLVER_METAL_LIBRARY
        "${GPUSOLVER_METAL_OUTPUT_DIR}/vector.metallib"
    )

    file(MAKE_DIRECTORY
        "${GPUSOLVER_METAL_OUTPUT_DIR}"
    )


    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_AIR}

        COMMAND
            xcrun
            -sdk macosx
            metal
            -c
            ${GPUSOLVER_METAL_KERNEL_SOURCE}
            -o
            ${GPUSOLVER_METAL_AIR}

        DEPENDS
            ${GPUSOLVER_METAL_KERNEL_SOURCE}

        COMMENT
            "Compiling Metal kernel vector.metal"
    )

    add_custom_command(
        OUTPUT
            ${GPUSOLVER_METAL_LIBRARY}

        COMMAND
            xcrun
            -sdk macosx
            metallib
            ${GPUSOLVER_METAL_AIR}
            -o
            ${GPUSOLVER_METAL_LIBRARY}

        DEPENDS
            ${GPUSOLVER_METAL_AIR}

        COMMENT
            "Linking Metal library vector.metallib"
    )

    add_custom_target(gpuSolverMetalKernels
        DEPENDS
            ${GPUSOLVER_METAL_LIBRARY}
    )

    add_dependencies(
        gpuSolver
        gpuSolverMetalKernels
    )

    target_compile_definitions(gpuSolver
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
