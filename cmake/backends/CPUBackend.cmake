# ------------------------------------------------------------
# CPU (OpenMP) backend
# ------------------------------------------------------------

find_package(OpenMP)

add_library(
    gpuSolverCPU

    src/CPU/CPUBackend.cc
    src/CPU/CPUDeviceVector.cc
    src/CPU/CPUDeviceIndexVector.cc
    src/CPU/CPUDeviceScalar.cc
    src/CPU/CPUDeviceMatrix.cc
    src/CPU/CPUDeviceELLMatrix.cc
)

add_library(
    gpuSolver::CPU
    ALIAS gpuSolverCPU
)

target_include_directories(
    gpuSolverCPU
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    gpuSolverCPU
    PUBLIC
        gpuSolverCore
)

# OpenMP is optional on purpose: without it, every #pragma omp is
# silently ignored by the compiler and this becomes a correct,
# single-threaded backend -- useful as a plain correctness oracle
# even on a machine where OpenMP isn't set up, just not what you want
# for the actual multithreading benchmarks.
if(OpenMP_CXX_FOUND)
    target_link_libraries(
        gpuSolverCPU
        PUBLIC
            OpenMP::OpenMP_CXX
    )
else()
    message(
        WARNING
        "OpenMP not found -- gpuSolverCPU will build but run single-threaded."
    )
endif()

target_compile_definitions(
    gpuSolverCPU
    PUBLIC
        GPUSOLVER_USE_CPU
)

set_target_properties(
    gpuSolverCPU
    PROPERTIES
        FOLDER backend
)

# set(
#     GPUSOLVER_BACKEND_TARGET
#     gpuSolverCPU
# )


