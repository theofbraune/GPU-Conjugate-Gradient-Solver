
# ------------------------------------------------------------
# CUDA backend
# ------------------------------------------------------------

include(CheckLanguage)

check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)

    message(
        FATAL_ERROR
        "CUDA backend requested, but no CUDA compiler was found."
    )
  else()
    message(" Analyze the cuda build cmake ")
endif()

enable_language(CUDA)

find_package(
    CUDAToolkit REQUIRED
)

add_library(
    gpuSolverCuda

    src/cuda/CudaBackend.cc
    kernels/cuda/CudaSmokeTest.cu
)

target_include_directories(
    gpuSolverCuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    gpuSolverCuda
    PUBLIC
        gpuSolverCore

    PRIVATE
        CUDA::cudart
)

set_target_properties(
    gpuSolverCuda
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
        FOLDER backend
)

