
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
    src/cuda/CudaEncoder.cc 
    src/cuda/CudaDeviceVector.cc
    src/cuda/CudaDeviceIndexVector.cc
    src/cuda/CudaDeviceCSRMatrix.cc
    src/cuda/CudaDeviceScalar.cc
    kernels/cuda/CudaSmokeTest.cu
    kernels/cuda/VectorKernels.cu
    kernels/cuda/GaussSeidelKernels.cu
    kernels/cuda/ScalarKernels.cu
    kernels/cuda/BlockKernels.cu
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
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
)

set_target_properties(
    gpuSolverCuda
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
        FOLDER backend
)

set(
    GPUSOLVER_GPU_BACKEND_TARGET
    gpuSolverCuda
)
