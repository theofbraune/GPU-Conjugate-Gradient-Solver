# Portable OpenMP setup with graceful fallback.
# Usage: include(openmp)   -- resolved via CMAKE_MODULE_PATH, same as
#                              include(eigen) / include(glog).
# Provides target: OpenMP::OpenMP_CXX (always -- real impl if found,
# harmless no-op INTERFACE IMPORTED target otherwise, so callers never
# need to branch on whether OpenMP was actually found).

if(TARGET OpenMP::OpenMP_CXX)
    return()
endif()

option(ENABLE_OPENMP "Enable OpenMP if available" ON)

# If user disabled, provide a harmless no-op target and exit.
if(NOT ENABLE_OPENMP)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  message(STATUS "OpenMP disabled by user (ENABLE_OPENMP=OFF); building single-threaded.")
  return()
endif()

# -------------------------
# 1) Try the standard way.
# -------------------------
find_package(OpenMP QUIET)

if(TARGET OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled via standard find_package(OpenMP).")
  return()
endif()

# ---------------------------------------------------------
# 2) AppleClang on macOS: hint Homebrew's libomp if present
# ---------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
  set(_LIBOMP_CANDIDATES
      /opt/homebrew/opt/libomp      # Apple Silicon Homebrew
      /usr/local/opt/libomp)        # Intel Homebrew

  set(_LIBOMP_DIR "")
  foreach(dir IN LISTS _LIBOMP_CANDIDATES)
    if(EXISTS "${dir}/lib/libomp.dylib")
      set(_LIBOMP_DIR "${dir}")
      break()
    endif()
  endforeach()

  if(_LIBOMP_DIR)
    # Tell CMake's FindOpenMP where/what to link
    set(OpenMP_C_FLAGS       "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS     "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES   "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY   "${_LIBOMP_DIR}/lib/libomp.dylib")

    # Try again now that we've hinted the runtime
    find_package(OpenMP QUIET)
    if(TARGET OpenMP::OpenMP_CXX)
      # Ensure the runtime is found at run time (rpath) and (optionally) headers
      set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_LIBOMP_DIR}/lib")
      if(EXISTS "${_LIBOMP_DIR}/include")
        set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "${_LIBOMP_DIR}/include")
      endif()
      message(STATUS "OpenMP enabled with AppleClang using Homebrew libomp at ${_LIBOMP_DIR}.")
      return()
    endif()
  endif()
endif()

# -------------------------------------------------
# 3) Last resort: create a no-op target (build OK)
# -------------------------------------------------
add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
message(WARNING "OpenMP not found; continuing without it (single-threaded). "
                "Set ENABLE_OPENMP=OFF to silence this warning.")


