if(TARGET GTest::gtest_main)
    return()
endif()

message(STATUS "Third-party (external): creating target 'GTest::gtest_main'")

option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include(CPM)
CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.16.0
)
