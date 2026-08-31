if(TARGET glog::glog)
    return()
endif()

message(STATUS "Third-party (external): creating target 'glog::glog'")

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(WITH_GFLAGS "Use gflags" OFF)
option(WITH_GTEST "Use Google Test" OFF)
set(BUILD_TESTING OFF)

include(CPM)
CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    GIT_TAG v0.7.1
)

set_target_properties(glog PROPERTIES FOLDER third_party)
