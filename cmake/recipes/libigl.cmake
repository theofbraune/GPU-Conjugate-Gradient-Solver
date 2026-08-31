if(TARGET igl::core)
    return()
endif()

message(STATUS "Third-party (external): creating target 'igl::core'")

include(eigen)
option(LIBIGL_INSTALL "Enable installation of libigl targets" OFF)

include(CPM)
CPMAddPackage(
    NAME libigl
    GITHUB_REPOSITORY libigl/libigl
    GIT_TAG 788871103938ce92112714d11ea491e8e6b4d972
)

set_target_properties(igl_core PROPERTIES FOLDER third_party/libigl)
