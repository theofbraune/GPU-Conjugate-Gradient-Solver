if(TARGET metal::metal)
    return()
endif()

if(NOT APPLE)
    return()
endif()

message(STATUS "Third-party (external): creating target 'metal::metal'")

include(CPM)

set(GPUSOLVER_METAL_CPP_VERSION 26)

CPMAddPackage(
    NAME metal_cpp
    URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
    DOWNLOAD_ONLY YES
)

add_library(metal INTERFACE)
add_library(metal::metal ALIAS metal)

target_include_directories(metal
    SYSTEM INTERFACE
        ${metal_cpp_SOURCE_DIR}
)

target_compile_features(metal
    INTERFACE
        cxx_std_17
)

target_link_libraries(metal
    INTERFACE
        "-framework Foundation"
        "-framework Metal"
        "-framework CoreGraphics"
)

set_target_properties(metal PROPERTIES
    FOLDER third_party
)
