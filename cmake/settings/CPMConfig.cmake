# Must be set before including CPM.cmake / before CPMAddPackage is called
set(CPM_SOURCE_CACHE
    "${CMAKE_CURRENT_LIST_DIR}/../../external/CPM"
    CACHE PATH "CPM source cache directory"
)
