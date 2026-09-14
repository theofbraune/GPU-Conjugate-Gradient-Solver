if(TARGET amlgl::amgcl)
return()
endif()


message(STATUS "Third-party (external): creating target 'amlgl::amgcl'")

include(CPM)
CPMAddPackage(
NAME amgcl
GITHUB_REPOSITORY ddemidov/amgcl
GIT_TAG master
)

add_library(amlgl::amgcl ALIAS amgcl)
set_target_properties(amgcl PROPERTIES FOLDER third_party)
