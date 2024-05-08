include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: composable_kernel")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    composable_kernel
    GIT_REPOSITORY https://github.com/cameronshinn/composable_kernel.git
    GIT_TAG        44ae8dca0f9a6fb14e12e3cd19b0aa2351b0759b
)

# OVERRIDE_FIND_PACKAGE

FetchContent_GetProperties(composable_kernel)
set(INSTANCES_ONLY ON)
if(NOT composable_kernel_POPULATED)
    FetchContent_MakeAvailable(composable_kernel)
endif()
add_library(composable_kernel STATIC IMPORTED)
set_target_properties(composable_kernel PROPERTIES IMOPRTED_LOCATION ${composable_kernel_BINARY_DIR})
set(COMPOSABLE_KERNEL_INCLUDE_DIRS "${composable_kernel_SOURCE_DIR}/include" "${composable_kernel_SOURCE_DIR}/library/include")

# TODO: Turn off tests and extras if the library release version support it in the CMake file
