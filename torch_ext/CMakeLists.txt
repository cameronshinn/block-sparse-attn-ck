# Fetch the composable kernel fork
# Build the sparse and dense mha implementations
# Compile them with pytorch wrappers

cmake_minimum_required(VERSION 3.12)

project(blocksparse_attn LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${PROJECT_SOURCE_DIR}/cmake/LoadHIP.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchComposableKernel.cmake)

hip_add_library(blocksparse_attn INTERFACE)

target_compile_definitions(blocksparse_attn INTERFACE __HIP_PLATFORM_HCC__=1)

target_include_directories(blocksparse_attn
    INTERFACE ${ROCM_INCLUDE_DIRS}
)

####

# Setup PyTorch
set(ENV{PYTORCH_ROCM_ARCH} gfx90a)  # Specify multiple with semicolon seperator: "gfx90a;gfx906"
find_package(Torch REQUIRED)

# Setup PythonExtensions
find_package(PythonExtensions REQUIRED)

set(TORCH_EXT_SOURCES
    ${PROJECT_SOURCE_DIR}/bsp_attn/bsp_attn.cpp
)
set_source_files_properties(${TORCH_EXT_SOURCES} PROPERTIES LANGUAGE HIP)

# set(SPARSE_MHA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# https://github.com/pytorch/pytorch/issues/108041#issuecomment-1753895665
get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_library(bsp_attn_ext MODULE ${TORCH_EXT_SOURCES})
python_extension_module(bsp_attn_ext)
target_link_libraries(bsp_attn_ext
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)
target_include_directories(bsp_attn_ext
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${COMPOSABLE_KERNEL_INCLUDE_DIR}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)
set_property(TARGET bsp_attn_ext PROPERTY CXX_STANDARD 17)

install(TARGETS bsp_attn_ext LIBRARY DESTINATION bsp_attn)
