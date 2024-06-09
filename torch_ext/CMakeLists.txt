cmake_minimum_required(VERSION 3.21)

project(blocksparse_attn LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setup PyTorch (which also sets up HIP)
set(ENV{PYTORCH_ROCM_ARCH} ${GPU_TARGETS})
find_package(Torch REQUIRED)
# https://github.com/pytorch/pytorch/issues/108041#issuecomment-1753895665
get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Setup Composable Kernel
include(${PROJECT_SOURCE_DIR}/cmake/FetchComposableKernel.cmake)

# Setup PythonExtensions
find_package(PythonExtensions REQUIRED)

# Declare and configure my library
set(TORCH_EXT_SOURCES
    ${PROJECT_SOURCE_DIR}/bsp_attn/bsp_attn.cpp
)
set_source_files_properties(${TORCH_EXT_SOURCES} PROPERTIES LANGUAGE HIP)
hip_add_library(bsp_attn_ext MODULE ${TORCH_EXT_SOURCES})
set_property(TARGET bsp_attn_ext PROPERTY HIP_ARCHITECTURES ${GPU_TARGETS})
python_extension_module(bsp_attn_ext)
target_compile_definitions(bsp_attn_ext INTERFACE _GLIBCXX_USE_CXX11_ABI=1)  # https://github.com/facebookresearch/fairseq/issues/4246

target_include_directories(bsp_attn_ext
    PUBLIC ${COMPOSABLE_KERNEL_INCLUDE_DIRS}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)

message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message("TORCH_INTERFACE_LIB: ${TORCH_INTERFACE_LIB}")
message("TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

target_link_libraries(bsp_attn_ext
    ${TORCH_LIBRARIES}
    # ${TORCH_INTERFACE_LIB}
    ${TORCH_PYTHON_LIBRARY}
    # composable_kernel
)

install(TARGETS bsp_attn_ext LIBRARY DESTINATION bsp_attn)
