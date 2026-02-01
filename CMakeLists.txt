cmake_minimum_required(VERSION 3.15)
project(jax_mlx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")

# --- Dependencies ---

# 1. Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get pybind11 cmake path from pip install
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FIND_RESULT
)
if(NOT PYBIND11_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find pybind11. Install with: pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 REQUIRED)

# 2. Find MLX from pip installation
# The mlx package installs headers in site-packages/mlx/include
# and libmlx.dylib in site-packages/mlx/lib
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import mlx.core; import os; print(os.path.dirname(mlx.core.__file__))"
    OUTPUT_VARIABLE MLX_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MLX_FIND_RESULT
)

if(NOT MLX_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find MLX Python package. Install with: pip install mlx")
endif()

set(MLX_INCLUDE_DIR "${MLX_PYTHON_PATH}/include")
set(MLX_LIB_DIR "${MLX_PYTHON_PATH}/lib")

message(STATUS "Found MLX at: ${MLX_PYTHON_PATH}")
message(STATUS "  Include: ${MLX_INCLUDE_DIR}")
message(STATUS "  Lib: ${MLX_LIB_DIR}")

# Verify paths exist
if(NOT EXISTS "${MLX_INCLUDE_DIR}")
    message(FATAL_ERROR "MLX include directory not found: ${MLX_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${MLX_LIB_DIR}/libmlx.dylib")
    message(FATAL_ERROR "MLX library not found: ${MLX_LIB_DIR}/libmlx.dylib")
endif()

# 3. OpenXLA Headers (for PJRT C API)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# --- The Plugin Library ---
add_library(mlx_pjrt_plugin SHARED  
    src/jax_mlx_pjrt.cpp
)

# Include directories
target_include_directories(mlx_pjrt_plugin PRIVATE
    ${MLX_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Link against MLX and pybind11
target_link_libraries(mlx_pjrt_plugin   
    PRIVATE   
    ${MLX_LIB_DIR}/libmlx.dylib
    pybind11::pybind11
)

# Set RPATH to find libmlx.dylib at runtime
# Plugin installs to site-packages/jax_mlx/
# MLX is at site-packages/mlx/lib/
# So relative path is @loader_path/../mlx/lib
set_target_properties(mlx_pjrt_plugin PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../mlx/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# macOS specific: Allow undefined symbols (Python provides them at runtime)
if (APPLE)  
    target_link_options(mlx_pjrt_plugin PRIVATE "-undefined" "dynamic_lookup")  
endif()

# Install target
install(TARGETS mlx_pjrt_plugin
    LIBRARY DESTINATION jax_mlx
    RUNTIME DESTINATION jax_mlx
)
