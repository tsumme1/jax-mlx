cmake_minimum_required(VERSION 3.16)
project(mlx_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX installation path dynamically via Python
execute_process(
    COMMAND python -c "import mlx.core as mx; import os; print(os.path.dirname(mx.__file__))"
    OUTPUT_VARIABLE MLX_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MLX_FOUND
)

if(NOT MLX_FOUND EQUAL 0)
    message(FATAL_ERROR "MLX not found. Please install with: pip install mlx")
endif()

message(STATUS "Found MLX at: ${MLX_ROOT}")

set(MLX_INCLUDE_DIR "${MLX_ROOT}/include")
set(MLX_LIB_DIR "${MLX_ROOT}/lib")

# Add include directory
include_directories(${MLX_INCLUDE_DIR})

# Add library directory
link_directories(${MLX_LIB_DIR})

add_executable(mlx_cpp_benchmark mlx_cpp_benchmark.cpp)
target_link_libraries(mlx_cpp_benchmark PRIVATE mlx)

# Set rpath so the executable can find libmlx.dylib
set_target_properties(mlx_cpp_benchmark PROPERTIES
    BUILD_RPATH "${MLX_LIB_DIR}"
    INSTALL_RPATH "${MLX_LIB_DIR}"
)

# Optimize for release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
