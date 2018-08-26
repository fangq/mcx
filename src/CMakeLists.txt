#################################################################
#  CMake configure file for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2018/08/26
#################################################################

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# NVCC Options
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -lineinfo -Xcompiler -Wall -Xcompiler -fopenmp -O3 -arch=sm_30
    )

# C Options
set(CMAKE_C_FLAGS "-g -Wall -std=c99 -fopenmp -m64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../bin)

# Add include directories
include_directories(cjson)

# Add all project units
cuda_add_executable(
    mcx
    mcextreme.c
    mcx_core.cu
    mcx_core.h
    mcx_utils.c
    mcx_utils.h
    mcx_shapes.c
    mcx_shapes.h
    tictoc.c
    tictoc.h
    cjson/cJSON.c
    cjson/cJSON.h
    )

# Link options
target_link_libraries(
    mcx
    -fopenmp
    )