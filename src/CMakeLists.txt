#################################################################
#  CMake configure file for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2018/08/26
#################################################################

cmake_minimum_required(VERSION 3.3)

project(mcx)

find_package(ZLIB REQUIRED)
find_package(CUDA QUIET REQUIRED)

add_subdirectory(zmat)

option(BUILD_MEX "Build mex" ON)

if(BUILD_MEX)
    find_package(Matlab)
endif()

# NVCC Options
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -lineinfo -Xcompiler -Wall -Xcompiler -fopenmp -O3 -arch=sm_30
    -DMCX_TARGET_NAME="Fermi MCX" -DUSE_ATOMIC -use_fast_math
    -DSAVE_DETECTORS -Xcompiler -fPIC
    )

# C Options
set(CMAKE_C_FLAGS "-g -Wall -std=c99 -fopenmp -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib)

# Add include directories
include_directories(cjson ubj zmat zmat/easylzma)

# Add link directories
link_directories(zmat)

# Create mcx library
cuda_add_library(mcx STATIC
    mcx_core.cu
    mcx_core.h
    mcx_utils.c
    mcx_utils.h
    mcx_shapes.c
    mcx_shapes.h
    mcx_bench.c
    mcx_bench.h
    tictoc.c
    tictoc.h
    cjson/cJSON.c
    cjson/cJSON.h
    ubj/ubj.h
    ubj/ubjw.c
    )

# Add all project units
cuda_add_executable(
    mcx-exe
    mcextreme.c
    )

set_target_properties(mcx-exe
        PROPERTIES OUTPUT_NAME mcx)

# Link options
target_link_libraries(
    mcx-exe
    mcx
    -fopenmp
    zmat
    ZLIB::ZLIB
    )

# Build mex file
if(BUILD_MEX AND Matlab_FOUND)
    matlab_add_mex(
      NAME mcxlab
      SRC mcxlab.cpp
      LINK_TO mex mx zmat mcx
    )
set_target_properties(mcxlab
        PROPERTIES OUTPUT_NAME ${CMAKE_SOURCE_DIR}/../mcxlab/mcx)
endif()
