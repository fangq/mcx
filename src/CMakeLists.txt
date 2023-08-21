#################################################################
#  CMake configure file for Monte Carlo eXtreme (MCX)
#  Qianqian Fang <q.fang at neu.edu>
#  2018/08/26
#################################################################
# Note: when multiple gcc versions are installed, cmake may find
# the highest version, but nvcc may throw an error saying it does
# not support this gcc version. To solve this, one should run
#      CC=gcc-X CXX=g++-X cmake ..
# where "X" is the version (such as 9) that both installed on
# your system and is supported by nvcc
#################################################################

cmake_minimum_required(VERSION 3.3)

project(mcx)

find_package(CUDA QUIET REQUIRED)

find_package(OpenMP REQUIRED)

add_subdirectory(zmat)

option(BUILD_MEX "Build mex" ON)

if(BUILD_PYTHON)
    add_subdirectory(pybind11)
    find_package (Python3 COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

if(BUILD_MEX)
    find_package(Matlab)
endif()

string(REGEX REPLACE "[ \t\r\n]+" " -Xcompiler " OMPFLAG ${OpenMP_CXX_FLAGS})
string(PREPEND OMPFLAG "-Xcompiler ")

# NVCC Options
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -lineinfo -Xcompiler -Wall -Xcompiler -O3 -arch=sm_35
    -DMCX_TARGET_NAME="Fermi MCX" -DUSE_ATOMIC -use_fast_math
    -DSAVE_DETECTORS -Xcompiler -fPIC ${OMPFLAG}
    )

# C Options
set(CMAKE_C_FLAGS "-g -Wall -std=c99 -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    mcx_mie.cpp
    mcx_mie.h
    mcx_tictoc.c
    mcx_tictoc.h
    cjson/cJSON.c
    cjson/cJSON.h
    ubj/ubj.h
    ubj/ubjw.c
    )

# Add all project units
cuda_add_executable(
    mcx-exe
    mcx.c
    )

set_target_properties(mcx-exe
        PROPERTIES OUTPUT_NAME mcx)

# Link options
target_link_libraries(
    mcx-exe
    mcx OpenMP::OpenMP_CXX
    zmat
    )

if (BUILD_PYTHON)
    cuda_add_library(_pmcx MODULE
            mcx_core.cu
            mcx_core.h
            mcx_utils.c
            mcx_utils.h
            mcx_shapes.c
            mcx_shapes.h
            mcx_bench.c
            mcx_bench.h
            mcx_mie.cpp
            mcx_mie.h
            mcx_tictoc.c
            mcx_tictoc.h
            cjson/cJSON.c
            cjson/cJSON.h
            pmcx.cpp
            )
    target_compile_definitions(_pmcx PUBLIC MCX_CONTAINER PYBIND11_VERSION_MAJOR)

    target_link_libraries(_pmcx OpenMP::OpenMP_CXX pybind11::module pybind11::lto pybind11::windows_extras)

    pybind11_extension(_pmcx)
    pybind11_strip(_pmcx)

    set_target_properties(_pmcx PROPERTIES CXX_VISIBILITY_PRESET "hidden"
            CUDA_VISIBILITY_PRESET "hidden")
endif()

# Build mex file
if(BUILD_MEX AND Matlab_FOUND)
    # Create mcx-matlab library
    cuda_add_library(mcx-matlab STATIC
            mcx_core.cu
            mcx_core.h
            mcx_utils.c
            mcx_utils.h
            mcx_shapes.c
            mcx_shapes.h
            mcx_bench.c
            mcx_bench.h
            mcx_mie.cpp
            mcx_mie.h
            mcx_tictoc.c
            mcx_tictoc.h
            cjson/cJSON.c
            cjson/cJSON.h
            )

    target_compile_definitions(mcx-matlab PUBLIC MCX_CONTAINER MATLAB_MEX_FILE)

    if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
            matlab_add_mex(
              NAME mcxlab
              SRC mcxlab.cpp
              LINK_TO OpenMP::OpenMP_CXX mcx-matlab
            )
    else()
            matlab_add_mex(
              NAME mcxlab
              SRC mcxlab.cpp
              NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES
              LINK_TO ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY} OpenMP::OpenMP_CXX mcx-matlab
            )
    endif()


    target_compile_definitions(mcxlab PUBLIC MCX_CONTAINER MATLAB_MEX_FILE)

    set_target_properties(mcxlab
            PROPERTIES OUTPUT_NAME ${CMAKE_SOURCE_DIR}/../mcxlab/mcx)
endif()
