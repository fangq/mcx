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

cmake_minimum_required(VERSION 3.5)

project(mcx)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)
option(BUILD_MEX "Build mex" ON)

# GPU sources that will be compiled with CUDA or HIP
set(MCX_GPU_SOURCES mcx_core.cu)

# Common sources (C/C++)
set(MCX_COMMON_SOURCES
    mcx_utils.c
    mcx_shapes.c
    mcx_bench.c
    mcx_lang.c
    mcx_mie.cpp
    mcx_tictoc.c
    mcx_neurojson.cpp
    cjson/cJSON.c
    ubj/ubjw.c
)

if(USE_HIP)
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
        set(CMAKE_HIP_ARCHITECTURES "gfx90a")
    endif()
    find_package(hip REQUIRED)

    # Mark .cu files as HIP language
    set_source_files_properties(${MCX_GPU_SOURCES} PROPERTIES LANGUAGE HIP)

    # HIP compile flags for .cu files
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DUSE_HIP -DUSE_ATOMIC -DSAVE_DETECTORS")

    # Add USE_HIP define to C and C++ files as well (for header guards)
    add_compile_definitions(USE_HIP)
else()
    find_package(CUDA QUIET REQUIRED)
endif()

find_package(OpenMP REQUIRED)

add_subdirectory(zmat)

if(BUILD_PYTHON)
    add_subdirectory(pybind11)
    find_package(Python3 COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

if(BUILD_MEX)
    find_package(Matlab)
endif()

# C Options
if(WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -std=c99 -DWIN32")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -std=c99 -fPIC")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib)

# Add include directories
include_directories(cjson ubj zmat zmat/easylzma zmat/easylzma/lzma)

# Add link directories
link_directories(zmat)

if(USE_HIP)
    # Create GPU object library (HIP sources only)
    add_library(mcx_gpu OBJECT ${MCX_GPU_SOURCES})
    target_link_libraries(mcx_gpu hip::device)
    set_target_properties(mcx_gpu PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # Create mcx library (combine GPU objects with C/C++ sources)
    add_library(mcx STATIC ${MCX_COMMON_SOURCES} $<TARGET_OBJECTS:mcx_gpu>)
    target_link_libraries(mcx hip::host)

    # Create mcx executable
    add_executable(mcx-exe mcx.c)
    set_target_properties(mcx-exe PROPERTIES OUTPUT_NAME mcx)
    # On Windows, MSVC-style linkers generate a .lib import stub for the exe
    # with the same name as the static mcx library; redirect the exe's implib
    # to a different path to avoid overwriting the static lib at link time.
    if(WIN32)
        set_target_properties(mcx-exe PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/implibs")
    endif()
    target_link_libraries(mcx-exe mcx OpenMP::OpenMP_CXX zmat)

    if(BUILD_PYTHON)
        add_library(_pmcx MODULE ${MCX_COMMON_SOURCES} $<TARGET_OBJECTS:mcx_gpu> pmcx.cpp)
        target_compile_definitions(_pmcx PUBLIC MCX_CONTAINER PYBIND11_VERSION_MAJOR)
        target_link_libraries(_pmcx hip::host OpenMP::OpenMP_CXX pybind11::module pybind11::lto pybind11::windows_extras zmat)
        pybind11_extension(_pmcx)
        pybind11_strip(_pmcx)
        set_target_properties(_pmcx PROPERTIES CXX_VISIBILITY_PRESET "hidden")
    endif()

    if(BUILD_MEX AND Matlab_FOUND)
        add_library(mcx-matlab STATIC ${MCX_COMMON_SOURCES} $<TARGET_OBJECTS:mcx_gpu>)
        target_compile_definitions(mcx-matlab PUBLIC MCX_CONTAINER MATLAB_MEX_FILE)
        target_link_libraries(mcx-matlab hip::host)

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
        set_target_properties(mcxlab PROPERTIES OUTPUT_NAME ${CMAKE_SOURCE_DIR}/../mcxlab/mcx)
    endif()
else()
    # Original CUDA build using legacy FindCUDA
    string(REGEX REPLACE "[ \t\r\n]+" " -Xcompiler " OMPFLAG ${OpenMP_CXX_FLAGS})
    string(PREPEND OMPFLAG "-Xcompiler ")

    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -g -lineinfo -Xcompiler -Wall -Xcompiler -O3 -arch=sm_50
        -DMCX_TARGET_NAME="Fermi MCX" -DUSE_ATOMIC -use_fast_math
        -DSAVE_DETECTORS -Xcompiler -fPIC ${OMPFLAG}
    )

    # Create mcx library
    cuda_add_library(mcx STATIC
        ${MCX_GPU_SOURCES}
        mcx_core.h
        ${MCX_COMMON_SOURCES}
        mcx_utils.h
        mcx_shapes.h
        mcx_bench.h
        mcx_lang.h
        mcx_mie.h
        mcx_tictoc.h
        mcx_neurojson.h
        cjson/cJSON.h
        ubj/ubj.h
    )

    # Add all project units
    cuda_add_executable(mcx-exe mcx.c)

    set_target_properties(mcx-exe PROPERTIES OUTPUT_NAME mcx)

    # Link options
    target_link_libraries(mcx-exe mcx OpenMP::OpenMP_CXX zmat)

    if(BUILD_PYTHON)
        cuda_add_library(_pmcx MODULE
            ${MCX_GPU_SOURCES}
            mcx_core.h
            ${MCX_COMMON_SOURCES}
            mcx_utils.h
            mcx_shapes.h
            mcx_bench.h
            mcx_lang.h
            mcx_mie.h
            mcx_tictoc.h
            cjson/cJSON.h
            pmcx.cpp
        )
        target_compile_definitions(_pmcx PUBLIC MCX_CONTAINER PYBIND11_VERSION_MAJOR)
        target_link_libraries(_pmcx OpenMP::OpenMP_CXX pybind11::module pybind11::lto pybind11::windows_extras zmat)
        pybind11_extension(_pmcx)
        pybind11_strip(_pmcx)
        set_target_properties(_pmcx PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
    endif()

    # Build mex file
    if(BUILD_MEX AND Matlab_FOUND)
        cuda_add_library(mcx-matlab STATIC
            ${MCX_GPU_SOURCES}
            mcx_core.h
            ${MCX_COMMON_SOURCES}
            mcx_utils.h
            mcx_shapes.h
            mcx_bench.h
            mcx_lang.h
            mcx_mie.h
            mcx_tictoc.h
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
        set_target_properties(mcxlab PROPERTIES OUTPUT_NAME ${CMAKE_SOURCE_DIR}/../mcxlab/mcx)
    endif()
endif()
