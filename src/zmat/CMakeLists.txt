#################################################################
#  CMake configure file for ZMat
#  Qianqian Fang <q.fang at neu.edu>
#  2020/05/23
#################################################################

cmake_minimum_required(VERSION 3.3)

project(zmat)

find_package(ZLIB REQUIRED)
#find_package(Matlab)

option(STATIC_LIB "Build static library" ON)

# C Options
set(CMAKE_C_FLAGS "-g -Wall -O3 -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

# Add include directories
#include_directories(./)
include_directories(lz4)
include_directories(easylzma)
include_directories(easylzma/pavlov)

# Add all project units

if(STATIC_LIB)
  add_library(zmat STATIC
    zmatlib.c
    lz4/lz4.c
    lz4/lz4hc.c
    easylzma/compress.c
    easylzma/decompress.c
    easylzma/lzma_header.c
    easylzma/lzip_header.c
    easylzma/common_internal.c
    easylzma/pavlov/LzmaEnc.c
    easylzma/pavlov/LzmaDec.c
    easylzma/pavlov/LzmaLib.c
    easylzma/pavlov/LzFind.c
    easylzma/pavlov/Bra.c
    easylzma/pavlov/BraIA64.c
    easylzma/pavlov/Alloc.c
    easylzma/pavlov/7zCrc.c
    )
else()
# Add all project units
  add_library(zmat SHARED
    zmatlib.c
    lz4/lz4.c
    lz4/lz4hc.c
    easylzma/compress.c
    easylzma/decompress.c
    easylzma/lzma_header.c
    easylzma/lzip_header.c
    easylzma/common_internal.c
    easylzma/pavlov/LzmaEnc.c
    easylzma/pavlov/LzmaDec.c
    easylzma/pavlov/LzmaLib.c
    easylzma/pavlov/LzFind.c
    easylzma/pavlov/Bra.c
    easylzma/pavlov/BraIA64.c
    easylzma/pavlov/Alloc.c
    easylzma/pavlov/7zCrc.c
    )
endif()

# Link options
target_link_libraries(
    zmat
    ZLIB::ZLIB
    )

if(Matlab_FOUND)
    matlab_add_mex(
      NAME zipmat
      SRC zmat.cpp
      LINK_TO mex mx zmat
    )
endif()

