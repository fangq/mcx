#################################################################
#  CMake configure file for ZMat
#  Qianqian Fang <q.fang at neu.edu>
#  2020/05/23
#################################################################

cmake_minimum_required(VERSION 3.5)

project(zmat)

find_package(Threads REQUIRED)

option(STATIC_LIB "Build static library" ON)

# C Options
set(CMAKE_C_FLAGS "-g -Wall -O3 -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

# Add include directories
include_directories(lz4)
include_directories(miniz)
include_directories(easylzma)
include_directories(easylzma/lzma)

set(LZMA_SDK_SOURCES
    easylzma/compress.c
    easylzma/decompress.c
    easylzma/lzma_header.c
    easylzma/lzip_header.c
    easylzma/common_internal.c
    easylzma/lzma/LzmaEnc.c
    easylzma/lzma/LzmaDec.c
    easylzma/lzma/LzmaLib.c
    easylzma/lzma/LzFind.c
    easylzma/lzma/Bra.c
    easylzma/lzma/BraIA64.c
    easylzma/lzma/Alloc.c
    easylzma/lzma/7zCrc.c
    easylzma/lzma/7zCrcOpt.c
    easylzma/lzma/CpuArch.c
    easylzma/lzma/Lzma2Dec.c
    easylzma/lzma/Lzma2DecMt.c
    easylzma/lzma/Lzma2Enc.c
    easylzma/lzma/MtCoder.c
    easylzma/lzma/MtDec.c
    easylzma/lzma/Xz.c
    easylzma/lzma/XzCrc64.c
    easylzma/lzma/XzCrc64Opt.c
    easylzma/lzma/XzEnc.c
    easylzma/lzma/XzDec.c
    easylzma/lzma/Sha256.c
    easylzma/lzma/Sha256Opt.c
    easylzma/lzma/Delta.c
    easylzma/lzma/Bra86.c
    easylzma/lzma/7zStream.c
    easylzma/lzma/LzFindMt.c
    easylzma/lzma/LzFindOpt.c
    easylzma/lzma/Threads.c
)

set(ZMAT_SOURCES
    zmatlib.c
    lz4/lz4.c
    lz4/lz4hc.c
    miniz/miniz.c
    miniz/miniz.h
    ${LZMA_SDK_SOURCES}
)

if(STATIC_LIB)
  add_library(zmat STATIC ${ZMAT_SOURCES})
else()
  add_library(zmat SHARED ${ZMAT_SOURCES})
endif()

target_compile_definitions(zmat PRIVATE
    ZMAT_USE_LZMA_SDK
    NO_BLOSC2
    NO_ZLIB
    NO_ZSTD
    _LARGEFILE64_SOURCE=1
)

# Link options
target_link_libraries(zmat Threads::Threads)

if(Matlab_FOUND)
    matlab_add_mex(
      NAME zipmat
      SRC zmat.cpp
      LINK_TO mex mx zmat
    )
endif()
