cmake_minimum_required(VERSION 2.8)
project(ubj)

add_library(ubj ubjw.c ubjr.c ubjrw.c ubj_internal.h ubj.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(tests)



add_subdirectory(tools)