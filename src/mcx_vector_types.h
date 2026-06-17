/**
 * @file mcx_vector_types.h
 * @brief Vector type definitions for MCX
 *
 * @author Jeff Daily <jeff.daily@amd.com>
 * @copyright Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Provides float3, float4, uint3, uint4, int3, int4 types that are compatible
 * with both CUDA vector_types.h and HIP hip_vector_types.h.
 *
 * For C code compiled with gcc/clang (not hipcc), we define the types ourselves.
 * For CUDA (.cu) and HIP (.cu compiled as HIP), we use the native headers.
 */

#ifndef _MCX_VECTOR_TYPES_H
#define _MCX_VECTOR_TYPES_H

#if defined(__HIPCC__) || defined(__HIP__)
/* Compiled with hipcc -- include HIP runtime (provides __align__ and vector types) */
#include <hip/hip_runtime.h>

#elif defined(__CUDACC__)
/* Compiled with nvcc -- use CUDA vector types */
#include <vector_types.h>

#elif defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
/* Plain C/C++ code for HIP build -- define types manually */

/* Portable alignment macro for struct definitions */
#if defined(__GNUC__) || defined(__clang__)
#define __align__(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define __align__(n) __declspec(align(n))
#else
#define __align__(n)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Align float4/uint4/int4 to 16 bytes to match HIP vector types */
typedef struct { float x, y, z; } float3;
typedef struct __align__(16) { float x, y, z, w; } float4;
typedef struct { unsigned int x, y, z; } uint3;
typedef struct __align__(16) { unsigned int x, y, z, w; } uint4;
typedef struct { int x, y, z; } int3;
typedef struct __align__(16) { int x, y, z, w; } int4;
typedef struct { int x, y; } int2;
typedef struct { unsigned int x, y; } uint2;
typedef struct { float x, y; } float2;
typedef struct { float x; } float1;

#ifdef __cplusplus
}
#endif

#else
/* CUDA build -- use CUDA vector types */
#include <vector_types.h>

#endif

#endif /* _MCX_VECTOR_TYPES_H */
