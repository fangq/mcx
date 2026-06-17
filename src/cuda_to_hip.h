/**
 * @file cuda_to_hip.h
 * @brief CUDA-to-HIP compatibility header for MCX
 *
 * @author Jeff Daily <jeff.daily@amd.com>
 * @copyright Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * On AMD/ROCm (USE_HIP defined), this header aliases CUDA runtime API calls
 * to their HIP equivalents. On NVIDIA, it is a transparent passthrough to
 * the CUDA runtime.
 */

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Runtime API
#define cudaError_t                     hipError_t
#define cudaSuccess                     hipSuccess
#define cudaGetLastError                hipGetLastError
#define cudaGetErrorString              hipGetErrorString

// Device management
#define cudaGetDeviceCount              hipGetDeviceCount
#define cudaGetDeviceProperties         hipGetDeviceProperties
#define cudaSetDevice                   hipSetDevice
#define cudaDeviceSynchronize           hipDeviceSynchronize
#define cudaDeviceProp                  hipDeviceProp_t

// Memory management
#define cudaMalloc                      hipMalloc
#define cudaFree                        hipFree
#define cudaMemcpy                      hipMemcpy
#define cudaMemset                      hipMemset
#define cudaMemcpyHostToDevice          hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost          hipMemcpyDeviceToHost
#define cudaMemcpyToSymbol              hipMemcpyToSymbol
#define cudaMemcpyFromSymbol            hipMemcpyFromSymbol

// Events
#define cudaEvent_t                     hipEvent_t
#define cudaEventCreate                 hipEventCreate
#define cudaEventRecord                 hipEventRecord
#define cudaEventQuery                  hipEventQuery
#define cudaEventDestroy                hipEventDestroy

// Device reset
#define cudaDeviceReset                 hipDeviceReset

// Host allocation
#define cudaHostAlloc                   hipHostMalloc
#define cudaFreeHost                    hipHostFree
#define cudaHostAllocMapped             hipHostMallocMapped
#define cudaHostAllocPortable           hipHostMallocPortable
#define cudaHostGetDevicePointer        hipHostGetDevicePointer

// Math intrinsics
// umin/umax are CUDA device math functions; HIP uses min/max from std
__device__ __forceinline__ unsigned int umin(unsigned int a, unsigned int b) { return min(a, b); }
__device__ __forceinline__ unsigned int umax(unsigned int a, unsigned int b) { return max(a, b); }

// exp10f is provided by HIP already

// __int_as_float / __float_as_int are provided by HIP as-is

// FP16: HIP provides __half, __half_raw, __half2float in hip/hip_fp16.h

// CUDA version macros -- set to HIP/ROCm equivalent or safe values
#include <hip/hip_version.h>
#define __CUDACC_VER_MAJOR__            HIP_VERSION_MAJOR
#define __CUDACC_VER_MINOR__            HIP_VERSION_MINOR
#ifndef __CUDA_ARCH_LIST__
#define __CUDA_ARCH_LIST__              0
#endif

#else

#include <cuda_runtime.h>
#include "cuda_fp16.h"

#endif
