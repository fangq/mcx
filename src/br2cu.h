/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504, 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    br2cu.h

@brief   Short vector name conversion unit
*******************************************************************************/

#ifndef _BR2CU_H
#define _BR2CU_H

/***********************************************************
Translation unit for BrookGPU to CUDA

Qianqian Fang <q.fang at neu.edu>

************************************************************/

#include <cuda.h>

#define kernel __global__
#define streamRead(a,b)  {size_t sy;cudaGetSymbolSize(&sy,b);printf("ss %d\n",sy);cudaMemcpy(a, b[0], sy, \
cudaMemcpyHostToDevice);}
#define streamWrite(a,b) {size_t sy;cudaGetSymbolSize(&sy,a);printf("ss %d\n",sy);cudaMemcpy(b[0], a, sy, \
cudaMemcpyDeviceToHost);}
#define int2(a,b) make_int2(a,b)                   /**< int2 constructor */
#define int3(a,b,c) make_int3(a,b,c)               /**< int3 constructor */
#define uint2(a,b) make_uint2(a,b)                 /**< uint2 constructor */
#define uint3(a,b,c) make_uint3(a,b,c)             /**< uint3 constructor */
#define uint4(a,b,c,d) make_uint4(a,b,c,d)         /**< uint4 constructor */
#define float1(a) make_float1(a)                   /**< float1 constructor */
#define float2(a,b) make_float2(a,b)               /**< float2 constructor */
#define float3(a,b,c) make_float3(a,b,c)           /**< float3 constructor */
#define float4(a,b,c,d) make_float4(a,b,c,d)       /**< float4 constructor */
typedef unsigned int uint;                         /**< use uint for unsigned int */

#endif
