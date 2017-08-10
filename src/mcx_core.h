/*******************************************************************************
**
**  \mainpage Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
**
**  \author Qianqian Fang <q.fang at neu.edu>
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  
**  \section slicense License
**        GNU General Public License v3, see LICENSE.txt for details
**
*******************************************************************************/

/***************************************************************************//**
\file    mcx_core.h

\brief   MCX GPU kernel header file
*******************************************************************************/

#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif


#define ABS(a)  ((a)<0?-(a):(a))
#define DETINC	32
#define MCX_DEBUG_RNG       1
#define MCX_DEBUG_MOVE      2
#define MCX_DEBUG_PROGRESS  4

#define ROULETTE_SIZE       10.f

#ifdef  MCX_DEBUG
#define GPUDEBUG(x)        printf x             /**< enable debugging in CPU mode */
#else
#define GPUDEBUG(x)
#endif

typedef float4 MCXpos; /**< x,y,z: position of the photon, w: weight of the photon*/

typedef struct  __align__(16) MCXDir{
        float x; /**< directional vector of the photon, x-component*/
	float y; /**< directional vector of the photon, y-component*/
	float z; /**< directional vector of the photon, z-component*/
        float nscat; /**< total number of scattering events*/
}MCXdir;

typedef struct  __align__(16) MCXTimer{
        float pscat; /**< remaining scattering probability, unit-less*/
        float t;     /**< photon elapse time, unit=s*/
	float tnext; /**< time for the next accumulation,unit=s*/
	float ndone; /**< number of completed photons*/
}MCXtime;

typedef union  __align__(16) GPosition{
	MCXpos d;
	float4 v;
	float  f[4];
}Gpos;

typedef union  __align__(16) GDirection{
        MCXdir d;
        float4 v;
	float  f[4];
}Gdir;

typedef union  __align__(16) GLength{
        MCXtime d;
        float4 v;
	float  f[4];
}Glen;

typedef union  __align__(16) GProperty{
        Medium d; /*defined in mcx_utils.h*/
        float4 v;
	float  f[4];
}Gprop;

typedef unsigned char uchar;

typedef struct  __align__(16) KernelParams {
  float3 vsize;
  float  minstep;
  float  twin0,twin1,tmax;
  float  oneoverc0;
  unsigned int save2pt,doreflect,dorefint,savedet;
  float  Rtstep;
  float4 ps,c0;
  float3 maxidx;
  uint3  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minenergy;
  float  skipradius2;
  float  minaccumtime;
  int    srctype;
  float4 srcparam1;
  float4 srcparam2;
  int voidtime;
  unsigned int maxdetphoton;
  unsigned int maxmedia;
  unsigned int detnum;
  unsigned int idx1dorig;
  unsigned int mediaidorig;
  unsigned int reseedlimit;
  unsigned int isatomic;
  unsigned int maxvoidstep;
  unsigned int issaveseed;
  unsigned int issaveexit;
  unsigned int issaveref;
  unsigned int seedoffset;
  int seed;
  unsigned int outputtype;
  int threadphoton;
  int oddphotons;
  int faststep;
  unsigned int debuglevel;
  unsigned int maxjumpdebug;
  unsigned int gscatter;
  unsigned int is2d;
}MCXParam;

void mcx_run_simulation(Config *cfg,GPUInfo *gpu);
int  mcx_list_gpu(Config *cfg, GPUInfo **info);

#ifdef  __cplusplus
}
#endif

#endif

