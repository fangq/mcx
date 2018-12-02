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
\file    mcx_core.h

@brief   MCX GPU kernel header file
*******************************************************************************/

#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif


#define ABS(a)  ((a)<0?-(a):(a))
#define DETINC	32
#define MCX_DEBUG_RNG       1                   /**< MCX debug flags */
#define MCX_DEBUG_MOVE      2
#define MCX_DEBUG_PROGRESS  4
#define MAX_ACCUM           1000.f

#define ROULETTE_SIZE       10.f                /**< Russian Roulette size */

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
        float pscat; /**< remaining unit-less scattering length = length * scattering coeff */
        float t;     /**< photon elapse time, unit=s*/
	float pathlen; /**< photon total pathlength inside a voxel, in grid unit*/
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

/**
 * @brief Simulation constant parameters stored in the constant memory
 *
 * This struct stores all constants used in the simulation.
 */

typedef struct  __align__(16) KernelParams {
  float3 vsize;                      /**< volume voxel size in grid unit, always 1,1,1 */
  float  minstep;                    /**< minimum step of the 3, always 1 */
  float  twin0;                      /**< starting time of the current time gate, unit is s */
  float  twin1;                      /**< end time of the current time gate, unit is s  */
  float  tmax;                       /**< maximum time gate length, same as cfg.tend */
  float  oneoverc0;                  /**< 1/(speed of light in the vacuum)*/
  unsigned int save2pt;              /**< flag if mcx outputs fluence volume */
  unsigned int doreflect;            /**< flag if mcx performs reflection calculations */
  unsigned int dorefint;             /**< flag if mcx perform reflection calculations at internal boundaries */
  unsigned int savedet;              /**< flag if mcx outputs detected photon partial length data */
  float  Rtstep;                     /**< reciprocal of the step size */
  float4 ps;                         /**< initial position vector, for pencil beam */
  float4 c0;                         /**< initial directon vector, for pencil beam */
  float3 maxidx;                     /**< maximum index in x/y/z directions for out-of-bound tests */
  uint4  dimlen;                     /**< maximum index used to convert x/y/z to 1D array index */
  uint3  cp0;                        /**< 3D coordinates of one diagonal of the cached region  (obsolete) */
  uint3  cp1;                        /**< 3D coordinates of the other diagonal of the cached region  (obsolete) */
  uint2  cachebox;                   /**< stride for cachebox data acess  (obsolete) */
  float  minenergy;                  /**< threshold of weight to trigger Russian roulette */
  float  skipradius2;                /**< square of the radius within which the data is cached (obsolete) */
  float  minaccumtime;               /**< time steps for tMCimg like weight accummulation (obsolete) */
  int    srctype;                    /**< type of the source */
  float4 srcparam1;                  /**< source parameters set 1 */
  float4 srcparam2;                  /**< source parameters set 2 */
  int voidtime;                      /**< flag if the time-of-flight in the background is counted */
  unsigned int maxdetphoton;         /**< max number of detected photons */
  unsigned int maxmedia;             /**< max number of media labels */
  unsigned int detnum;               /**< max number of detectors */
  unsigned int maxgate;              /**< max number of time gates */
  unsigned int idx1dorig;            /**< pre-computed 1D index of the photon at launch for pencil/isotropic beams */
  unsigned int mediaidorig;          /**< pre-computed media index of the photon at launch for pencil/isotropic beams */
  unsigned int isatomic;             /**< whether atomic operations are used */
  unsigned int maxvoidstep;          /**< max steps that photon can travel in the background before entering non-zero voxels */
  unsigned int issaveseed;           /**< flag if one need to save the detected photon seeds for replay */
  unsigned int issaveexit;           /**< flag if one need to save the detected photon positions and dir vectors */
  unsigned int issaveref;            /**< flag if one need to save diffuse reflectance data in the 0-voxel layer next to the boundary */
  unsigned int ismomentum;           /**< 1 to save momentum transfer for detected photons, implies issavedet=1*/
  unsigned int isspecular;           /**< 0 do not perform specular reflection at launch, 1 do specular reflection */
  unsigned int seedoffset;           /**< offset of the seed, not used */
  int seed;                          /**< RNG seed passted from the host */
  unsigned int outputtype;           /**< Type of output to be accummulated */
  int threadphoton;                  /**< how many photons to be simulated in a thread */
  int oddphotons;                    /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
  int faststep;                      /**< use an approximated stepping approach, not used */
  unsigned int debuglevel;           /**< debug flags */
  unsigned int maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
  unsigned int gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
  unsigned int is2d;                 /**< is the domain a 2D slice? */
  int replaydet;                     /**< select which detector to replay, 0 for all, -1 save all separately */
  unsigned char bc[8];               /**< boundary conditions */
}MCXParam;

void mcx_run_simulation(Config *cfg,GPUInfo *gpu);
int  mcx_list_gpu(Config *cfg, GPUInfo **info);

#ifdef  __cplusplus
}
#endif

#endif

