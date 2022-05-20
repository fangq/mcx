/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2022
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
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


#define ABS(a)  ((a)<0?-(a):(a))                 /**< macro to take absolute value */

/**
  * To avoid round-off errors when adding very small number to large number, we
  * split the GPU accumulation array into 2 copies, if the first copy exceeds
  * this limit, we add the first copy to the 2nd copy, and clear the first one.
  * See https://github.com/fangq/mcx/issues/41 for details
  */
#define MAX_ACCUM           1000.f

#define ROULETTE_SIZE       10.f                  /**< Russian Roulette size */

#ifdef  MCX_DEBUG
#define GPUDEBUG(x)        printf x             /**< enable debugging in CPU mode */
#else
#define GPUDEBUG(x)                             /**< printing commands are ignored if MCX_DEBUG macro is not defined */
#endif

typedef float4 MCXpos; /**< x,y,z: position of the photon, w: weight of the photon*/

typedef struct __align__(16) SplitVoxel {
    unsigned char issplit; /**< flag if a photon is inside a mixed voxel*/
    unsigned char lower;   /**< label of the tissue type at -nv direction (lower region)*/
    unsigned char upper;   /**< label of the tissue type at +nv direction (upper region)*/
    unsigned char isupper; /**< flag if a photon is in the upper region*/
} SVox;

typedef struct __align__(16) StokesVector {
    float i; /**< total light intensity: IH + IV */
    float q; /**< IH - IV */
    float u; /**< I(+pi/4) - I(-pi/4) */
    float v; /**< IR - IL */
} Stokes;

typedef struct __align__(16) MCXSplit {
    SVox   sv; /**< indicator of tissue type under split-voxel MC (SVMC) mode*/
    float3 rp; /**< reference point of the intra-voxel interface*/
    float3 nv; /**< normal vector of the intra-voxel interafece (direction: lower -> upper)*/
} MCXsp;

typedef struct  __align__(16) MCXDir {
    float x; /**< directional vector of the photon, x-component*/
    float y; /**< directional vector of the photon, y-component*/
    float z; /**< directional vector of the photon, z-component*/
    float nscat; /**< total number of scattering events*/
} MCXdir;

typedef struct  __align__(16) MCXTimer {
    float pscat; /**< remaining unit-less scattering length = length * scattering coeff */
    float t;     /**< photon elapse time, unit=s*/
    float pathlen; /**< photon total pathlength inside a voxel, in grid unit*/
    float ndone; /**< number of completed photons*/
} MCXtime;

typedef union  __align__(16) GPosition {
    MCXpos d;
    float4 v;
    float  f[4];
} Gpos;

typedef union  __align__(16) GDirection {
    MCXdir d;
    float4 v;
    float  f[4];
} Gdir;

typedef union  __align__(16) GLength {
    MCXtime d;
    float4 v;
    float  f[4];
} Glen;

typedef union  __align__(16) GProperty {
    Medium d; /*defined in mcx_utils.h*/
    float4 v;
    float  f[4];
} Gprop;

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
    float4 s0;                         /**< initial stokes parameters, for polarized photon simulation */
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
    unsigned int maxpolmedia;          /**< max number of media labels for polarized light */
    unsigned int maxgate;              /**< max number of time gates */
    unsigned int idx1dorig;            /**< pre-computed 1D index of the photon at launch for pencil/isotropic beams */
    unsigned int mediaidorig;          /**< pre-computed media index of the photon at launch for pencil/isotropic beams */
    unsigned int isatomic;             /**< whether atomic operations are used */
    unsigned int maxvoidstep;          /**< max steps that photon can travel in the background before entering non-zero voxels */
    unsigned int issaveseed;           /**< flag if one need to save the detected photon seeds for replay */
    unsigned int issaveref;            /**< flag if one need to save diffuse reflectance data in the 0-voxel layer next to the boundary */
    unsigned int isspecular;           /**< 0 do not perform specular reflection at launch, 1 do specular reflection */
    unsigned int seedoffset;           /**< offset of the seed, not used */
    int seed;                          /**< RNG seed passted from the host */
    unsigned int outputtype;           /**< Type of output to be accummulated */
    int threadphoton;                  /**< how many photons to be simulated in a thread */
    int oddphotons;                    /**< how many threads need to simulate 1 more photon above the basic load (threadphoton) */
    int faststep;                      /**< use an approximated stepping approach, not used */
    unsigned int debuglevel;           /**< debug flags */
    unsigned int savedetflag;          /**< detected photon save flags */
    unsigned int reclen;               /**< length of buffer per detected photon */
    unsigned int partialdata;          /**< per-medium detected photon data length */
    unsigned int w0offset;             /**< photon-sharing buffer offset */
    unsigned int mediaformat;          /**< format of the media buffer */
    unsigned int maxjumpdebug;         /**< max number of positions to be saved to save photon trajectory when -D M is used */
    unsigned int gscatter;             /**< how many scattering events after which mus/g can be approximated by mus' */
    unsigned int is2d;                 /**< is the domain a 2D slice? */
    int replaydet;                     /**< select which detector to replay, 0 for all, -1 save all separately */
    unsigned int srcnum;               /**< total number of source patterns */
    unsigned int nphase;               /**< number of samples for inverse-cdf, will be added by 2 to include -1 and 1 on the two ends */
    float omega;                       /**< modulation angular frequency (2*pi*f), in rad/s, for FD/RF replay */
    unsigned char bc[12];              /**< boundary conditions */
} MCXParam;

void mcx_run_simulation(Config* cfg, GPUInfo* gpu);
int  mcx_list_gpu(Config* cfg, GPUInfo** info);

#ifdef  __cplusplus
}
#endif

#endif

