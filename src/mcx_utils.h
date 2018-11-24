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
\file    mcx_utils.h

@brief   MCX configuration header
*******************************************************************************/
                                                                                         
#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>
#include <vector_types.h>
#include "br2cu.h"
#include "cjson/cJSON.h"
#include "float.h"
#include "nifti1.h"

#define EPS                FLT_EPSILON                   /**< round-off limit */
#define VERY_BIG           (1.f/FLT_EPSILON)             /**< a big number */

#define MAX_PATH_LENGTH     1024                         /**< max characters in a full file name string */
#define MAX_SESSION_LENGTH  256                          /**< max session name length */
#define MAX_DEVICE          256                          /**< max number of GPUs to be used */

#ifndef MCX_CUDA_ARCH
  #define MCX_CUDA_ARCH       100                        /**< fallback CUDA version */
#endif

#define MCX_ERROR(id,msg)   mcx_error(id,msg,__FILE__,__LINE__)  /**< macro for error handling */
#define MIN(a,b)           ((a)<(b)?(a):(b))             /**< macro to get the min values of two numbers */
#define MAX(a,b)           ((a)>(b)?(a):(b))             /**< macro to get the max values of two numbers */

enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWP, otDCS};   /**< types of output */
enum TMCXParent  {mpStandalone, mpMATLAB};                          /**< whether MCX is run in binary or mex mode */
enum TOutputFormat {ofMC2, ofNifti, ofAnalyze, ofUBJSON};           /**< output data format */
enum TBoundary {bcUnknown, bcReflect, bcAbsorb, bcMirror, bcCylic};            /**< boundary conditions */

/**
 * The structure to store optical properties
 * Four relevant optical properties are needed
 */
typedef struct MCXMedium{
	float mua;                     /**< absorption coefficient (in 1/mm) */
	float mus;                     /**< scattering coefficient (in 1/mm) */
	float g;                       /**< anisotropy factor g: -1 to 1 */
	float n;                       /**< refractive index */
} Medium;

/**
 * Filer header data structure to .mch files to store detected photon data 
 * This header has a total of 256 bytes
 */

typedef struct MCXHistoryHeader{
	char magic[4];                 /**< magic bits= 'M','C','X','H' */
	unsigned int  version;         /**< version of the mch file format */
	unsigned int  maxmedia;        /**< number of media in the simulation */
	unsigned int  detnum;          /**< number of detectors in the simulation */
	unsigned int  colcount;        /**< how many output files per detected photon */
	unsigned int  totalphoton;     /**< how many total photon simulated */
	unsigned int  detected;        /**< how many photons are detected (not necessarily all saved) */
	unsigned int  savedphoton;     /**< how many detected photons are saved in this file */
	float unitinmm;                /**< what is the voxel size of the simulation */
	unsigned int  seedbyte;        /**< how many bytes per RNG seed */
        float normalizer;              /**< what is the normalization factor */
	int respin;                    /**< if positive, repeat count so total photon=totalphoton*respin; if negative, total number is processed in respin subset */
	int reserved[4];               /**< reserved fields for future extension */
} History;

/**
 * Data structure for photon replay
 */

typedef struct PhotonReplay{
	int   *detid;                 /**< pointer to the detector index */
	void  *seed;                  /**< pointer to the seeds of the replayed photon */
	float *weight;                /**< pointer to the detected photon weight array */
	float *tof;                   /**< pointer to the detected photon time-of-fly array */
} Replay;

/**
 * Data structure for GPU information
 */

typedef struct MCXGPUInfo {
        char name[MAX_SESSION_LENGTH];/**< name of the GPU */
	int id;                       /**< global index of the GPU, starting from 0 */
	int devcount;                 /**< total GPU count */
	int major;                    /**< major version of the CUDA device */
	int minor;                    /**< minor version of the CUDA device */
	size_t globalmem;             /**< size of the global memory in the GPU */
	size_t constmem;              /**< size of the constant memory in the GPU */
	size_t sharedmem;             /**< size of the shared memory in the GPU */
	int regcount;                 /**< size of the register file in the GPU */
	int clock;                    /**< clock in Hz of the GPU processor */
	int sm;                       /**< number of multi processors */
	int core;                     /**< number of stream processors */
	int autoblock;                /**< optimized number of blocks to launch */
	int autothread;               /**< optimized number of threads to launch */
	int maxgate;                  /**< max number of time gates that can be saved in one call */
	int maxmpthread;              /**< maximum thread number per multi-processor */
} GPUInfo;

/*! \struct MCXConfig
 * Data structure for MCX simulation settings
 */

typedef struct MCXConfig{
	size_t nphoton;               /**<total simulated photon number*/
        unsigned int nblocksize;      /**<thread block size*/
	unsigned int nthread;         /**<num of total threads, multiple of 128*/
	int seed;                     /**<random number generator seed*/

	float4 srcpos;                /**<src position in mm*/
	float4 srcdir;                /**<src normal direction*/
	float tstart;                 /**<start time in second*/
	float tstep;                  /**<time step in second*/
	float tend;                   /**<end time in second*/
	float3 steps;                 /**<voxel sizes along x/y/z in mm*/

	uint3 dim;                    /**<domain size*/
	uint3 crop0;                  /**<sub-volume for cache*/
	uint3 crop1;                  /**<the other end of the caching box*/
	unsigned int medianum;        /**<total types of media*/
	unsigned int detnum;          /**<total detector numbers*/
	unsigned int maxdetphoton;    /**<anticipated maximum detected photons*/
	float detradius;              /**<default detector radius*/
        float sradius;                /**<source region radius, if set to non-zero, accumulation will not perform for dist<sradius*/

	Medium *prop;                 /**<optical property mapping table*/
	float4 *detpos;               /**<detector positions and radius, overwrite detradius*/

	unsigned int maxgate;         /**<simultaneous recording gates*/
	int respin;          /**<number of repeatitions*/
	unsigned int printnum;        /**<number of printed threads (for debugging)*/
	int gpuid;                    /**<the ID of the GPU to use, starting from 1, 0 for auto*/

	unsigned int *vol;            /**<pointer to the volume*/
	char session[MAX_SESSION_LENGTH]; /**<session id, a string*/
	char isrowmajor;             /**<1 for C-styled array in vol, 0 for matlab-styled array*/
	char isreflect;              /**<1 for reflecting photons at boundary,0 for exiting*/
        char isref3;                 /**<1 considering maximum 3 ref. interfaces; 0 max 2 ref*/
        char isrefint;               /**<1 to consider reflections at internal boundaries; 0 do not*/
	char isnormalized;           /**<1 to normalize the fluence, 0 for raw fluence*/
	char issavedet;              /**<1 to count all photons hits the detectors*/
	char issave2pt;              /**<1 to save the 2-point distribution, 0 do not save*/
	char isgpuinfo;              /**<1 to print gpu info when attach, 0 do not print*/
        char isspecular;             /**<1 calculate the initial specular ref if outside the mesh, 0 do not calculate*/
	char issrcfrom0;             /**<1 do not subtract 1 from src/det positions, 0 subtract 1*/
        char isdumpmask;             /**<1 dump detector mask; 0 not*/
	char autopilot;              /**<1 optimal setting for dedicated card, 2, for non dedicated card*/
	char issaveseed;             /**<1 save the seed for a detected photon, 0 do not save*/
	char issaveexit;             /**<1 save the exit position and dir of a detected photon, 0 do not save*/
	char issaveref;              /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
        char ismomentum;             /**<1 to save momentum transfer for detected photons, implies issavedet=1*/
	char srctype;                /**<0:pencil,1:isotropic,2:cone,3:gaussian,4:planar,5:pattern,\
                                         6:fourier,7:arcsine,8:disk,9:fourierx,10:fourierx2d,11:zgaussian,12:line,13:slit*/
        char outputtype;             /**<'X' output is flux, 'F' output is fluence, 'E' energy deposit*/
        char outputformat;           /**<'mc2' output is text, 'nii': binary, 'img': regular json, 'ubj': universal binary json*/
	char faststep;               /**<1 use tMCimg-like approximated photon stepping (obsolete) */
        float minenergy;             /**<minimum energy to propagate photon*/
	float unitinmm;              /**<defines the length unit in mm for grid*/
        FILE *flog;                  /**<stream handle to print log information*/
        History his;                 /**<header info of the history file*/
	float *exportfield;          /**<memory buffer when returning the flux to external programs such as matlab*/
	float *exportdetected;       /**<memory buffer when returning the partial length info to external programs such as matlab*/
	unsigned long int detectedcount;  /**<total number of detected photons*/
        char rootpath[MAX_PATH_LENGTH]; /**<sets the input and output root folder*/
        char *shapedata;             /**<a pointer points to a string defining the JSON-formatted shape data*/
	int maxvoidstep;             /**< max number of steps that a photon can advance before reaching a non-zero voxel*/
	int voidtime;                /**<1 start counting photon time when moves inside 0 voxels; 0: count time only after enters non-zero voxel*/
	float4 srcparam1;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
	float4 srcparam2;            /**<a quadruplet {x,y,z,w} for additional source parameters*/
        float* srcpattern;           /**<a string for the source form, options include "pencil","isotropic", etc*/
	Replay replay;               /**<a structure to prepare for photon replay*/
	void *seeddata;              /**<poiinter to a buffer where detected photon seeds are stored*/
        int replaydet;               /**<the detector id for which to replay the detected photons, start from 1*/
        char seedfile[MAX_PATH_LENGTH];/**<if the seed is specified as a file (mch), mcx will replay the photons*/
        unsigned int debuglevel;     /**<a flag to control the printing of the debug information*/
        char deviceid[MAX_DEVICE];   /**<a 0-1 mask for all the GPUs, a mask of 1 means this GPU will be used*/
        float workload[MAX_DEVICE];  /**<an array storing the relative weight when distributing photons between multiple GPUs*/
        int parentid;                /**<flag for testing if mcx is executed inside matlab*/
	unsigned int runtime;        /**<variable to store the total kernel simulation time in ms*/

	double energytot;            /**<total launched photon packet weights*/
	double energyabs;            /**<total absorbed photon packet weights*/
	double energyesc;            /**<total escaped photon packet weights*/
	float normalizer;            /**<normalization factor*/
	unsigned int maxjumpdebug;   /**<num of  photon scattering events to save when saving photon trajectory is enabled*/
	unsigned int debugdatalen;   /**<max number of photon trajectory position length*/
	unsigned int gscatter;       /**<after how many scattering events that we can use mus' instead of mus */
	float *exportdebugdata;      /**<pointer to the buffer where the photon trajectory data are stored*/
	uint mediabyte;              /**< how many bytes per media index, mcx supports 1, 2 and 4, 4 is the default*/
	float *dx, *dy, *dz;         /**< anisotropic voxel spacing for x,y,z axis */
	char bc[8];                  /**<boundary condition flag for [-x,-y,-z,+x,+y,+z,unused,unused] */
} Config;

#ifdef	__cplusplus
extern "C" {
#endif
void mcx_savedata(float *dat, size_t len, Config *cfg);
void mcx_savenii(float *dat, size_t len, char* name, int type32bit, int outputformatid, Config *cfg);
void mcx_error(const int id,const char *msg,const char *file,const int linenum);
void mcx_loadconfig(FILE *in, Config *cfg);
void mcx_saveconfig(FILE *in, Config *cfg);
void mcx_readconfig(char *fname, Config *cfg);
void mcx_writeconfig(char *fname, Config *cfg);
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
void mcx_parsecmd(int argc, char* argv[], Config *cfg);
void mcx_usage(Config *cfg,char *exename);
void mcx_printheader(Config *cfg);
void mcx_loadvolume(char *filename,Config *cfg);
void mcx_normalize(float field[], float scale, int fieldlen, int option);
int  mcx_readarg(int argc, char *argv[], int id, void *output,const char *type);
void mcx_printlog(Config *cfg, char *str);
int  mcx_remap(char *opt);
void mcx_maskdet(Config *cfg);
void mcx_dumpmask(Config *cfg);
void mcx_version(Config *cfg);
void mcx_convertrow2col(unsigned int **vol, uint3 *dim);
int  mcx_loadjson(cJSON *root, Config *cfg);
int  mcx_keylookup(char *key, const char *table[]);
int  mcx_lookupindex(char *key, const char *index);
int  mcx_parsedebugopt(char *debugopt,const char *debugflag);
void mcx_savedetphoton(float *ppath, void *seeds, int count, int seedbyte, Config *cfg);
void mcx_loadseedfile(Config *cfg);
void mcx_cleargpuinfo(GPUInfo **gpuinfo);
int  mcx_isbinstr(const char * str);
void mcx_progressbar(float percent, Config *cfg);
void mcx_flush(Config *cfg);

#ifdef MCX_CONTAINER
#ifdef __cplusplus
extern "C"
#endif
 int  mcx_throw_exception(const int id, const char *msg, const char *filename, const int linenum);
 void mcx_matlab_flush(void);
#endif

#ifdef MCX_CONTAINER
  #define MCX_FPRINTF(fp,...) mexPrintf(__VA_ARGS__)
#else
  #define MCX_FPRINTF(fp,...) fprintf(fp,__VA_ARGS__)
#endif

#if defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER)
    int mexPrintf(const char * format, ... );
#else
    int mexPrintf(const char * format, ... );
#endif
int mexEvalString(const char *command);

#ifdef	__cplusplus
}
#endif

#endif
