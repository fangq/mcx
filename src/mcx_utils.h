#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>
#include <vector_types.h>
#include "br2cu.h"
#include "cjson/cJSON.h"
#include "float.h"

#define EPS                FLT_EPSILON                   //round-off limit
#define VERY_BIG           (1.f/FLT_EPSILON)             //a big number

#define MAX_PATH_LENGTH     1024
#define MAX_SESSION_LENGTH  256
#define MAX_DEVICE          256

#ifndef MCX_CUDA_ARCH
  #define MCX_CUDA_ARCH       100
#endif

#define MCX_ERROR(id,msg)   mcx_error(id,msg,__FILE__,__LINE__)
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MAX(a,b)           ((a)>(b)?(a):(b))

enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otTaylor};
enum TMCXParent  {mpStandalone, mpMATLAB};

typedef struct MCXMedium{
	float mua;
	float mus;
	float g;
	float n;
} Medium;

typedef struct MCXHistoryHeader{
	char magic[4];
	unsigned int  version;
	unsigned int  maxmedia;
	unsigned int  detnum;
	unsigned int  colcount;
	unsigned int  totalphoton;
	unsigned int  detected;
	unsigned int  savedphoton;
	float unitinmm;
	unsigned int  seedbyte;
	int reserved[6];
} History;

typedef struct PhotonReplay{
	void  *seed;
	float *weight;
	float *tof;
} Replay;

typedef struct MCXGPUInfo {
        char name[MAX_SESSION_LENGTH];
	int id;
	int devcount;
	int major, minor;
	size_t globalmem, constmem, sharedmem;
	int regcount;
	int clock;
	int sm, core;
	int autoblock, autothread;
	int maxgate;
	int maxmpthread;  /**< maximum thread number per multi-processor */
} GPUInfo;

typedef struct MCXConfig{
	int nphoton;      /**<total simulated photon number*/
        unsigned int nblocksize;   /**<thread block size*/
	unsigned int nthread;      /**<num of total threads, multiple of 128*/
	int seed;         /**<random number generator seed*/

	float3 srcpos;    /**<src position in mm*/
	float3 srcdir;    /**<src normal direction*/
	float tstart;     /**<start time in second*/
	float tstep;      /**<time step in second*/
	float tend;       /**<end time in second*/
	float3 steps;     /**<voxel sizes along x/y/z in mm*/

	uint3 dim;        /**<domain size*/
	uint3 crop0;      /**<sub-volume for cache*/
	uint3 crop1;      /**<the other end of the caching box*/
	unsigned int medianum;     /**<total types of media*/
	unsigned int detnum;       /**<total detector numbers*/
	unsigned int maxdetphoton; /**<anticipated maximum detected photons*/
	float detradius;  /**<detector radius*/
        float sradius;    /**<source region radius: if set to non-zero, accumulation 
                            will not perform for dist<sradius; this can reduce
                            normalization error when using non-atomic write*/

	Medium *prop;     /**<optical property mapping table*/
	float4 *detpos;   /**<detector positions and radius, overwrite detradius*/

	unsigned int maxgate;        /**<simultaneous recording gates*/
	unsigned int respin;         /**<number of repeatitions*/
	unsigned int printnum;       /**<number of printed threads (for debugging)*/
	unsigned int reseedlimit;    /**<number of scattering events per thread before the RNG is reseeded*/
	int gpuid;          /**<the ID of the GPU to use, starting from 1, 0 for auto*/

	unsigned char *vol; /**<pointer to the volume*/
	char session[MAX_SESSION_LENGTH]; /**<session id, a string*/
	char isrowmajor;    /**<1 for C-styled array in vol, 0 for matlab-styled array*/
	char isreflect;     /**<1 for reflecting photons at boundary,0 for exiting*/
        char isref3;        /**<1 considering maximum 3 ref. interfaces; 0 max 2 ref*/
        char isrefint;      /**<1 to consider reflections at internal boundaries; 0 do not*/
	char isnormalized;  /**<1 to normalize the fluence, 0 for raw fluence*/
	char issavedet;     /**<1 to count all photons hits the detectors*/
	char issave2pt;     /**<1 to save the 2-point distribution, 0 do not save*/
	char isgpuinfo;     /**<1 to print gpu info when attach, 0 do not print*/
        char issrcfrom0;    /**<1 do not subtract 1 from src/det positions, 0 subtract 1*/
        char isdumpmask;    /**<1 dump detector mask; 0 not*/
	char autopilot;     /**<1 optimal setting for dedicated card, 2, for non dedicated card*/
	char issaveseed;    /**<1 save the seed for a detected photon, 0 do not save*/
	char srctype;       /**<0:pencil,1:isotropic,2:cone,3:gaussian,4:planar,5:pattern,\
                                6:fourier,7:arcsine,8:disk,9:fourierx,10:fourierx2d,11:zgaussian,12:line,13:slit*/
        char outputtype;    /**<'X' output is flux, 'F' output is fluence, 'E' energy deposit*/
	char faststep;
        float minenergy;    /**<minimum energy to propagate photon*/
	float unitinmm;     /**<defines the length unit in mm for grid*/
        FILE *flog;         /**<stream handle to print log information*/
        History his;        /**<header info of the history file*/
	float *exportfield;     /**<memory buffer when returning the flux to external programs such as matlab*/
	float *exportdetected;  /**<memory buffer when returning the partial length info to external programs such as matlab*/
	unsigned int detectedcount; /**<total number of detected photons*/
        char rootpath[MAX_PATH_LENGTH]; /**<sets the input and output root folder*/
        char *shapedata;    /**<a pointer points to a string defining the JSON-formatted shape data*/
	int maxvoidstep;
	int voidtime;
	float4 srcparam1;   /**<a quadruplet {x,y,z,w} for additional source parameters*/
	float4 srcparam2;   /**<a quadruplet {x,y,z,w} for additional source parameters*/
        float* srcpattern;  /**<a string for the source form, options include "pencil","isotropic",\
	                        "cone","gaussian","planar", "pattern","fourier","arcsine","disk",\
				"fourierx","fourierx2d","zgaussian","line","slit"*/
	Replay replay;
	void *seeddata;
        int replaydet;      /**<the detector id for which to replay the detected photons, start from 1*/
        char seedfile[MAX_PATH_LENGTH];
        unsigned int debuglevel; /**<a flag to control the printing of the debug information*/
        char deviceid[MAX_DEVICE];
        float workload[MAX_DEVICE];
        int parentid;
	unsigned int runtime;

	double energytot, energyabs, energyesc;
	float normalizer;
} Config;

#ifdef	__cplusplus
extern "C" {
#endif
void mcx_savedata(float *dat, int len, int doappend, const char *suffix, Config *cfg);
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
void mcx_normalize(float field[], float scale, int fieldlen);
int  mcx_readarg(int argc, char *argv[], int id, void *output,const char *type);
void mcx_printlog(Config *cfg, char *str);
int  mcx_remap(char *opt);
void mcx_maskdet(Config *cfg);
void mcx_version(Config *cfg);
void mcx_convertrow2col(unsigned char **vol, uint3 *dim);
int  mcx_loadjson(cJSON *root, Config *cfg);
int  mcx_keylookup(char *key, const char *table[]);
int  mcx_lookupindex(char *key, const char *index);
int  mcx_parsedebugopt(char *debugopt,const char *debugflag);
void mcx_savedetphoton(float *ppath, void *seeds, int count, int seedbyte, Config *cfg);
void mcx_loadseedfile(Config *cfg);
void mcx_cleargpuinfo(GPUInfo **gpuinfo);
int  mcx_isbinstr(const char * str);

#ifdef MCX_CONTAINER
#ifdef __cplusplus
extern "C"
#endif
 int mcx_throw_exception(const int id, const char *msg, const char *filename, const int linenum);
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
