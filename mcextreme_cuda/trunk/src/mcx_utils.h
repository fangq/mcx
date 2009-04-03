#ifndef _MCEXTREME_UTILITIES_H
#define _MCEXTREME_UTILITIES_H

#include <stdio.h>
#include <vector_types.h>

#define MAX_PROP            256
#define MAX_DETECTORS       256
#define MAX_PATH_LENGTH     1024
#define MAX_SESSION_LENGTH  256

typedef struct MCXMedium{
	float mua;
	float mus;
	float n;
	float g;
} Medium;  /*this order shall match prop.{xyzw} in mcx_main_loop*/

typedef struct MCXConfig{
	int nphoton;      /*reserved for future*/
	int totalmove;    /*total move per photon*/
	int nthread;      /*num of total threads, multiple of 128*/
	int seed;         /*random number generator seed*/
	float3 srcpos;    /*src position in mm*/
	float3 srcdir;    /*src normal direction*/
	float tstart;     /*start time in second*/
	float tstep;      /*time step in second*/
	float tend;       /*end time in second*/
	float3 steps;     /*voxel sizes along x/y/z in mm*/
	uint3 dim;        /*domain size*/
	uint3 crop0;      /*sub-volume for cache*/
	uint3 crop1;      /*the other end of the caching box*/
	int medianum;     /*total types of media*/
	int detnum;       /*total detector numbers*/
	Medium *prop;     /*optical property mapping table*/
	float detradius;  /*detector radius*/
	float4 *detpos;   /*detector positions and radius, overwrite detradius*/
	unsigned char *vol; /*pointer to the volume*/
	char session[MAX_SESSION_LENGTH]; /*session id, a string*/
	unsigned isrowmajor;/*1 for C-styled array in vol, 0 for matlab-styled array*/
	int maxgate;        /*simultaneous recording gates*/
} Config;

void mcx_savedata(float *dat,int len,Config *cfg);
void mcx_error(int id,char *msg);
void mcx_loadconfig(FILE *in, Config *cfg);
void mcx_saveconfig(FILE *in, Config *cfg);
void mcx_readconfig(char *fname, Config *cfg);
void mcx_writeconfig(char *fname, Config *cfg);
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
void mcx_parsecmd(int argc, char* argv[], Config *cfg);
void mcx_usage(void);
void mcx_loadvolume(char *filename,Config *cfg);

#endif
