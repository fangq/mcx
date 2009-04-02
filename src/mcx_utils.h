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
} Medium;

typedef struct MCXConfig{
	int nphoton; // this is not used
	int totalmove;
	int nthread;
	int seed;
	float3 srcpos;
	float3 srcdir;
	float tstart;
	float tstep;
	float tend;
	float3 steps;
	uint3 dim;
	uint3 crop0;
	uint3 crop1;
	int medianum;
	int detnum;
	Medium *prop;
	float detradius;
	float4 *detpos;
	unsigned char *vol;
	char session[MAX_SESSION_LENGTH];
	unsigned isrowmajor;
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
