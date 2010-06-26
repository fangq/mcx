#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct MCXPhoton{
	float x;
	float y;
	float z;
	float w;
}MCXpos;

typedef struct MCXDir{
        float x;
	float y;
	float z;
        float nscat;
}MCXdir;

typedef struct MCXTimer{
        float tscat;
        float t;
	float tnext;
	float ndone;
}MCXtime;

typedef union GPosition{
	MCXpos d;
	float4 v;
}Gpos;

typedef union GDirection{
        MCXdir d;
        float4 v;
}Gdir;

typedef union GLength{
        MCXtime d;
        float4 v;
}Glen;

typedef union GProperty{
        Medium d; /*defined in mcx_utils.h*/
        float4 v;
}Gprop;

void mcx_run_simulation(Config *cfg);
int  mcx_set_gpu(Config *cfg);

#ifdef  __cplusplus
}
#endif

#endif
