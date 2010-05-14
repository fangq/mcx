#ifndef _MCEXTREME_GPU_LAUNCH_H
#define _MCEXTREME_GPU_LAUNCH_H

#include "mcx_utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

void mcx_run_simulation(Config *cfg);
int  mcx_set_gpu(Config *cfg);

#ifdef  __cplusplus
}
#endif

#endif
