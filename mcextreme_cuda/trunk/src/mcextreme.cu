//////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
//  
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//  History: 
//    2009/02/14 initial version written in BrookGPU
//    2009/02/15 translated to CUDA
//    2009/02/20 translated to Brook+
//    2009/02/21 added MT random number generator initial version
//    2009/02/24 MT rand now works fine, added FAST_MATH
//    2009/02/25 added CACHE_MEDIA read
//    2009/02/27 early support of boundary reflection
//    2009/03/02 added logistic-map based random number generator
//    2009/04/01 split the main function to units, add utils. and config file
//    2009/04/03 time gating support
//    2009/04/07 energy conservation for normalization of the solution
//
// License: unpublished version, use by author's permission only
//
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "tictoc.h"
#include "mcx_utils.h"
#include "mcx_core.cu"

int main (int argc, char *argv[]) {
     Config mcxconfig;
     mcx_initcfg(&mcxconfig);
     
     // parse command line options to initialize the configurations
     mcx_parsecmd(argc,argv,&mcxconfig);
     
     // identify gpu number and set one gpu active
     if(!mcx_set_gpu(mcxconfig.isgpuinfo)){
         mcx_error(-1,"no GPU device found\n");
     }
          
     // this launches the MC simulation
     mcx_run_simulation(&mcxconfig);
     
     // clean up the allocated memory in the config
     mcx_clearcfg(&mcxconfig);
     return 0;
}
