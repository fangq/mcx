/*******************************************************************************
**
**  Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**  
**  Author     : Qianqian Fang
**  Email      : <q.fang at neu.edu>
**  Institution: Department of Bioengineering, Northeastern University
**  Address    : 360 Huntington Ave, Boston, MA 02115
**  Homepage   : http://nmr.mgh.harvard.edu/~fangq/
**
**  MCX Web    : http://mcx.sourceforge.net
**
**  License    : GNU General Public License version 3 (GPLv3), see LICENSE.txt
**
*******************************************************************************/

#include <stdio.h>
#include "tictoc.h"
#include "mcx_utils.h"
#include "mcx_core.h"
#ifdef _OPENMP
  #include <omp.h>
#endif

int main (int argc, char *argv[]) {
     Config  mcxconfig;
     GPUInfo *gpuinfo=NULL;
     unsigned int activedev=0;

     mcx_initcfg(&mcxconfig);

     // parse command line options to initialize the configurations
     mcx_parsecmd(argc,argv,&mcxconfig);

     // identify gpu number and set one gpu active
     if(!(activedev=mcx_list_gpu(&mcxconfig,&gpuinfo))){
         mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
     }

#ifdef _OPENMP
     omp_set_num_threads(activedev);
     #pragma omp parallel
     {
#endif

     // this launches the MC simulation
     mcx_run_simulation(&mcxconfig,gpuinfo);

#ifdef _OPENMP
     }
#endif

     // clean up the allocated memory in config and gpuinfo
     mcx_cleargpuinfo(&gpuinfo);
     mcx_clearcfg(&mcxconfig);
     return 0;
}
