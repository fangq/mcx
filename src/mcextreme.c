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
\file    mcextreme.c

@brief   << MCX main program >>
*******************************************************************************/

#include <stdio.h>
#include "tictoc.h"
#include "mcx_utils.h"
#include "mcx_core.h"
#ifdef _OPENMP
  #include <omp.h>
#endif

int main (int argc, char *argv[]) {
     /*! structure to store all simulation parameters 
      */
     Config  mcxconfig;            /** mcxconfig: structure to store all simulation parameters */
     GPUInfo *gpuinfo=NULL;        /** gpuinfo: structure to store GPU information */
     unsigned int activedev=0;     /** activedev: count of total active GPUs to be used */

     /** 
        To start an MCX simulation, we first create a simulation configuration and
	set all elements to its default settings.
      */
     mcx_initcfg(&mcxconfig);

     /** 
        Then, we parse the full command line parameters and set user specified settings
      */
     mcx_parsecmd(argc,argv,&mcxconfig);

     /** The next step, we identify gpu number and query all GPU info */
     if(!(activedev=mcx_list_gpu(&mcxconfig,&gpuinfo))){
         mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
     }

#ifdef _OPENMP
     /** 
        Now we are ready to launch one thread for each involked GPU to run the simulation 
      */
     omp_set_num_threads(activedev);
     #pragma omp parallel
     {
#endif

     /** 
        This line runs the main MCX simulation for each GPU inside each thread 
      */
     mcx_run_simulation(&mcxconfig,gpuinfo); 

#ifdef _OPENMP
     }
#endif

     /** 
        Once simulation is complete, we clean up the allocated memory in config and gpuinfo, and exit 
      */
     mcx_cleargpuinfo(&gpuinfo);
     mcx_clearcfg(&mcxconfig);
     return 0;
}
