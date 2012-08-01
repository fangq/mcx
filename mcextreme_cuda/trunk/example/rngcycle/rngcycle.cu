/////////////////////////////////////////////////////////////////////
//
//  Monte-Carlo Extreme (MCX) - a GPU accelerated Monte-Carlo Simulation
//  Random Number Generator Benchmark
//
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//  History: 
//     2009/04/16 test for the speed of the two RNG in MCX
//
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "br2cu.h"

#define USE_OS_TIMER  /* use MT19937 RNG */
#include "tictoc.c"

#define ABS(a) ((a>0)?(a):(-a))

#define RAND_TEST_LEN 5

#ifdef USE_MT_RAND  /* use MT19937 RNG */

//===================================================================
// GPU kernels for MT19937
//===================================================================

#include "mt_rand_s.cu"

#else   /* use Logistic-map lattice RNG */

//===================================================================
// GPU kernels for Logistic-map lattice LL3 and LL5
//===================================================================
//#include "logistic_rand_ring3.cu" // for LL3 RNG

#include "logistic_rand.cu"

#endif

kernel void bench_rng(uint seed[],int output[],RandType last[], int loop){
     int idx= blockDim.x * blockIdx.x + threadIdx.x;
     int i,j,flag=1;
     
     RandType t[RAND_TEST_LEN],tnew[RAND_TEST_LEN],t0[RAND_TEST_LEN];
     gpu_rng_init(t,tnew,seed,idx);

     for(i=0;i<RAND_TEST_LEN;i++)
          t0[i]=t[i];
     for(i=0;i<loop;i++){
          int isbad=1;
          rand_need_more(t,tnew);

          for(j=0;j<RAND_TEST_LEN;j++)
               isbad &= (t[j]==t0[j]);
          if(isbad){
               flag=-1;
               break;
          }
          isbad=1;
          for(j=0;j<RAND_TEST_LEN;j++)
               isbad &= (t[j]==0.f);
          if(isbad)
               break;
     }
     output[idx]= flag*i;
     for(i=0;i<RAND_TEST_LEN;i++)
#ifdef USE_MT_RAND  /* use MT19937 RNG */
          last[idx*RAND_TEST_LEN+i]=rand_uniform01(mt19937s());
#else
          last[idx*RAND_TEST_LEN+i]=rand_uniform01(t[i]);
#endif
}

//===================================================================
// utility functions
//===================================================================

void usage(char *exename){
	printf("usage: %s <num_block|128> <num_thread|128> <rand_per_thread|10000> <num_repeat|10>\n",exename);
}

void mcx_savedata(float *dat,int len,char *name){
     FILE *fp;
     fp=fopen(name,"wb");
     fwrite(dat,sizeof(float),len,fp);
     fclose(fp);
}

//===================================================================
// main program
//===================================================================

int main(int argc, char *argv[]){
    dim3 griddim=128, blockdim=128;
    int count=999,repeat=10, threadnum,tic,tic2,toc;
    uint   *Pseed;
    uint   *gPseed;
    int   *Poutput;
    int   *gPoutput;
    RandType *Prand;
    RandType *gPrand;
    double totalrand;
    int i;

    // parse arguments
    
    if(argc==1){
	usage(argv[0]);
	exit(0);
    }
    if(argc>=2) griddim.x=atoi(argv[1]);
    if(argc>=3) blockdim.x=atoi(argv[2]);
    if(argc>=4) count=atoi(argv[3]);
    if(argc>=5) repeat=atoi(argv[4]);
    
    if(RAND_TEST_LEN>0)
        count=(count/RAND_TEST_LEN)*RAND_TEST_LEN; // make count modulo of 5

    threadnum=griddim.x*blockdim.x;

    // allocate CPU and GPU arrays
    
    Pseed=(uint*)malloc(sizeof(uint)*threadnum*RAND_TEST_LEN);
    cudaMalloc((void **) &gPseed, sizeof(uint)*threadnum*RAND_TEST_LEN);
    Poutput=(int*)malloc(sizeof(int)*threadnum);
    cudaMalloc((void **) &gPoutput, sizeof(int)*threadnum);
    Prand=(RandType*)malloc(sizeof(RandType)*threadnum*RAND_TEST_LEN);
    cudaMalloc((void **) &gPrand, sizeof(RandType)*threadnum*RAND_TEST_LEN);

    // initialize seeds
        
    srand(time(0));
    for (i=0; i<threadnum*RAND_TEST_LEN; i++){
	   Pseed[i]=rand();
    }

    // copy CPU data to GPU
    
    tic=StartTimer();
    totalrand=(double)threadnum*count*repeat;
    printf("total thread=%d, total rand num=%f\n",threadnum,totalrand);
    cudaMemcpy(gPseed, Pseed, sizeof(uint)*threadnum*RAND_TEST_LEN,cudaMemcpyHostToDevice);

    printf("init complete : %d ms\n",GetTimeMillis()-tic);

    // begin benchmark
    
    tic2=StartTimer();
    for(i=0;i<repeat;i++)
        bench_rng<<<griddim,blockdim>>>(gPseed,gPoutput,gPrand,count);

    // get only one element to make sure all kernels are complete
    
    cudaMemcpy(Pseed,gPseed, sizeof(uint),cudaMemcpyDeviceToHost);
    toc=GetTimeMillis()-tic2;

    // take results back to CPU
        
    printf("kernel complete: %d ms\nspeed: %f random numbers per second\n",\
        toc, (1000./toc)*totalrand);
//    cudaMemcpy(Pseed, gPseed,sizeof(uint)*threadnum*RAND_TEST_LEN,cudaMemcpyDeviceToHost);
    cudaMemcpy(Poutput,gPoutput, sizeof(int)*threadnum,cudaMemcpyDeviceToHost);
    cudaMemcpy(Prand,gPrand, sizeof(RandType)*threadnum*RAND_TEST_LEN,cudaMemcpyDeviceToHost);

    // dump random numbers to disk
    for(i=0;i<threadnum;i++)
//       if(ABS(Poutput[i])<count-1)
         printf("i=%d\tc=%d\tv0=%.16e %.16e %.16e\n",i,Poutput[i],Prand[i*RAND_TEST_LEN],Prand[i*RAND_TEST_LEN+1],Prand[i*RAND_TEST_LEN+2]);

    // memory clean-up
    
    cudaFree(gPseed);
    cudaFree(gPoutput);
    free(Pseed);
    free(Poutput);
    return 0;
}
