/*********************************************************************
*A Random Number Generator based on coupled chaotic Logistic lattice *
*                                                                    *
*  (both double and single precision random numbers are supported)   *
*                                                                    *
*  Author: Qianqian Fang <q.fang at neu.edu>              *
*                                                                    *
*  History: 2009/03/02  CUDA version based on Neal Wagner 1993       *
*         http://www.cs.utsa.edu/~wagner/pubs/logistic/logistic.pdf  *
*                                                                    *
*********************************************************************/

#ifndef _MCEXTREME_LOGISTIC_RAND_H
#define _MCEXTREME_LOGISTIC_RAND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <float.h>

#define MCX_RNG_NAME       "Logistic-Lattice"

#define RAND_BUF_LEN       5        //register arrays
#define R_PI               0.318309886183791f
#define INIT_LOGISTIC      100
#define INIT_MULT          1812433253      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */

#ifndef DOUBLE_PREC_LOGISTIC
  typedef float RandType;
  #define FUN(x)               (4.f*(x)*(1.f-(x)))
  #define NU                   1e-7f
  #define NU2                  (1.f-2.f*NU)
  #define MIN_INVERSE_LIMIT    1e-7f
  #define logistic_uniform(v)  (acosf(1.f-2.f*(v))*R_PI)
  #define R_MAX_C_RAND         (1.f/RAND_MAX)
  #define LOG_MT_MAX           22.1807097779182f
#else
  typedef double RandType;
  #define FUN(x)               (4.0*(x)*(1.0-(x)))
  #define NU                   1e-14
  #define NU2                  (1.0-2.0*NU)
  #define MIN_INVERSE_LIMIT    1e-12
  #define logistic_uniform(v)  (acos(1.0-2.0*(v))*R_PI)
  #define R_MAX_C_RAND         (1./RAND_MAX)
  #define LOG_MT_MAX           22.1807097779182
#endif

#define RING_FUN(x,y,z)        (NU2*(x)+NU*((y)+(z)))

//typedef unsigned int uint;

__device__ void logistic_step(RandType *t, RandType *tnew){
    t[0]=FUN(t[0]);
    t[1]=FUN(t[1]);
    t[2]=FUN(t[2]);
    t[3]=FUN(t[3]);
    t[4]=FUN(t[4]);
    tnew[4]=RING_FUN(t[0],t[4],t[1]);   /* shuffle the results by separation of 1*/
    tnew[0]=RING_FUN(t[1],t[0],t[2]);
    tnew[1]=RING_FUN(t[2],t[1],t[3]);
    tnew[2]=RING_FUN(t[3],t[2],t[4]);
    tnew[3]=RING_FUN(t[4],t[3],t[0]);
}
__device__ void copystate(RandType *t, RandType *tnew){
    tnew[0]=t[0];
    tnew[1]=t[1];
    tnew[2]=t[2];
    tnew[3]=t[3];
    tnew[4]=t[4];
}
// generate random number for the next zenith angle
__device__ void rand_need_more(RandType t[RAND_BUF_LEN]){
    RandType tnew[RAND_BUF_LEN]={0.f};
    logistic_step(t,tnew);
    logistic_step(tnew,t);
}

__device__ void logistic_init(RandType *t,uint seed[],uint idx){
     int i;
     for(i=0;i<RAND_BUF_LEN;i++)
           t[i]=(RandType)seed[idx*RAND_BUF_LEN+i]*R_MAX_C_RAND;

     for(i=0;i<INIT_LOGISTIC;i++)  /*initial randomization*/
           rand_need_more(t);
}
// transform into [0,1] random number
__device__ RandType rand_uniform01(RandType t[RAND_BUF_LEN]){
    rand_need_more(t);
    return logistic_uniform(t[0]);
}
__device__ void gpu_rng_init(RandType t[RAND_BUF_LEN],uint *n_seed,int idx){
    logistic_init(t,n_seed,idx);
}
__device__ void gpu_rng_reseed(RandType t[RAND_BUF_LEN],uint cpuseed[],uint idx,float reseed){
    uint *newt=(uint *)(&t[0]), seed,i;
    seed=*((uint *)&reseed);
    for (i = 0; i<RAND_BUF_LEN; i++){
        seed = (INIT_MULT * (seed ^ (seed >> 30)) + i) ^ cpuseed[idx*RAND_BUF_LEN+i];
        newt[i] = seed % RAND_MAX;
        t[i]=(RandType)(newt[i]*R_MAX_C_RAND);
    }
    for(i=0;i<INIT_LOGISTIC;i++)  /*initial randomization*/
        rand_need_more(t);
}
// generate [0,1] random number for the next scattering length
__device__ float rand_next_scatlen(RandType t[RAND_BUF_LEN]){
	return -logf(rand_uniform01(t) + EPS);
}
// generate [0,1] random number for the next arimuthal angle
__device__ float rand_next_aangle(RandType t[RAND_BUF_LEN]){
    return rand_uniform01(t);
}

#define rand_next_zangle(t)  rand_next_aangle(t)
#define rand_next_reflect(t) rand_next_aangle(t)
#define rand_do_roulette(t)  rand_next_aangle(t)

#endif
