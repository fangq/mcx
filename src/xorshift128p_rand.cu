/*********************************************************************
*   A Random Number Generator based on the xorshift128+ algorithm    *
*                                                                    *
*   Author: Qianqian Fang <q.fang at neu.edu>                        *
*                                                                    *
*********************************************************************/

#ifndef _MCEXTREME_XORSHIFT128PLUS_RAND_H
#define _MCEXTREME_XORSHIFT128PLUS_RAND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#if defined(__clang__) || defined(_MSC_VER)
  #include "mcx_ieee754.h"
#else
  #include <ieee754.h>
#endif

#define MCX_RNG_NAME       "xorshift128+"

#define RAND_BUF_LEN       2        //register arrays
#define RAND_SEED_LEN      4        //128 bit/8 byte seed needed
#define LOG_MT_MAX         22.1807097779182f

typedef uint64_t  RandType;

__device__ float xorshift128p_nextf(RandType t[RAND_BUF_LEN]){
    union {
        ieee754_double dd;
        uint64_t i;
    } s1;
    const uint64_t s0 = t[1];
    s1.i = t[0];
    t[0] = s0;
    s1.i ^= s1.i << 23; // a
    t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
    s1.i = t[1] + s0;
    s1.dd.ieee.negative = 0;
    s1.dd.ieee.exponent = IEEE754_DOUBLE_BIAS;

    return (float)s1.dd.d - 1.0f;
}

__device__ void xorshift128p_seed (uint seed[4],RandType t[RAND_BUF_LEN])
{
    t[0] = (uint64_t)seed[0] << 32 | seed[1] ;
    t[1] = (uint64_t)seed[2] << 32 | seed[3];
}

__device__ void copystate(RandType *t, RandType *tnew){
    tnew[0]=t[0];
    tnew[1]=t[1];
}

// generate random number for the next zenith angle
__device__ void rand_need_more(RandType t[RAND_BUF_LEN]){
}

__device__ float rand_uniform01(RandType t[RAND_BUF_LEN]){
    return xorshift128p_nextf(t);
}

__device__ void gpu_rng_init(RandType t[RAND_BUF_LEN], uint *n_seed,int idx){
    xorshift128p_seed((n_seed+idx*RAND_SEED_LEN),t);
}
__device__ void gpu_rng_reseed(RandType t[RAND_BUF_LEN],uint cpuseed[],uint idx,float reseed){
}
// generate [0,1] random number for the next scattering length
__device__ float rand_next_scatlen(RandType t[RAND_BUF_LEN]){
    return -logf(rand_uniform01(t)+EPS);
}
// generate [0,1] random number for the next arimuthal angle
__device__ float rand_next_aangle(RandType t[RAND_BUF_LEN]){
    return rand_uniform01(t);
}

#define rand_next_zangle(t)  rand_next_aangle(t)
#define rand_next_reflect(t) rand_next_aangle(t)
#define rand_do_roulette(t)  rand_next_aangle(t)

#endif
