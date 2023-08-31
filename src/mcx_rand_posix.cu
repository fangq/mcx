/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2023
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#ifndef _MCEXTREME_POSIX_RAND_H
#define _MCEXTREME_POSIX_RAND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <ieee754.h>
#include <stdint.h>

#define MCX_RNG_NAME       "POSIX erand48"

#define RAND_BUF_LEN       4        //register arrays
#define LOG_MT_MAX         22.1807097779182f
#define LCG_MULTIPLIER     0x5deece66dull
#define LCG_INCREMENT      0xb

typedef unsigned int       uint32;
typedef unsigned short     RandType;

__device__ int __drand48_iterate (RandType t[RAND_BUF_LEN]) {
    uint64_t result;

    result = (uint64_t) t[2] << 32 | (uint32) t[1] << 16 | t[0];
    result = result * LCG_MULTIPLIER + LCG_INCREMENT;

    t[0] = result & 0xffff;
    t[1] = (result >> 16) & 0xffff;
    t[2] = (result >> 32) & 0xffff;

    return 0;
}

__device__ float __erand48_r (RandType t[RAND_BUF_LEN]) {
    union ieee754_double temp;

    /* Compute next state.  */
    __drand48_iterate (t);

    /* Construct a positive double with the 48 random bits distributed over
     its fractional part so the resulting FP number is [0.0,1.0).  */

    temp.ieee.negative = 0;
    temp.ieee.exponent = IEEE754_DOUBLE_BIAS;
    temp.ieee.mantissa0 = (t[2] << 4) | (t[1] >> 12);
    temp.ieee.mantissa1 = ((t[1] & 0xfff) << 20) | (t[0] << 4);

    /* Please note the lower 4 bits of mantissa1 are always 0.  */
    return (float)temp.d - 1.0f;
}

__device__ void __seed48_r (ushort seed16v[3], RandType t[RAND_BUF_LEN]) {
    t[0] = seed16v[0];
    t[1] = seed16v[1];
    t[2] = seed16v[2];
}

__device__ void copystate(RandType* t, RandType* tnew) {
    tnew[0] = t[0];
    tnew[1] = t[1];
    tnew[2] = t[2];
}

// generate random number for the next zenith angle
__device__ void rand_need_more(RandType t[RAND_BUF_LEN]) {
}

__device__ float rand_uniform01(RandType t[RAND_BUF_LEN]) {
    return __erand48_r(t);
}

__device__ void gpu_rng_init(RandType t[RAND_BUF_LEN], uint* n_seed, int idx) {
    __seed48_r((ushort*)(n_seed) + idx * RAND_BUF_LEN, t);
}
__device__ void gpu_rng_reseed(RandType t[RAND_BUF_LEN], uint cpuseed[], uint idx, float reseed) {
}
// generate [0,1] random number for the next scattering length
__device__ float rand_next_scatlen(RandType t[RAND_BUF_LEN]) {
    float ran = __erand48_r(t);
    return ((ran == 0.f) ? LOG_MT_MAX : (-logf(ran)));
}
// generate [0,1] random number for the next arimuthal angle
__device__ float rand_next_aangle(RandType t[RAND_BUF_LEN]) {
    return rand_uniform01(t);
}

#define rand_next_zangle(t)  rand_next_aangle(t)
#define rand_next_reflect(t) rand_next_aangle(t)
#define rand_do_roulette(t)  rand_next_aangle(t)

#endif
