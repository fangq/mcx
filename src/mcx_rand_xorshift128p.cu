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

/***************************************************************************//**
\file    xorshift128p_rand.cu

@brief    A Random Number Generator based on the xorshift128+ algorithm
*******************************************************************************/

#ifndef _MCEXTREME_XORSHIFT128PLUS_RAND_H
#define _MCEXTREME_XORSHIFT128PLUS_RAND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#define MCX_RNG_NAME       "xorshift128+"

#define RAND_BUF_LEN       2        //register arrays
#define LOG_MT_MAX         22.1807097779182f

typedef uint64_t  RandType;

/**
 * @brief Change RNG state by one step and provide one random floating point value
 * We only use the 1st 32 random bits of the 64bit state for random number generation
 */

__device__ float xorshift128p_nextf(RandType t[RAND_BUF_LEN]) {
    union {
        uint64_t i;
        float f[2];
        uint  u[2];
    } s1;
    const uint64_t s0 = t[1];
    s1.i = t[0];
    t[0] = s0;
    s1.i ^= s1.i << 23; // a
    t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
    s1.i = t[1] + s0;

    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);

    return s1.f[0] - 1.0f;
}

/**
 * @brief Initialize the xorshift128+ RNG with host seeds
 * 64bit host seeds are computed by the host and are different for each thread
 */

__device__ void xorshift128p_seed (uint seed[4], RandType t[RAND_BUF_LEN]) {
    t[0] = (uint64_t)seed[0] << 32 | seed[1];
    t[1] = (uint64_t)seed[2] << 32 | seed[3];
}

/**
 * @brief Copy the RNG state
 */

__device__ void copystate(RandType* t, RandType* tnew) {
    tnew[0] = t[0];
    tnew[1] = t[1];
}

/**
 * @brief Generate random number for the next zenith angle
 */
__device__ void rand_need_more(RandType t[RAND_BUF_LEN]) {
}

/**
 * @brief Generate random floating point between 0 and 1
 */
__device__ float rand_uniform01(RandType t[RAND_BUF_LEN]) {
    return xorshift128p_nextf(t);
}

/**
 * @brief Inteface function to initialize the RNG
 */
__device__ void gpu_rng_init(RandType t[RAND_BUF_LEN], uint* n_seed, int idx) {
    xorshift128p_seed((n_seed + idx * (sizeof(RandType) >> 2)*RAND_BUF_LEN), t);
}

/**
 * @brief Reseed the RNG during the simulation
 */
__device__ void gpu_rng_reseed(RandType t[RAND_BUF_LEN], uint cpuseed[], uint idx, float reseed) {
}

/**
 * @brief Generate exponentially distributed unitless scattering length
 */
__device__ float rand_next_scatlen(RandType t[RAND_BUF_LEN]) {
    return -logf(rand_uniform01(t) + EPS);
}

/**
 * @brief Generate a random 0-1 floating point for arzith angle calculation
 */
__device__ float rand_next_aangle(RandType t[RAND_BUF_LEN]) {
    return rand_uniform01(t);
}

/**
 * @brief Other needed random values are simplified to 0-1 random floating point generation
 */
#define rand_next_zangle(t)  rand_next_aangle(t)
#define rand_next_reflect(t) rand_next_aangle(t)
#define rand_do_roulette(t)  rand_next_aangle(t)

#endif
