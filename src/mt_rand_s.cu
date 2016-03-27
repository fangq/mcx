/*
   A multithreaded C-program for MT19937.
   Original single threaded C reference coded by Takuji Nishimurar
   and Makoto Matsumoto, with initialization improved 2002/1/26.
   Multithreaded C implementation coded by Eric Mills.

   Before using, initialize the state by using mt19937gi(seed)
   or mt19937gai(init_key, key_length) for the global memory versions or
   mt19937si(seed) or mt19937sai(init_key, key_length) for all shared
   memory versions.

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.
   Multithreaded implementation Copyright (C) 2007, Eric Mills.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#ifndef _MCEXTREME_MT_RAND_H
#define _MCEXTREME_MT_RAND_H

#define NVG80                           /* For Nvidia G80 achitecture where mod is VERY slow */

#ifdef NVG80
#define mod(x, y)       ((x) < (y) ? (x) : (x) - (y))   /* Short mod - known input range */
#else
#define mod(x, y)       ((x) % (y))
#endif

#ifdef _WIN32
typedef unsigned int uint;
#endif

typedef char RandType;

#define MCX_RNG_NAME    "MT19937 (shared memory)"

#define N               624
#define M               397
#define INIT_MULT       1812433253      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
#define ARRAY_SEED      19650218        /* Seed for initial setup before incorp array seed */
#define MATRIX_A        0x9908b0df      /* Constant vector a */
#define UPPER_MASK      0x80000000      /* Most significant w-r bits */
#define LOWER_MASK      0x7fffffff      /* Least significant r bits */
#define TEMPER1         0x9d2c5680
#define TEMPER2         0xefc60000

#define RAND_BUF_LEN       1                    //zero-length array buffer
#define RAND_SEED_LEN      1                    //how many 32bit seeds
#define MAX_MT_RAND        4294967296           //2^32
#define R_MAX_MT_RAND      2.3283064365387e-10f //1/2^32
#define LOG_MT_MAX         22.1807097779182f    //log(2^32)

// structure for fast int->float convert

union ifconvert{
     uint i;
     float f;
};

/*************************************************************************************
 * This is a shared memory implementation that keeps the full 626 words of state
 * in shared memory. Faster for heavy random work where you can afford shared mem. */

__shared__ int  mtNexts;                        /* Start of next block of seeds */
__shared__ uint s_seeds[N + 1];
__constant__ uint mag01[2] = {0, MATRIX_A};     /* 2 way bus conflict for each read */

/* Init by single seed - single threaded as only used once */
__device__ void 
mt19937si(uint *n_seed,int idx)
{
    int         i;
    uint     seed;
    if (threadIdx.x == 0)
    {
        mtNexts = 0;
        for (i = 0; i < RAND_SEED_LEN; i++){
	    s_seeds[i] = n_seed[i];
        }
        seed=s_seeds[RAND_SEED_LEN-1];
        for (i = RAND_SEED_LEN; i < N; i++){
            seed = (INIT_MULT * (seed ^ (seed >> 30)) + i);
            s_seeds[i] = seed;
        }
    }
    __syncthreads();                            /* Ensure mtNexts set & needed for mt19937w() */
    return;
}
/* Return next MT random by increasing thread ID for 1-227 threads. */
__device__ uint
mt19937s(void)
{
    int         kk;
    uint        y;
    const int   tid = threadIdx.x;

    kk = mod(mtNexts + tid, N);
    __syncthreads();                            /* Finished with mtNexts & s_seed[] ready from last run */

    if (tid == blockDim.x - 1)
    {
        mtNexts = kk + 1;                       /* Will get modded on next call */
    }
    y = (s_seeds[kk] & UPPER_MASK) | (s_seeds[kk + 1] & LOWER_MASK);
    y = s_seeds[kk < N - M ? kk + M : kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1];
    //y = s_seeds[kk < N - M ? kk + M : kk + (M - N)] ^ (y >> 1) ^ (y & 1 ? MATRIX_A : 0);      // Same speed
    __syncthreads();                            /* All done before we update */

    s_seeds[kk] = y;
    if (kk == 0)                                /* Copy up for next round */
    {
        s_seeds[N] = y;
    }
    y ^= (y >> 11);                             /* Tempering */
    y ^= (y <<  7) & TEMPER1;
    y ^= (y << 15) & TEMPER2;
    y ^= (y >> 18);
    return y;
}

// Return calculated values
__global__ void
mt19937sc(int loops, uint* result, uint* seeds)
{
    mt19937si(seeds,blockIdx.x);
    for (int i = 0; i < loops; ++i){
        result[(blockIdx.x * loops + i) * blockDim.x + threadIdx.x] = mt19937s();
    }
}

// generic interfaces for MC simulations

// t[] and tnew[] are zero-length arrays and are not used,
// the only purpose to keep them is to share the same format
// as in logistic RNG

__device__ void gpu_rng_init(char t[RAND_BUF_LEN], uint *n_seed,int idx){
    mt19937si(n_seed+idx*RAND_SEED_LEN,idx);
}
__device__ void gpu_rng_reseed(RandType t[RAND_BUF_LEN], uint cpuseed[],uint idx,float reseed){
}
__device__ void copystate(RandType *t,RandType *tnew){
}
// transform into [0,1] random number
// use a trick found from 
// http://xor0110.wordpress.com/2010/09/24/how-to-generate-floating-point-random-numbers-efficiently/
__device__ float rand_uniform01(uint ran){
    ifconvert myran;
    myran.i = ran & 0x007fffff | 0x40000000;
    return myran.f*0.5f-1.0f;
}
// generate [0,1] random number for the next scattering length
__device__ float rand_next_scatlen(RandType t[RAND_BUF_LEN]){
    return -logf(rand_uniform01(mt19937s()));
}
// generate [0,1] random number for the next arimuthal angle
__device__ float rand_next_aangle(RandType t[RAND_BUF_LEN]){
    return rand_uniform01(mt19937s());
}
#define rand_next_zangle(t)  rand_next_aangle(t)
#define rand_next_reflect(t) rand_next_aangle(t)
#define rand_do_roulette(t)  rand_next_aangle(t)

// generate random number for the next zenith angle
__device__ void rand_need_more(RandType t[RAND_BUF_LEN]){
    // do nothing
}
#endif
