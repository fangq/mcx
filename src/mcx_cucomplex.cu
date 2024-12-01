/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2024
**
**  \section sref Reference
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
**  \section sformat Formatting
**          Please always run "make pretty" inside the \c src folder before each commit.
**          The above command requires \c astyle to perform automatic formatting.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

#ifndef _MCEXTREME_FREQUENCYDOMAIN_COMPLEX_H
#define _MCEXTREME_FREQUENCYDOMAIN_COMPLEX_H

#include <cuComplex.h>


inline __device__ __host__ void operator +=(cuFloatComplex& a, float b) {
    a.x += b;
}

inline __device__ __host__ void operator=(cuFloatComplex& a, float b) {
    a.x = b;
    a.y = 0.f;
}

// complex op complex
inline __device__ __host__ cuFloatComplex operator +(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a, b);
}

inline __device__ __host__ cuFloatComplex operator -(cuFloatComplex a, cuFloatComplex b) {
    return cuCsubf(a, b);
}

inline __device__ __host__ cuFloatComplex operator *(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a, b);
}

inline __device__ __host__ cuFloatComplex operator /(cuFloatComplex a, cuFloatComplex b) {
    return cuCdivf(a, b);
}


// complex op float
inline __device__ __host__ cuFloatComplex operator +(cuFloatComplex a, float b) {
    return make_cuFloatComplex (cuCrealf(a) + b, cuCimagf(a));
}

inline __device__ __host__ cuFloatComplex operator -(cuFloatComplex a, float b) {
    return make_cuFloatComplex (cuCrealf(a) - b, cuCimagf(a));
}

inline __device__ __host__ cuFloatComplex operator *(cuFloatComplex a, float b) {
    return make_cuFloatComplex (cuCrealf(a) * b, cuCimagf(a) * b);
}

inline __device__ __host__ cuFloatComplex operator /(cuFloatComplex a, float b) {
    b = 1.f / b;
    return make_cuFloatComplex (cuCrealf(a) * b, cuCimagf(a) * b);
}

// float op complex
inline __device__ __host__ cuFloatComplex operator +(float a, cuFloatComplex b) {
    return make_cuFloatComplex (cuCrealf(b) + a, cuCimagf(b));
}

inline __device__ __host__ cuFloatComplex operator -(float a, cuFloatComplex b) {
    return make_cuFloatComplex (cuCrealf(b) - a, cuCimagf(b));
}

inline __device__ __host__ cuFloatComplex operator *(float a, cuFloatComplex b) {
    return make_cuFloatComplex (cuCrealf(b) * a, cuCimagf(b) * a);
}

inline __device__ __host__ cuFloatComplex operator /(float a, cuFloatComplex b) {
    a = 1.f / a;
    return make_cuFloatComplex (cuCrealf(b) * a, cuCimagf(b) * a);
}

#endif
