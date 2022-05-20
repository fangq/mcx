/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2022
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
\file    mcx_ieee754.h

@brief   GNU random function header for Windows only
*******************************************************************************/

#ifndef MCX_IEEE754_H
#define MCX_IEEE754_H

#define MCX_LITTLE_ENDIAN 0x41424344UL
#define MCX_BIG_ENDIAN    0x44434241UL
#define MCX_PDP_ENDIAN    0x42414443UL
#define MCX_BYTE_ORDER    ('ABCD')

union ieee754_double {
    double d;

    /* This is the IEEE 754 double-precision format.  */
    struct {
#if MCX_BYTE_ORDER == MCX_BIG_ENDIAN
        unsigned int negative: 1;
        unsigned int exponent: 11;
        /* Together these comprise the mantissa.  */
        unsigned int mantissa0: 20;
        unsigned int mantissa1: 32;
#else
        /* Together these comprise the mantissa.  */
        unsigned int mantissa1: 32;
        unsigned int mantissa0: 20;
        unsigned int exponent: 11;
        unsigned int negative: 1;
#endif              /* Little endian.  */
    } ieee;

    /* This format makes it easier to see if a NaN is a signalling NaN.  */
    struct {
#if MCX_BYTE_ORDER == MCX_BIG_ENDIAN
        unsigned int negative: 1;
        unsigned int exponent: 11;
        unsigned int quiet_nan: 1;
        /* Together these comprise the mantissa.  */
        unsigned int mantissa0: 19;
        unsigned int mantissa1: 32;
#else
        /* Together these comprise the mantissa.  */
        unsigned int mantissa1: 32;
        unsigned int mantissa0: 19;
        unsigned int quiet_nan: 1;
        unsigned int exponent: 11;
        unsigned int negative: 1;
#endif
    } ieee_nan;
};

#define IEEE754_DOUBLE_BIAS 0x3ff /* Added to exponent.  */

#endif
