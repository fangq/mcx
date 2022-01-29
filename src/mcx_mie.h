/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2021
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
\file    mcx_mie.h

@brief   MCX Mie scattering functions header
*******************************************************************************/
 
#ifndef _MCEXTREME_MIE_H
#define _MCEXTREME_MIE_H

#include <vector_types.h>

#ifdef _MSC_VER
#include <complex>
typedef std::complex<double> Dcomplex;
#else
#include <complex.h>
typedef double _Complex Dcomplex;
#endif

#ifdef _MSC_VER
inline static Dcomplex make_Dcomplex(double re, double im) {
    return Dcomplex(re,im);
}
inline static double creal(Dcomplex z) {
    return real(z);
}
inline static double cimag(Dcomplex z) {
    return imag(z);
}
inline static double cabs(Dcomplex z) {
    return abs(z);
}
inline static Dcomplex ctan(Dcomplex z) {
    return tan(z);
}
#else
inline static Dcomplex make_Dcomplex(double re, double im) {
    return re + I * im;
}
#endif

Dcomplex Lentz_Dn(Dcomplex z,long n);
void Dn_up(Dcomplex z, long nstop, Dcomplex *D);
void Dn_down(Dcomplex z, long nstop, Dcomplex *D);

#ifdef	__cplusplus
extern "C" {
#endif

void Mie(double x, double nre, const double *mu, float4 *smatrix, double *qsca, double *g);
void small_Mie(double x, double nre, const double *mu, float4 *smatrix, double *qsca, double *g);

#ifdef	__cplusplus
}
#endif

#endif