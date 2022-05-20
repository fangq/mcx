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
\file    mcx_utils.c

@brief   Mie scattering parameters handling for polarized light simulations
*******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include "mcx_utils.h"
#include "mcx_mie.h"
#include "mcx_const.h"

#include <vector_types.h>

#ifdef _MSC_VER
    #include <complex>
    typedef std::complex<double> Dcomplex;
#else
    #include <complex.h>
    typedef double _Complex Dcomplex;
#endif

#ifdef _MSC_VER
inline Dcomplex make_Dcomplex(double re, double im) {
    return Dcomplex(re, im);
}
inline double creal(Dcomplex z) {
    return real(z);
}
inline double cimag(Dcomplex z) {
    return imag(z);
}
inline double cabs(Dcomplex z) {
    return abs(z);
}
inline Dcomplex ctan(Dcomplex z) {
    return tan(z);
}
#else
inline Dcomplex make_Dcomplex(double re, double im) {
    return re + I * im;
}
#endif


Dcomplex Lentz_Dn(Dcomplex z, long n);
void Dn_up(Dcomplex z, long nstop, Dcomplex* D);
void Dn_down(Dcomplex z, long nstop, Dcomplex* D);

/**
 * @brief Precompute scattering parameters based on Mie theory [bohren and huffman]
 *
 * For each combination of sphere and background medium, compute the scattering
 * efficiency and scattering Mueller matrix w.r.t. different scattering angles.
 *
 * @param[in] x: sphere particle size parameters
 * @param[in] m: complex relative refractive index
 * @param[in] mu: precomputed cosine of sampled scattering angles
 * @param[out] smatrix: scattering Mueller matrix
 * @param[out] qsca: scattering efficiency
 */

void Mie(double x, double mx, const double* mu, float4* smatrix, double* qsca, double* g) {
    Dcomplex  m = make_Dcomplex(mx, 0.0);
    Dcomplex* D, *s1, *s2;
    Dcomplex z1 = make_Dcomplex(0.0, 0.0);
    Dcomplex an, bn, bnm1, anm1;
    double* pi0, *pi1, *tau;
    Dcomplex xi, xi0, xi1;
    double psi0, psi1;
    double alpha, beta, factor;
    long n, k, nstop, sign;

    if (x <= 0.0) {
        MCX_ERROR(-6, "sphere size must be positive");
    }

    if (x > 20000.0) {
        MCX_ERROR(-6, "spheres with x>20000 are not validated");
    }

    if ((creal(m) == 0.0 && x < 0.1) || (creal(m) > 0.0 && cabs(m)*x < 0.1)) {
        small_Mie(x, mx, mu, smatrix, qsca, g);
        return;
    }

    nstop = floor(x + 4.05 * pow(x, 0.33333) + 2.0);

    s1 = (Dcomplex*)calloc(NANGLES, sizeof(Dcomplex));
    s2 = (Dcomplex*)calloc(NANGLES, sizeof(Dcomplex));
    pi0 = (double*)calloc(NANGLES, sizeof(double));
    tau = (double*)calloc(NANGLES, sizeof(double));
    pi1 = (double*)calloc(NANGLES, sizeof(double));

    for (int i = 0; i < NANGLES; i++) {
        pi1[i] = 1.0;
    }

    if (creal(m) > 0.0) {
        Dcomplex z = x * m;
        D = (Dcomplex*)calloc(nstop + 1, sizeof(Dcomplex));

        if (fabs(cimag(m)*x) < ((13.78 * creal(m) - 10.8)*creal(m) + 3.9)) {
            Dn_up(z, nstop, D);
        } else {
            Dn_down(z, nstop, D);
        }
    }

    psi0 = sin(x);
    psi1 = psi0 / x - cos(x);
    xi0 = make_Dcomplex(psi0, cos(x));
    xi1 = make_Dcomplex(psi1, cos(x) / x + sin(x));
    *qsca = 0.0;
    *g = 0.0;
    sign = 1;
    anm1 = make_Dcomplex(0.0, 0.0);
    bnm1 = make_Dcomplex(0.0, 0.0);

    for (n = 1; n <= nstop; n++) {
        if (creal(m) == 0.0) {
            an = (n * psi1 / x - psi0) / (n / x * xi1 - xi0);
            bn = psi1 / xi1;
        } else if (cimag(m) == 0.0) {
            z1 = make_Dcomplex(creal(D[n]) / creal(m) + n / x, cimag(z1));
            an = (creal(z1) * psi1 - psi0) / (creal(z1) * xi1 - xi0);

            z1 = make_Dcomplex(creal(D[n]) * creal(m) + n / x, cimag(z1));
            bn = (creal(z1) * psi1 - psi0) / (creal(z1) * xi1 - xi0);
        } else {
            z1 = D[n] / m;
            z1 += n / x;
            an = make_Dcomplex(creal(z1) * psi1 - psi0, cimag(z1) * psi1) / (z1 * xi1 - xi0);

            z1 = D[n] * m;
            z1 += n / x;
            bn = make_Dcomplex(creal(z1) * psi1 - psi0, cimag(z1) * psi1) / (z1 * xi1 - xi0);
        }

        for (k = 0; k < NANGLES; k++) {
            factor = (2.0 * n + 1.0) / (n + 1.0) / n;
            tau[k] = n * mu[k] * pi1[k] - (n + 1) * pi0[k];
            alpha = factor * pi1[k];
            beta = factor * tau[k];
            s1[k] += make_Dcomplex(alpha * creal(an) + beta * creal(bn), alpha * cimag(an) + beta * cimag(bn));
            s2[k] += make_Dcomplex(alpha * creal(bn) + beta * creal(an), alpha * cimag(bn) + beta * cimag(an));
        }

        for (k = 0; k < NANGLES; k++) {
            factor = pi1[k];
            pi1[k] = ((2.0 * n + 1.0) * mu[k] * pi1[k] - (n + 1.0) * pi0[k]) / n;
            pi0[k] = factor;
        }

        factor = 2.0 * n + 1.0;
        *g += (n - 1.0 / n) * (creal(anm1) * creal(an) + cimag(anm1) * cimag(an) + creal(bnm1) * creal(bn) + cimag(bnm1) * cimag(bn));
        *g += factor / n / (n + 1.0) * (creal(an) * creal(bn) + cimag(an) * cimag(bn));
        *qsca += factor * (cabs(an) * cabs(an) + cabs(bn) * cabs(bn));
        sign *= -1;

        factor = (2.0 * n + 1.0) / x;
        xi = factor * xi1 - xi0;
        xi0 = xi1;
        xi1 = xi;

        psi0 = psi1;
        psi1 = creal(xi1);

        anm1 = an;
        bnm1 = bn;
    }

    /* compute scattering efficiency and smatrix */
    (*qsca) *= 2.0 / (x * x);
    (*g) *= 4.0 / (*qsca) / (x * x);

    for (int i = 0; i < NANGLES; i++) {
        smatrix[i].x = 0.5 * cabs(s2[i]) * cabs(s2[i]) + 0.5 * cabs(s1[i]) * cabs(s1[i]);
        smatrix[i].y = 0.5 * cabs(s2[i]) * cabs(s2[i]) - 0.5 * cabs(s1[i]) * cabs(s1[i]);
        smatrix[i].z = creal(conj(s1[i]) * s2[i]);
        smatrix[i].w = cimag(conj(s1[i]) * s2[i]);
    }

    if (creal(m) > 0.0) {
        free(D);
    }

    free(s1);
    free(s2);
    free(pi0);
    free(pi1);
    free(tau);
}

/**
 * @brief Precompute scattering parameters for small particles
 * @param[in] x: sphere particle size parameters
 * @param[in] m: complex relative refractive index
 * @param[in] mu: precomputed cosine of sampled scattering angles
 * @param[out] smatrix: scattering Mueller matrix
 * @param[out] qsca: scattering efficiency
 */

void small_Mie(double x, double mx, const double* mu, float4* smatrix, double* qsca, double* g) {
    Dcomplex  m = make_Dcomplex(mx, 0.0);
    Dcomplex ahat1, ahat2, bhat1;
    Dcomplex z0, m2, m4;
    double x2, x3, x4;

    m2 = m * m;
    m4 = m2 * m2;
    x2 = x * x;
    x3 = x2 * x;
    x4 = x2 * x2;
    z0 = make_Dcomplex(-cimag(m2), creal(m2) - 1.0);
    {
        Dcomplex z1, z2, z3, z4, D;

        if (creal(m) == 0.0) {
            z3 = make_Dcomplex(0.0, 2.0 / 3.0 * (1.0 - 0.2 * x2));
            D = make_Dcomplex(1.0 - 0.5 * x2, 2.0 / 3.0 * x3);
        } else {
            z1 = 2.0 / 3.0 * z0;
            z2 = make_Dcomplex(1.0 - 0.1 * x2 + (4.0 * creal(m2) + 5.0) * x4 / 1400.0, 4.0 * x4 * cimag(m2) / 1400.0);
            z3 = z1 * z2;

            z4 = x3 * (1.0 - 0.1 * x2) * z1;
            D = make_Dcomplex(2.0 + creal(m2) + (1 - 0.7 * creal(m2)) * x2 + (8 * creal(m4) - 385 * creal(m2) + 350.0) / 1400 * x4 + creal(z4),
                              (-0.7 * cimag(m2)) * x2 + (8 * cimag(m4) - 385 * cimag(m2)) / 1400 * x4 + cimag(z4));
        }

        ahat1 = z3 / D;
    }
    {
        Dcomplex z2, z6, z7;

        if (creal(m) == 0.0) {
            bhat1 = make_Dcomplex(0.0, -(1.0 - 0.1 * x2) / 3.0) / make_Dcomplex(1 + 0.5 * x2, -x3 / 3.0);
        } else {
            z2 = x2 / 45.0 * z0;
            z6 = make_Dcomplex(1.0 + (2.0 * creal(m2) - 5.0) * x2 / 70.0, cimag(m2) * x2 / 35.0);
            z7 = make_Dcomplex(1.0 - (2.0 * creal(m2) - 5.0) * x2 / 30.0, -cimag(m2) * x2 / 15.0);
            bhat1 = z2 * (z6 / z7);
        }
    }
    {
        Dcomplex z3, z8;

        if (creal(m) == 0.0) {
            ahat2 = make_Dcomplex(0.0, x2 / 30.0);
        } else {
            z3 = (1.0 - x2 / 14) * x2 / 15.0 * z0;
            z8 = make_Dcomplex(2.0 * creal(m2) + 3.0 - (creal(m2) / 7.0 - 0.5) * x2, 2.0 * cimag(m2) - cimag(m2) / 7.0 * x2);
            ahat2 = z3 / z8;
        }
    }
    {
        double T;

        T = cabs(ahat1) * cabs(ahat1) + cabs(bhat1) * cabs(bhat1) + 5.0 / 3.0 * cabs(ahat2) * cabs(ahat2);
        *qsca = 6.0 * x4 * T;
        *g = (creal(ahat1) * (creal(ahat2) + creal(bhat1)) + cimag(ahat1) * (cimag(ahat2) + cimag(bhat1))) / T;
    }
    {
        double muj, angle;
        Dcomplex s1, s2;

        x3 *= 1.5;
        ahat1 *= x3;
        bhat1 *= x3;
        ahat2 *= x3 * 5.0 / 3.0;

        for (int j = 0; j < NANGLES; j++) {
            muj = mu[j];
            angle = 2 * muj * muj - 1;
            s1 = ahat1 + (bhat1 + ahat2) * muj;
            s2 = bhat1 + (ahat1 + ahat2) * angle;

            smatrix[j].x = 0.5 * cabs(s2) * cabs(s2) + 0.5 * cabs(s1) * cabs(s1);
            smatrix[j].y = 0.5 * cabs(s2) * cabs(s2) - 0.5 * cabs(s1) * cabs(s1);
            smatrix[j].z = creal(conj(s1) * s2);
            smatrix[j].w = cimag(conj(s1) * s2);
        }
    }
}

/**
 * @brief
 * @param z
 * @param n
 */

Dcomplex Lentz_Dn(Dcomplex z, long n) {
    Dcomplex alpha_j1, alpha_j2, zinv, aj;
    Dcomplex alpha, ratio, runratio;

    zinv = 2.0 / z;
    alpha = (n + 0.5) * zinv;
    aj = (-n - 1.5) * zinv;
    alpha_j1 = aj + 1.0 / alpha;
    alpha_j2 = aj;
    ratio = alpha_j1 / alpha_j2;
    runratio = alpha * ratio;

    do {
        aj = zinv - aj;
        alpha_j1 = 1.0 / alpha_j1 + aj;
        alpha_j2 = 1.0 / alpha_j2 + aj;
        ratio = alpha_j1 / alpha_j2;
        zinv = -zinv;
        runratio *= ratio;
    } while (fabs(cabs(ratio) - 1.0) > 1e-12);

    return ((double) - n) / z + runratio;
}

/**
 * @brief
 * @param z
 * @param nstop
 * @param D
 */

void Dn_up(Dcomplex z, long nstop, Dcomplex* D) {
    Dcomplex zinv, k_over_z;
    zinv = 1.0 / z;

    D[0] = 1.0 / ctan(z);

    for (long k = 1; k < nstop; k++) {
        k_over_z = ((double)k) * zinv;
        D[k] = 1.0 / (k_over_z - D[k - 1]) - k_over_z;
    }
}

/**
 * @brief
 * @param z
 * @param nstop
 * @param D
 */

void Dn_down(Dcomplex z, long nstop, Dcomplex* D) {
    Dcomplex zinv, k_over_z;
    zinv = 1.0 / z;

    D[nstop - 1] = Lentz_Dn(z, nstop);

    for (long k = nstop - 1; k >= 1; k--) {
        k_over_z = ((double)k) * zinv;
        D[k - 1] = k_over_z - 1.0 / (D[k] + k_over_z);
    }
}