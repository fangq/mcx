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
\file    mcx_const.h

@brief   Constants assumed in MCX codes
*******************************************************************************/

#ifndef _MCEXTREME_CONSTANT_H
#define _MCEXTREME_CONSTANT_H


#define ONE_PI             3.1415926535897932f     /**< pi */
#define TWO_PI             6.28318530717959f       /**< 2*pi */
#define R_PI               0.318309886183791f      /**<  1/pi */

#define C0                 299792458000.f          /**< speed of light in mm/s */
#define R_C0               3.335640951981520e-12f  /**< 1/C0 in s/mm */

#define JUST_ABOVE_ONE     1.0001f                 /**< test for boundary */
#define JUST_BELOW_ONE     0.9998f                 /**< test for boundary */
#define SAME_VOXEL         -9999.f                 /**< scatter within a voxel */
#define NO_LAUNCH          9999                    /**< when fail to launch, for debug */
#define OUTSIDE_VOLUME_MIN 0xFFFFFFFF              /**< flag indicating the index is outside of the volume from x=xmax,y=ymax,z=zmax*/
#define OUTSIDE_VOLUME_MAX 0x7FFFFFFF              /**< flag indicating the index is outside of the volume from x=0/y=0/z=0*/
#define BOUNDARY_DET_MASK  0xFFFF0000              /**< flag indicating a boundary face is used as a detector*/
#define MAX_PROP_AND_DETECTORS   4000              /**< maximum number of property + number of detectors */
#define SEED_FROM_FILE      -999                   /**< special flag indicating to read seeds from an mch file for replay */
#define NANGLES            1000                    /**< number of discretization points in scattering angles */

#define SIGN_BIT           0x80000000U
#define DET_MASK           0x80000000              /**< mask of the sign bit to get the detector */
#define MED_MASK           0x7FFFFFFF              /**< mask of the remaining bits to get the medium index */
#define LOWER_MASK         0xFF000000              /**< mask of the lower label for SVMC */
#define UPPER_MASK         0x00FF0000              /**< mask of the upper label for SVMC */

#define MCX_DEBUG_RNG          1   /**< debug flags: 1 - run RNG testing kernel and return RNG numbers */
#define MCX_DEBUG_MOVE         2   /**< debug flags: 2 - save and output photon trajectory data */
#define MCX_DEBUG_PROGRESS     4   /**< debug flags: 4 - print progress bar */

#define MEDIA_2LABEL_SPLIT    97   /**<  media Format: 64bit:{[byte: lower label][byte: upper label][byte*3: reference point][byte*3: normal vector]} */
#define MEDIA_2LABEL_MIX      98   /**<  media format: {[int: label1][int: label2][float32: label1 %]} -> 32bit:{[half: label1 %],[byte: label2],[byte: label1]} */
#define MEDIA_LABEL_HALF      99   /**<  media format: {[float32: 1/2/3/4][float32: type][float32: mua/mus/g/n]} -> 32bit:{[half: mua/mus/g/n][int16: [B15-B16: 0/1/2/3][B1-B14: tissue type]} */
#define MEDIA_AS_F2H          100  /**<  media format: {[float32: mua][float32: mus]} -> 32bit:{[half: mua],{half: mus}} */
#define MEDIA_MUA_FLOAT       101  /**<  media format: 32bit:{[float32: mua]} */
#define MEDIA_AS_HALF         102  /**<  media format: 32bit:{[half: mua],[half: mus]} */
#define MEDIA_ASGN_BYTE       103  /**<  media format: 32bit:{[byte: mua],[byte: mus],[byte: g],[byte: n]} */
#define MEDIA_AS_SHORT        104  /**<  media format: 32bit:{[short: mua],[short: mus]} */

#define MCX_DEBUG_REC_LEN  6  /**<  number of floating points per position saved when -D M is used for trajectory */

#define MCX_SRC_PENCIL     0  /**<  default-Pencil beam src, no param */
#define MCX_SRC_ISOTROPIC  1  /**<  isotropic source, no param */
#define MCX_SRC_CONE       2  /**<  uniform cone, srcparam1.x=max zenith angle in rad */
#define MCX_SRC_GAUSSIAN   3  /**<  Gaussian beam, srcparam1.x=sigma */
#define MCX_SRC_PLANAR     4  /**<  quadrilateral src, vectors spanned by srcparam{1}.{x,y,z} */
#define MCX_SRC_PATTERN    5  /**<  same as above, load srcpattern as intensity */
#define MCX_SRC_FOURIER    6  /**<  same as above, srcparam1.w and 2.w defines the spatial freq in x/y */
#define MCX_SRC_ARCSINE    7  /**<  same as isotropic, but more photons near the pole dir */
#define MCX_SRC_DISK       8  /**<  uniform 2D disk along v */
#define MCX_SRC_FOURIERX   9  /**<  same as Fourier, except the v1/v2 and v are orthogonal */
#define MCX_SRC_FOURIERX2D 10 /**<  2D (sin(kx*x+phix)*sin(ky*y+phiy)+1)/2 */
#define MCX_SRC_ZGAUSSIAN  11 /**<  Gaussian zenith anglular distribution */
#define MCX_SRC_LINE       12 /**<  a non-directional line source */
#define MCX_SRC_SLIT       13 /**<  a collimated line source */
#define MCX_SRC_PENCILARRAY 14 /**<  a rectangular array of pencil beams */
#define MCX_SRC_PATTERN3D  15  /**<  a 3D pattern source, starting from srcpos, srcparam1.{x,y,z} define the x/y/z dimensions */
#define MCX_SRC_HYPERBOLOID_GAUSSIAN 16 /**<  Gaussian-beam with spot focus, scrparam1.{x,y,z} define beam waist, distance from source to focus, rayleigh range */

#define SAVE_DETID(a)         ((a)    & 0x1)   /**<  mask to save detector ID*/
#define SAVE_NSCAT(a)         ((a)>>1 & 0x1)   /**<  output partial scattering counts */
#define SAVE_PPATH(a)         ((a)>>2 & 0x1)   /**<  output partial path */
#define SAVE_MOM(a)           ((a)>>3 & 0x1)   /**<  output momentum transfer */
#define SAVE_PEXIT(a)         ((a)>>4 & 0x1)   /**<  save exit positions */
#define SAVE_VEXIT(a)         ((a)>>5 & 0x1)   /**<  save exit vector/directions */
#define SAVE_W0(a)            ((a)>>6 & 0x1)   /**<  save initial weight */
#define SAVE_IQUV(a)          ((a)>>7 & 0x1)   /**<  save stokes parameters */

#define SET_SAVE_DETID(a)     ((a) | 0x1   )   /**<  mask to save detector ID*/
#define SET_SAVE_NSCAT(a)     ((a) | 0x1<<1)   /**<  output partial scattering counts */
#define SET_SAVE_PPATH(a)     ((a) | 0x1<<2)   /**<  output partial path */
#define SET_SAVE_MOM(a)       ((a) | 0x1<<3)   /**<  output momentum transfer */
#define SET_SAVE_PEXIT(a)     ((a) | 0x1<<4)   /**<  save exit positions */
#define SET_SAVE_VEXIT(a)     ((a) | 0x1<<5)   /**<  save exit vector/directions */
#define SET_SAVE_W0(a)        ((a) | 0x1<<6)   /**<  save initial weight */
#define SET_SAVE_IQUV(a)      ((a) | 0x1<<7)   /**<  save stokes parameters */

#define UNSET_SAVE_DETID(a)     ((a) & ~(0x1)   )   /**<  mask to save detector ID*/
#define UNSET_SAVE_NSCAT(a)     ((a) & ~(0x1<<1))   /**<  output partial scattering counts */
#define UNSET_SAVE_PPATH(a)     ((a) & ~(0x1<<2))   /**<  output partial path */
#define UNSET_SAVE_MOM(a)       ((a) & ~(0x1<<3))   /**<  output momentum transfer */
#define UNSET_SAVE_PEXIT(a)     ((a) & ~(0x1<<4))   /**<  save exit positions */
#define UNSET_SAVE_VEXIT(a)     ((a) & ~(0x1<<5))   /**<  save exit vector/directions */
#define UNSET_SAVE_W0(a)        ((a) & ~(0x1<<6))   /**<  save initial weight */
#define UNSET_SAVE_IQUV(a)      ((a) & ~(0x1<<7))   /**<  unsave stokes parameters */

#if !defined(MCX_CONTAINER) && !defined(_MSC_VER)
    #define S_RED     "\x1b[31m"
    #define S_GREEN   "\x1b[32m"
    #define S_YELLOW  "\x1b[33m"
    #define S_BLUE    "\x1b[34m"
    #define S_MAGENTA "\x1b[35m"
    #define S_CYAN    "\x1b[36m"
    #define S_BOLD    "\x1b[1m"
    #define S_ITALIC  "\x1b[3m"
    #define S_RESET   "\x1b[0m"
#else
    #define S_RED
    #define S_GREEN
    #define S_YELLOW
    #define S_BLUE
    #define S_MAGENTA
    #define S_CYAN
    #define S_BOLD
    #define S_ITALIC
    #define S_RESET
#endif

#endif
