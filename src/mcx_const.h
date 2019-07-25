/*******************************************************************************
**
**  \mainpage Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
**
**  \author Qianqian Fang <q.fang at neu.edu>
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  
**  \section slicense License
**        GNU General Public License v3, see LICENSE.txt for details
**
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
#define MAX_PROP_AND_DETECTORS   4000              /**< maximum number of property + number of detectors */
#define SEED_FROM_FILE      -999                   /**< special flag indicating to read seeds from an mch file for replay */

#define SIGN_BIT           0x80000000U
#define DET_MASK           0x80000000              /**< mask of the sign bit to get the detector */
#define MED_MASK           0x7FFFFFFF              /**< mask of the remaining bits to get the medium index */

#define MCX_DEBUG_RNG       1                   /**< MCX debug flags */
#define MCX_DEBUG_MOVE      2
#define MCX_DEBUG_PROGRESS  4

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

#define SAVE_DETID(a)         ((a)    & 0x1)   /**<  mask to save detector ID*/
#define SAVE_NSCAT(a)         ((a)>>1 & 0x1)   /**<  output partial scattering counts */
#define SAVE_PPATH(a)         ((a)>>2 & 0x1)   /**<  output partial path */
#define SAVE_MOM(a)           ((a)>>3 & 0x1)   /**<  output momentum transfer */
#define SAVE_PEXIT(a)         ((a)>>4 & 0x1)   /**<  save exit positions */
#define SAVE_VEXIT(a)         ((a)>>5 & 0x1)   /**<  save exit vector/directions */
#define SAVE_W0(a)            ((a)>>6 & 0x1)   /**<  save initial weight */

#define SET_SAVE_DETID(a)     ((a) | 0x1   )   /**<  mask to save detector ID*/
#define SET_SAVE_NSCAT(a)     ((a) | 0x1<<1)   /**<  output partial scattering counts */
#define SET_SAVE_PPATH(a)     ((a) | 0x1<<2)   /**<  output partial path */
#define SET_SAVE_MOM(a)       ((a) | 0x1<<3)   /**<  output momentum transfer */
#define SET_SAVE_PEXIT(a)     ((a) | 0x1<<4)   /**<  save exit positions */
#define SET_SAVE_VEXIT(a)     ((a) | 0x1<<5)   /**<  save exit vector/directions */
#define SET_SAVE_W0(a)        ((a) | 0x1<<6)   /**<  save initial weight */

#define UNSET_SAVE_DETID(a)     ((a) & ~(0x1)   )   /**<  mask to save detector ID*/
#define UNSET_SAVE_NSCAT(a)     ((a) & ~(0x1<<1))   /**<  output partial scattering counts */
#define UNSET_SAVE_PPATH(a)     ((a) & ~(0x1<<2))   /**<  output partial path */
#define UNSET_SAVE_MOM(a)       ((a) & ~(0x1<<3))   /**<  output momentum transfer */
#define UNSET_SAVE_PEXIT(a)     ((a) & ~(0x1<<4))   /**<  save exit positions */
#define UNSET_SAVE_VEXIT(a)     ((a) & ~(0x1<<5))   /**<  save exit vector/directions */
#define UNSET_SAVE_W0(a)        ((a) & ~(0x1<<6))   /**<  save initial weight */

#ifndef MCX_CONTAINER
  #define S_RED     "\x1b[31m"
  #define S_GREEN   "\x1b[32m"
  #define S_YELLOW  "\x1b[33m"
  #define S_BLUE    "\x1b[34m"
  #define S_MAGENTA "\x1b[35m"
  #define S_CYAN    "\x1b[36m"
  #define S_BOLD     "\x1b[1m"
  #define S_ITALIC   "\x1b[3m"
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
