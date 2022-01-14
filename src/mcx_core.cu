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
\file    mcx_core.cu

@brief   GPU kernel for MC simulations and CUDA host code

This unit contains both the GPU kernels (running on the GPU device) and host code
(running on the host) that initializes GPU buffers, calling kernels, retrieving
all computed results (fluence, diffuse reflectance detected photon data) from GPU,
and post processing, such as normalization, saving data to file etc. The main 
function of the GPU kernel is \c mcx_main_loop and the main function of the
host code is \c mcx_run_simulation.

This unit is written with CUDA-C and shall be compiled using nvcc in cuda-toolkit.

*******************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>

#include "br2cu.h"
#include "mcx_core.h"
#include "tictoc.h"
#include "mcx_const.h"

//#ifdef USE_HALF                     //< use half-precision for ray-tracing
    #include "cuda_fp16.h"
//#endif

#ifdef USE_DOUBLE
    typedef double OutputType;
    #define SHADOWCOUNT 1
    #define ZERO        0.0
#else
    typedef float OutputType;
    #define SHADOWCOUNT 2
    #define ZERO        0.f
#endif

#if defined(USE_XOROSHIRO128P_RAND)
    #include "xoroshiro128p_rand.cu" //< Use USE_XOROSHIRO128P_RAND macro to enable xoroshiro128p+ RNG (XORSHIFT128P)
#elif defined(USE_LL5_RAND)
    #include "logistic_rand.cu"     //< Use USE_LL5_RAND macro to enable Logistic Lattice ring 5 RNG (LL5), used in the original MCX paper but depreciated
#elif defined(USE_POSIX_RAND)
    #include "posix_rand.cu"        //< Use USE_POSIX_RAND to enable POSIX erand48 RNG (POSIX)
#else                               //< The default RNG method is use xorshift128+ RNG (XORSHIFT128P)
    #include "xorshift128p_rand.cu"
#endif

#ifdef _OPENMP                      //< If compiled with -fopenmp with GCC, this enables OpenMP multi-threading for running simulation on multiple GPUs
    #include <omp.h>
#endif

#define CUDA_ASSERT(a)      mcx_cu_assess((a),__FILE__,__LINE__) //< macro to report CUDA errors

#define FL3(f) make_float3(f,f,f)

/**
 * @brief Adding two float3 vectors c=a+b
 */

__device__ float3 operator +(const float3 &a, const float3 &b){
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Increatment a float3 vector by another float3, a+=b
 */

__device__ void operator +=(float3 &a, const float3 &b){
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

/**
 * @brief Subtracting two float3 vectors c=a+b
 */

__device__ float3 operator -(const float3 &a, const float3 &b){
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


/**
 * @brief Negating a float3 vector c=-a
 */

__device__ float3 operator -(const float3 &a){
	return make_float3(-a.x, -a.y, -a.z);
}

/**
 * @brief Front-multiplying a float3 with a scalar c=a*b
 */

__device__ float3 operator *(const float &a, const float3 &b){
	return make_float3(a * b.x, a * b.y, a * b.z);
}

/**
 * @brief Post-multiplying a float3 with a scalar c=a*b
 */

__device__ float3 operator *(const float3 &a, const float &b){
	return make_float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Multiplying two float3 vectors c=a*b
 */

__device__ float3 operator *(const float3 &a, const float3 &b){
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

/**
 * @brief Dot-product of two float3 vectors c=a*b
 */

__device__ float dot(const float3 &a, const float3 &b){
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Concatenated optical properties and det positions, stored in constant memory
 *
 * The first cfg.maxmedia elements of this array contain the optical properties of the 
 * domains. Format: {x}:mua,{y}:mus,{z}:anisotropy (g),{w}:refractive index (n).
 * The following cfg.detnum elements of this array contains the detector information.
 * Format: {x,y,z}: the x/y/z coord. of the detector, and {w}: radius; all in grid unit.
 * The total length (both media properties and detector) is defined by 
 * MAX_PROP_AND_DETECTORS, which is 4000 to fully utilize the constant memory space
 * (64kb=4096 float4)
 */

__constant__ float4 gproperty[MAX_PROP_AND_DETECTORS];


/**
 * @brief Simulation constant parameters stored in the constant memory
 *
 * This variable stores all constants used in the simulation.
 */

__constant__ MCXParam gcfg[1];

/**
 * @brief Global variable to store the number of photon movements for debugging purposes
 */

__device__ uint gjumpdebug[1];

/**
 * @brief Pointer to the shared memory (storing photon data and spilled registers)
 */

extern __shared__ char sharedmem[];

/**
 * @brief Texture memory for storing media indices
 *
 * Tested with texture memory for media, only improved 1% speed
 * to keep code portable, use global memory for now
 * also need to change all media[idx1d] to tex1Dfetch() below
 */

//texture<uchar, 1, cudaReadModeElementType> texmedia;

/**
 * @brief Floating-point atomic addition
 */
 
__device__ inline OutputType atomicadd(OutputType* address, OutputType value){

#if ! defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200 //< for Fermi, atomicAdd supports floats

  return atomicAdd(address,value);

#else

// float-atomic-add from 
// http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
  return old;
#endif

}

/**
 * @brief Reset shared memory buffer to store photon partial-path data for a new photon
 * @param[in] p: pointer to the partial-path buffer
 * @param[in] maxmediatype: length of the buffer to be reset
 */

__device__ inline void clearpath(float *p,int maxmediatype){
      uint i;
      for(i=0;i<maxmediatype;i++)
      	   p[i]=0.f;
}

#ifdef SAVE_DETECTORS

/**
 * @brief Testing which detector detects an escaping photon
 * @param[in] p0: the position of the escaping photon
 * @return the index of the photon that captures this photon; 0 if none.
 */

__device__ inline uint finddetector(MCXpos *p0){
      uint i;
      for(i=gcfg->maxmedia+1;i<gcfg->maxmedia+gcfg->detnum+1;i++){
      	if((gproperty[i].x-p0->x)*(gproperty[i].x-p0->x)+
	   (gproperty[i].y-p0->y)*(gproperty[i].y-p0->y)+
	   (gproperty[i].z-p0->z)*(gproperty[i].z-p0->z) < gproperty[i].w*gproperty[i].w){
	        return i-gcfg->maxmedia;
	   }
      }
      return 0;
}

__device__ inline void saveexitppath(float n_det[],float *ppath,MCXpos *p0,uint *idx1d){
      if(gcfg->issaveref>1){
          if(*idx1d>=gcfg->maxdetphoton)
	      return;
          uint baseaddr=(*idx1d)*gcfg->reclen;
	  n_det[baseaddr]+=p0->w;
	  for(int i=0;i<gcfg->maxmedia;i++)
		n_det[baseaddr+i]+=ppath[i]*p0->w;
      }
}

/**
 * @brief Rotate Stokes vector s by phi
 * 
 * This function represents a rotation in the clockwise direction with respect
 * to an observer looking into the direction of the photon propagation.
 * 
 * @param[in] s: input Stokes parameter
 * @param[in] phi: rotation angle in radians
 * @param[out] s2: output Stokes parameter
 */

__device__ inline void rotsphi(Stokes *s, float phi, Stokes *s2){
    float sin2phi, cos2phi;
    sincosf(2.f*phi,&sin2phi,&cos2phi);
    
    s2->i = s->i;
    s2->q = s->q*cos2phi + s->u*sin2phi;
    s2->u = -s->q*sin2phi + s->u*cos2phi;
    s2->v = s->v;
}

/**
 * @brief Update Stokes vector after a scattering event
 * @param[in,out] s: input and output Stokes vector
 * @param[in] theta: scattering angle in radiance
 * @param[in] phi: azimuthal angle in radiance
 * @param[in] u: incident direction cosine
 * @param[in] u2: scattering direction cosine
 * @param[in] prop: pointer to the current optical properties
 */

__device__ inline void updatestokes(Stokes *s, float theta, float phi, float3 *u, float3 *u2, uint *mediaid, float4 *gsmatrix){
    float costheta = cosf(theta);
    Stokes s2;
    rotsphi(s,phi,&s2);
    
    uint imedia=NANGLES*((*mediaid & MED_MASK)-1);
    uint ithedeg=floorf(theta*NANGLES*(R_PI-EPS));
    
    s->i= gsmatrix[imedia+ithedeg].x*s2.i+gsmatrix[imedia+ithedeg].y*s2.q;
    s->q= gsmatrix[imedia+ithedeg].y*s2.i+gsmatrix[imedia+ithedeg].x*s2.q;
    s->u= gsmatrix[imedia+ithedeg].z*s2.u+gsmatrix[imedia+ithedeg].w*s2.v;
    s->v=-gsmatrix[imedia+ithedeg].w*s2.u+gsmatrix[imedia+ithedeg].z*s2.v;
    
    float temp,sini,cosi,sin22,cos22;
    
    temp=(u2->z>-1.f && u2->z<1.f) ? rsqrtf((1.f-costheta*costheta)*(1.f-u2->z*u2->z)) : 0.f;
    
    cosi=(temp==0.f) ? 0.f :(((phi>ONE_PI && phi<TWO_PI) ? 1.f : -1.f)*(u2->z*costheta-u->z)*temp);
    cosi=fmax(-1.f,fmin(cosi,1.f));

    sini=sqrtf(1.f-cosi*cosi);
    cos22=2.f*cosi*cosi-1.f;
    sin22=2.f*sini*cosi;
    
    s2.i=s->i;
    s2.q=s->q*cos22-s->u*sin22;
    s2.u=s->q*sin22+s->u*cos22;
    s2.v=s->v;
    
    temp=__fdividef(1.f,s2.i); 
    s->q=s2.q*temp;
    s->u=s2.u*temp;
    s->v=s2.v*temp;
    s->i=1.f;
}

/**
 * @brief Recording detected photon information at photon termination
 * @param[in] n_det: pointer to the detector position array
 * @param[in] detectedphoton: variable in the global-mem recording the total detected photons
 * @param[in] ppath: buffer in the shared-mem to store the photon partial-pathlengths
 * @param[in] p0: the position/weight of the current photon packet
 * @param[in] v: the direction vector of the current photon packet
 * @param[in] t: random number generator (RNG) states
 * @param[in] seeddata: the RNG seed of the photon at launch, need to save for replay 
 */

__device__ inline void savedetphoton(float n_det[],uint *detectedphoton,float *ppath,MCXpos *p0,MCXdir *v,Stokes *s,RandType t[RAND_BUF_LEN],RandType *seeddata,uint isdet){
      int detid;
      detid=(isdet==OUTSIDE_VOLUME_MIN)?-1:(int)finddetector(p0);
      if(detid){
	 uint baseaddr=atomicAdd(detectedphoton,1);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    for(i=0;i<gcfg->issaveseed*RAND_BUF_LEN;i++)
	        seeddata[baseaddr*RAND_BUF_LEN+i]=t[i]; //< save photon seed for replay
	    baseaddr*=gcfg->reclen;
	    if(SAVE_DETID(gcfg->savedetflag))
	        n_det[baseaddr++]=detid;
	    for(i=0;i<gcfg->partialdata;i++)
		n_det[baseaddr++]=ppath[i]; //< save partial pathlength to the memory
            if(SAVE_PEXIT(gcfg->savedetflag)){
	            *((float3*)(n_det+baseaddr))=float3(p0->x,p0->y,p0->z);
		    baseaddr+=3;
	    }
	    if(SAVE_VEXIT(gcfg->savedetflag)){
		    *((float3*)(n_det+baseaddr))=float3(v->x,v->y,v->z);
		    baseaddr+=3;
	    }
	    if(SAVE_W0(gcfg->savedetflag))
	        n_det[baseaddr++]=ppath[gcfg->w0offset-1];
            if(SAVE_IQUV(gcfg->savedetflag)){
                n_det[baseaddr++]=s->i;
                n_det[baseaddr++]=s->q;
                n_det[baseaddr++]=s->u;
                n_det[baseaddr++]=s->v;
            }
	 }
      }
}
#endif

/**
 * @brief Saving photon trajectory data for debugging purposes
 * @param[in] p: the position/weight of the current photon packet
 * @param[in] id: the global index of the photon
 * @param[in] gdebugdata: pointer to the global-memory buffer to store the trajectory info
 */

__device__ inline void savedebugdata(MCXpos *p,uint id,float *gdebugdata){
      uint pos=atomicAdd(gjumpdebug,1);
      if(pos<gcfg->maxjumpdebug){
         pos*=MCX_DEBUG_REC_LEN;
         ((uint *)gdebugdata)[pos++]=id;
         gdebugdata[pos++]=p->x;
         gdebugdata[pos++]=p->y;
         gdebugdata[pos++]=p->z;
         gdebugdata[pos++]=p->w;
         gdebugdata[pos++]=0;
      }
}

/**
 * @brief A simplified nextafterf() to ensure a photon moves outside of the current voxel after each move
 * @param[in] a: a floating point number
 * @param[in] dir: 1: change 1 bit in the positive direction; 0: no change, -1: change 1 bit in the negative direction
 */

__device__ inline float mcx_nextafterf(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a+gcfg->maxvoidstep;         /** First, shift coordinate by 1000 to make sure values are always positive */
      num.i+=dir ^ (num.i & SIGN_BIT);/** Then make 1 bit difference along the direction indicated by dir */
      return num.f-gcfg->maxvoidstep;    /** Last, undo the offset, and return */
}

#ifndef USE_HALF

/**
 * @brief Core function for photon-voxel ray-tracing 
 *
 * This is the heart of the MCX simulation algorithm. It calculates the nearest intersection
 * of the ray inside the current cubic voxel.
 *
 * @param[in] p0: the x/y/z position of the current photon
 * @param[in] v: the direction vector of the photon
 * @param[out] htime: the intersection x/y/z position on the bounding box, right outside of this voxel
 * @param[in] rv: pre-computed reciprocal of the velocity vector (v)
 * @param[out] id: 0: intersect with x=x0 plane; 1: intersect with y=y0 plane; 2: intersect with z=z0 plane first
 * @return the distance to the intersection to the voxel bounding box
 */

__device__ inline float hitgrid(float3 *p0, float3 *v, float *htime,float* rv,int *id){
      float dist;

      //< time-of-flight to hit the wall in each direction
      htime[0]=fabs((floorf(p0->x)+(v->x>0.f)-p0->x)*rv[0]); //< time-of-flight in x
      htime[1]=fabs((floorf(p0->y)+(v->y>0.f)-p0->y)*rv[1]);
      htime[2]=fabs((floorf(p0->z)+(v->z>0.f)-p0->z)*rv[2]);

      //< get the direction with the smallest time-of-flight
      dist=fminf(fminf(htime[0],htime[1]),htime[2]);
      (*id)=(dist==htime[0]?0:(dist==htime[1]?1:2));

      //< p0 is inside, htime is the 1st intersection point
      htime[0]=p0->x+dist*v->x;
      htime[1]=p0->y+dist*v->y;
      htime[2]=p0->z+dist*v->z;

      //< make sure the intersection point htime is immediately outside of the current voxel (i.e. not within the current voxel)
      int index = (*id & (int)3); 

      if(index == 0) htime[0] = mcx_nextafterf(roundf(htime[0]), (v->x > 0.f)-(v->x < 0.f));
      if(index == 1) htime[1] = mcx_nextafterf(roundf(htime[1]), (v->y > 0.f)-(v->y < 0.f));
      if(index == 2) htime[2] = mcx_nextafterf(roundf(htime[2]), (v->z > 0.f)-(v->z < 0.f));

      return dist;
}

#else

/**
 * @brief Half-precision version of the simplified nextafter
 *
 * @param[in] a: a half-precision floating point number
 * @param[in] dir: 1: change 1 bit in the positive direction; 0: no change, -1: change 1 bit in the negative direction
 */
 
__device__ inline half mcx_nextafter_half(const half a, const short dir){
      union{
#if ! defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ >= 9
          __half_raw f;
#else
          half f;
#endif
          short i;
      } num;
      num.f=a;
      ((num.i & 0x7FFFU) == 0) ? (num.i = ((dir & 0x8000U) ) | 1) : ((num.i & 0x8000U) ? num.i-= dir: num.i+= dir);
      return num.f;
}

/**
 * @brief Core function for photon-voxel ray-tracing (half-precision version)
 *
 * This is the heart of the MCX simulation algorithm. It calculates the nearest intersection
 * of the ray inside the current cubic voxel.
 *
 * @param[in] p0: the x/y/z position of the current photon
 * @param[in] v: the direction vector of the photon
 * @param[out] htime: the intersection x/y/z position on the bounding box, right outside of this voxel
 * @param[in] rv: pre-computed reciprocal of the velocity vector (v)
 * @param[out] id: 0: intersect with x=x0 plane; 1: intersect with y=y0 plane; 2: intersect with z=z0 plane first
 * @return the distance to the intersection to the voxel bounding box
 */

__device__ inline float hitgrid(float3 *p0, float3 *v, float *htime,float* rv,int *id){
      float dist;

      union {
           unsigned int i;
           float f;
#if ! defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ >= 9
          __half2_raw h2;
          __half_raw h[2];
#else
           half2 h2;
           half h[2];
#endif
      } pxy, pzw, vxy, vzw, h1, h2, temp;

      pxy.h2=__floats2half2_rn(floorf(p0->x) - p0->x, floorf(p0->y) - p0->y);
      pzw.h2=__floats2half2_rn(floorf(p0->z) - p0->z, 1e5f);
      vxy.h2=__floats2half2_rn(rv[0],rv[1]);
      vzw.h2=__floats2half2_rn(rv[2],1.f);

      temp.h2 = __floats2half2_rn(0.f, 0.f);

      h1.h2 = __hmul2(__hadd2(pxy.h2,__hgt2(vxy.h2, temp.h2 )), vxy.h2);
      h2.h2 = __hmul2(__hadd2(pzw.h2,__hgt2(vzw.h2, temp.h2 )), vzw.h2);

      // abs
      h1.i &= 0x7FFF7FFF;
      h2.i &= 0x7FFF7FFF;

      temp.h[0]=(__hlt(h1.h[0], h1.h[1]))   ? (*id=0,h1.h[0])  : (*id=1,h1.h[1]);
      temp.h[1]=(__hlt(temp.h[0], h2.h[0])) ?    temp.h[0]     : (*id=2,h2.h[0]);

      dist=__half2float(temp.h[1]);

      //p0 is inside, p is outside, move to the 1st intersection pt, now in the air side, to be corrected in the else block
      vxy.h2=__floats2half2_rn(v->x,v->y);
      vzw.h2=__floats2half2_rn(v->z,0.f);

      pxy.h2=__floats2half2_rn(p0->x, p0->y);
      pzw.h2=__floats2half2_rn(p0->z, 0.f);

      h1.h2 =__hfma2(vxy.h2,__floats2half2_rn(dist,dist),pxy.h2);
      h2.h2 =__hfma2(vzw.h2,__floats2half2_rn(dist,dist),pzw.h2);
      htime[0]=__half2float(h1.h[0]);
      htime[1]=__half2float(h1.h[1]);
      htime[2]=__half2float(h2.h[0]);

      temp.h2 = __floats2half2_rn(0.f, 0.f);
      pxy.h2=__hgt2(vxy.h2, temp.h2 );
      pzw.h2=__hlt2(vxy.h2, temp.h2 );
      pxy.h2=__hsub2(pxy.h2, pzw.h2 );

      pzw.h2=__hlt2(vzw.h2, temp.h2 );
      temp.h2=__hgt2(vzw.h2, temp.h2 );
      pzw.h2=__hsub2(temp.h2,pzw.h2 );

      if((*id) == 0) htime[0] = __half2float(mcx_nextafter_half(hrint(h1.h[0]), __half2short_rn(pxy.h[0])));
      if((*id) == 1) htime[1] = __half2float(mcx_nextafter_half(hrint(h1.h[1]), __half2short_rn(pxy.h[1])));
      if((*id) == 2) htime[2] = __half2float(mcx_nextafter_half(hrint(h2.h[0]), __half2short_rn(pzw.h[0])));

      return dist;
}

#endif

/**
 * @brief Calculating the direction vector after transmission
 *
 * This function updates the direction vector after the photon passing
 * an interface of different refrective indicex (n1/n2). Because MCX only
 * handles voxelated domain, transmission is applied only to 1 of the components,
 * and then the vector is normalized.
 *
 * @param[in,out] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 */
 
__device__ inline void transmit(MCXdir *v, float n1, float n2,int flipdir){
      float tmp0=n1/n2;
      v->x*=tmp0;
      v->y*=tmp0;
      v->z*=tmp0;
      (flipdir==0) ?
          (v->x= ((tmp0 = v->y*v->y + v->z*v->z) <1.f) ? sqrtf(1.f - tmp0)*((v->x>0.f)-(v->x<0.f)) : 0.f):
	  ((flipdir==1) ? 
	      (v->y=((tmp0 = v->x*v->x + v->z*v->z) <1.f) ? sqrtf(1.f - tmp0)*((v->y>0.f)-(v->y<0.f)) : 0.f):
	      (v->z=((tmp0 = v->x*v->x + v->y*v->y) <1.f) ? sqrtf(1.f - tmp0)*((v->z>0.f)-(v->z<0.f)) : 0.f));
      tmp0=rsqrtf(v->x*v->x + v->y*v->y + v->z*v->z);
      v->x*=tmp0;
      v->y*=tmp0;
      v->z*=tmp0;
}

/**
 * @brief Calculating the reflection coefficient at an interface
 *
 * This function calculates the reflection coefficient at
 * an interface of different refrective indicex (n1/n2)
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

__device__ inline float reflectcoeff(MCXdir *v, float n1, float n2, int flipdir){
      float Icos=fabs((flipdir==0) ? v->x : (flipdir==1 ? v->y : v->z));
      float tmp0=n1*n1;
      float tmp1=n2*n2;
      float tmp2=1.f-tmp0/tmp1*(1.f-Icos*Icos); /** 1-[n1/n2*sin(si)]^2 = cos(ti)^2*/
      if(tmp2>0.f){ //< partial reflection
          float Re,Im,Rtotal;
	  Re=tmp0*Icos*Icos+tmp1*tmp2;
	  tmp2=sqrtf(tmp2); /** to save one sqrt*/
	  Im=2.f*n1*n2*Icos*tmp2;
	  Rtotal=(Re-Im)/(Re+Im);     /** Rp*/
	  Re=tmp1*Icos*Icos+tmp0*tmp2*tmp2;
	  Rtotal=(Rtotal+(Re-Im)/(Re+Im))*0.5f; /** (Rp+Rs)/2*/
	  return Rtotal;
      }else{ //< total reflection
          return 1.f;
      }
}

/**
 * @brief Loading optical properties from constant memory
 *
 * This function parses the media input and load optical properties
 * from GPU memory
 *
 * @param[out] prop: pointer to the current optical properties {mua, mus, g, n}
 * @param[in] mediaid: the media ID (32 bit) of the current voxel, format is specified in gcfg->mediaformat or cfg->mediabyte
 */

template <const int islabel, const int issvmc>
__device__ void updateproperty(Medium *prop, unsigned int& mediaid, RandType t[RAND_BUF_LEN], unsigned int idx1d, 
                               uint media[], float3 *p, MCXsp *nuvox){
          /**
	   * The default mcx input volume is assumed to be 4-byte per voxel
	   * (SVMC mode requires 2x 4-byte voxels for 8 data points) 
	   *
	   * The data encoded in the voxel are parsed based on the gcfg->mediaformat flag.
	   * Below, we use [s*] to represent 2-byte short integers; [h*] to represent a 
	   * 2-byte half-precision floating point number; [c*] to represent 
	   * 1-byte unsigned char integers and [i*] to represent 4-byte integers;
	   * [f*] for 4-byte floating point number
	   * index 0 starts from the lowest (least significant bit) end
	   */
	  if(islabel){ //< [i0]: traditional MCX input type - voxels store integer labels, islabel is a template const for speed
	      *((float4*)(prop))=gproperty[mediaid & MED_MASK];
	  }else if(gcfg->mediaformat==MEDIA_LABEL_HALF){ //< [h1][s0]: h1: half-prec property value; highest 2bit in s0: index 0-3, low 14bit: tissue label
	      union{
	         unsigned int i;
#if ! defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ >= 9
                 __half_raw h[2];
#else
                 half h[2];
#endif
		 unsigned short s[2]; /**s[1]: half-prec property; s[0]: high 2bits: idx 0-3, low 14bits: tissue label*/
	      } val;
	      val.i=mediaid & MED_MASK;
	      *((float4*)(prop))=gproperty[val.s[0] & 0x3FFF];
              float *p=(float*)(prop);
	      p[(val.s[0] & 0xC000)>>14]=fabs(__half2float(val.h[1]));
          }else if(gcfg->mediaformat==MEDIA_MUA_FLOAT){ //< [f0]: single-prec mua every voxel; mus/g/n uses 2nd row in gcfg.prop
	      prop->mua=fabs(*((float *)&mediaid));
              prop->n=gproperty[!(mediaid & MED_MASK)==0].w;
	  }else if(gcfg->mediaformat==MEDIA_AS_F2H||gcfg->mediaformat==MEDIA_AS_HALF){ //< [h1][h0]: h1/h0: single-prec mua/mus for every voxel; g/n uses those in cfg.prop(2,:)
	      union {
                 unsigned int i;
#if ! defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ >= 9
                 __half_raw h[2];
#else
                 half h[2];
#endif
              } val;
	      val.i=mediaid & MED_MASK;
	      prop->mua=fabs(__half2float(val.h[0]));
	      prop->mus=fabs(__half2float(val.h[1]));
	      prop->n=gproperty[!(mediaid & MED_MASK)==0].w;
	  }else if(gcfg->mediaformat==MEDIA_2LABEL_MIX){ //< [s1][c1][c0]: s1: (volume fraction of tissue 1)*(2^16-1), c1: tissue 1 label, c0: tissue 0 label
	      union {
                 unsigned int   i;
                 unsigned short h[2];
		 unsigned char  c[4];
              } val;
	      val.i=mediaid & MED_MASK;
	      if(val.h[1]>0){
                  if((rand_uniform01(t)*32767.f)<val.h[1]){
	              *((float4*)(prop))=gproperty[val.c[1]];
                      mediaid>>=8;
	          }else
	              *((float4*)(prop))=gproperty[val.c[0]];
                  mediaid &= 0xFFFF;
              }else
                  *((float4*)(prop))=gproperty[val.c[0]];
	  }else if(gcfg->mediaformat==MEDIA_ASGN_BYTE){//< [c3][c2][c1][c0]: c0/c1/c2/c3: interpolation ratios (scaled to 0-255) of mua/mus/g/n between cfg.prop(1,:) and cfg.prop(2,:)
	      union {
                 unsigned int i;
                 unsigned char h[4];
              } val;
	      val.i=mediaid & MED_MASK;
	      prop->mua=val.h[0]*(1.f/255.f)*(gproperty[2].x-gproperty[1].x)+gproperty[1].x;
	      prop->mus=val.h[1]*(1.f/255.f)*(gproperty[2].y-gproperty[1].y)+gproperty[1].y;
	      prop->g  =val.h[2]*(1.f/255.f)*(gproperty[2].z-gproperty[1].z)+gproperty[1].z;
	      prop->n  =val.h[3]*(1.f/127.f)*(gproperty[2].w-gproperty[1].w)+gproperty[1].w;
          }else if(gcfg->mediaformat==MEDIA_AS_SHORT){//< [s1][s0]: s0/s1: interpolation ratios (scaled to 0-65535) of mua/mus between cfg.prop(1,:) and cfg.prop(2,:)
	      union {
                 unsigned int i;
                 unsigned short h[2];
              } val;
	      val.i=mediaid & MED_MASK;
	      prop->mua=val.h[0]*(1.f/65535.f)*(gproperty[2].x-gproperty[1].x)+gproperty[1].x;
	      prop->mus=val.h[1]*(1.f/65535.f)*(gproperty[2].y-gproperty[1].y)+gproperty[1].y;
	      prop->n=gproperty[!(mediaid & MED_MASK)==0].w;
          }else if(issvmc){ //< SVMC mode [c7][c6][c5][c4] and [c3][c2][c1][c0] stored as two 4-byte records;
	      if(idx1d==OUTSIDE_VOLUME_MIN || idx1d==OUTSIDE_VOLUME_MAX){
	          *((float4*)(prop))=gproperty[0]; // out-of-bounds
		  return;
	      }
	      union {
	          unsigned char c[8];
		  unsigned int  i[2];
	      } val; // c[7-6]: lower & upper label, c[5-3]: reference point, c[2-0]: normal vector
	      val.i[0]=media[idx1d+gcfg->dimlen.z];
	      val.i[1]=mediaid & MED_MASK;
	      nuvox->sv.lower=val.c[7];
	      nuvox->sv.upper=val.c[6];
	      if(val.c[6]){ // if upper label is not zero, the photon is inside a mixed voxel
	          /** Extract the reference point of the intra-voxel interface*/
		  nuvox->rp=float3(val.c[5]*(1.f/255.f),val.c[4]*(1.f/255.f),val.c[3]*(1.f/255.f));
		  (nuvox->rp)+=float3(floorf(p->x),floorf(p->y),floorf(p->z));
		  
		  /** Extract the normal vector of the intra-voxel interface*/
		  nuvox->nv=float3(val.c[2]*(2.f/255.f)-1,val.c[1]*(2.f/255.f)-1,val.c[0]*(2.f/255.f)-1);
		  nuvox->nv=nuvox->nv*rsqrtf(dot(nuvox->nv,nuvox->nv));
		  
		  /** Determine tissue label corresponding to the current photon position*/
		  if(dot(nuvox->rp-*p,nuvox->nv)<0){
		      *((float4*)(prop))=gproperty[nuvox->sv.upper]; // upper label
		      nuvox->sv.isupper=1;
		      nuvox->nv=-nuvox->nv; // normal vector always points to the other side (outward-pointing)
		  }else{
		      *((float4*)(prop))=gproperty[nuvox->sv.lower]; // lower label
		      nuvox->sv.isupper=0;
		  }
		  nuvox->sv.issplit=1;
	      }else{ // if upper label is zero, the photon is inside a regular voxel
	          *((float4*)(prop))=gproperty[val.c[7]]; // voxel uniquely labeled
		  nuvox->sv.issplit=0;
		  nuvox->sv.isupper=0;
	      }
	  }
}

/**
 * @brief Compute intersection point between a photon path and the intra-voxel interface if present
 *
 * This function tests if a ray intersects with an in-voxel marching-cube boundary (a plane) in SVMC mode
 *
 * @param[in] p0: current position
 * @param[in] v: current photon direction
 * @param[in] prop: optical properties
 * @param[in,out] len: photon movement length, updated if intersection is found
 * @param[in,out] slen: remaining unitless scattering length, updated if intersection is found
 * @param[in] nuvox: a struct storing normal direction (nv) and a point on the plane
 * @param[in] f: photon state including total time-of-flight, number of scattering etc
 * @param[in] htime: nearest intersection of the enclosing voxel, returned by hitgrid
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 */

__device__ int ray_plane_intersect(float3 *p0, MCXdir *v, Medium *prop, float &len, float &slen, 
                                   MCXsp *nuvox, MCXtime f, float3 &htime, int &flipdir){
	
	if(dot(*(float3*)v,nuvox->nv)<=0){ // no intersection, as nv always points to the other side
	    return 0;
	}else{
	    float3 p1=(gcfg->faststep || slen==f.pscat) ? (*p0+len*(*(float3*)v)) : float3(flipdir==0 ? roundf(htime.x) : htime.x,
                flipdir==1 ? roundf(htime.y) : htime.y, flipdir==2 ? roundf(htime.z) : htime.z);
	    float3 rp0=*p0-nuvox->rp;
	    float3 rp1=p1-nuvox->rp;
	    float d0=dot(rp0,nuvox->nv); // signed perpendicular distance from p0 to patch
	    float d1=dot(rp1,nuvox->nv); // signed perpendicular distance from p1 to patch
	    if(d0*d1>0.f){ // p0 and p1 are on the same side, no interection
		return 0;
	    }else{
	        float len0=len*d0/(d0-d1);
		len=(len0 > 0) ? len0 : len;
		slen=len*prop->mus*(v->nscat+1.f > gcfg->gscatter ? (1.f-prop->g) : 1.f);
		return 1;
	    }
	}
}

/**
 * @brief Perform reflection/refraction computation along mismatched intra-voxel interface
 *
 * This function returns a new direction when a ray is reflected/transmitted
 * through an in-voxel marching-cube boundary in the SVMC mode
 *
 * @param[in] n1: refractive index of the current subvolume in a split-voxel
 * @param[in,out] c0: current photon direction
 * @param[out] rv: reciprocated direction vector rv={1/c0.x,1/c0.y,1/c0.z}
 * @param[in] prop: optical properties
 * @param[in] nuvox: a struct storing normal direction (nv) and a point on the plane
 * @param[in] t: random number state for deciding reflection/transmission
 */

__device__ int reflectray(float n1, float3 *c0, float3 *rv, MCXsp *nuvox, Medium *prop, RandType t[RAND_BUF_LEN]){
	/*to handle refractive index mismatch*/
	float Icos,Re,Im,Rtotal,tmp0,tmp1,tmp2,n2;
	
	Icos=fabs(dot(*c0,nuvox->nv));
	
	n2=(nuvox->sv.isupper)? gproperty[nuvox->sv.upper].w : gproperty[nuvox->sv.lower].w;
		
	tmp0=n1*n1;
	tmp1=n2*n2;
	tmp2=1.f-tmp0/tmp1*(1.f-Icos*Icos); /*1-[n1/n2*sin(si)]^2 = cos(ti)^2*/	
	
	if(tmp2>0.f){ /*if no total internal reflection*/
	    Re=tmp0*Icos*Icos+tmp1*tmp2;      /*transmission angle*/
	    tmp2=sqrtf(tmp2); /*to save one sqrt*/
	    Im=2.f*n1*n2*Icos*tmp2;
	    Rtotal=(Re-Im)/(Re+Im);     /*Rp*/
	    Re=tmp1*Icos*Icos+tmp0*tmp2*tmp2;
	    Rtotal=(Rtotal+(Re-Im)/(Re+Im))*0.5f; /*(Rp+Rs)/2*/
	    if(rand_next_reflect(t)<=Rtotal){ /*do reflection*/
	        *c0+=(FL3(-2.f*Icos))*nuvox->nv;
		nuvox->sv.isupper=!nuvox->sv.isupper;
	    }else{   /*do transmission*/
	        *c0+=(FL3(-Icos))*nuvox->nv;
		*c0=(FL3(tmp2))*nuvox->nv+FL3(n1/n2)*(*c0);
		nuvox->nv=-nuvox->nv;
		if(((nuvox->sv.isupper)? nuvox->sv.isupper:nuvox->sv.lower)==0) /*transmit to background medium*/
	            return 1;
		*((float4*)prop)=gproperty[nuvox->sv.isupper ? nuvox->sv.upper:nuvox->sv.lower];
	    }
	}else{ /*total internal reflection*/
	    *c0+=(FL3(-2.f*Icos))*nuvox->nv;
	    nuvox->sv.isupper=!nuvox->sv.isupper;
	}
	tmp0=rsqrtf(dot(*c0,*c0));
	(*c0)=(*c0)*FL3(tmp0);
	(*rv)=float3(__fdividef(1.f,c0->x),__fdividef(1.f,c0->y),__fdividef(1.f,c0->z));
	return 0;
}

/**
 * @brief Loading optical properties from constant memory
 *
 * This function parses the media input and load optical properties
 * from GPU memory
 *
 * @param[in] mediaid: the media ID (32 bit) of the current voxel, format is specified in gcfg->mediaformat or cfg->mediabyte
 */
 
__device__ float getrefractiveidx(unsigned int mediaid){
          if((mediaid& MED_MASK)==0)
	      return gproperty[0].w;
          if(gcfg->mediaformat<=4)
	      return gproperty[mediaid & MED_MASK].w;
	  else if(gcfg->mediaformat==MEDIA_ASGN_BYTE)
	      return 0.9f;
          else
	      return gproperty[1].w;
}

/**
 * @brief Advance photon to the 1st non-zero voxel if launched in the backgruond 
 *
 * This function advances the photon to the 1st non-zero voxel along the direction
 * of v if the photon is launched outside of the cubic domain or in a zero-voxel.
 * To avoid large overhead, photon can only advance gcfg->minaccumtime steps, which
 * can be set using the --maxvoidstep flag; by default, this limit is 1000.
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

template <const int islabel, const int issvmc>
__device__ inline int skipvoid(MCXpos *p,MCXdir *v,MCXtime *f,float3* rv,uint media[],RandType t[RAND_BUF_LEN],
                               MCXsp *nuvox){
      int count=1,idx1d;
      while(1){
          if(p->x>=0.f && p->y>=0.f && p->z>=0.f && p->x < gcfg->maxidx.x
               && p->y < gcfg->maxidx.y && p->z < gcfg->maxidx.z){
	    idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	    if(media[idx1d] & MED_MASK){ //< if enters a non-zero voxel
                GPUDEBUG(("inside volume [%f %f %f] v=<%f %f %f>\n",p->x,p->y,p->z,v->x,v->y,v->z));
	        float4 htime;
                int flipdir;
                p->x-=v->x;
                p->y-=v->y;
                p->z-=v->z;
                f->t-=gcfg->minaccumtime;
                idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));

                GPUDEBUG(("look for entry p0=[%f %f %f] rv=[%f %f %f]\n",p->x,p->y,p->z,rv->x,rv->y,rv->z));
		count=0;
		while(!(p->x>=0.f && p->y>=0.f && p->z>=0.f && p->x < gcfg->maxidx.x
                  && p->y < gcfg->maxidx.y && p->z < gcfg->maxidx.z) || !(media[idx1d] & MED_MASK)){ // at most 3 times
	            f->t+=gcfg->minaccumtime*hitgrid((float3*)p,(float3*)v,&htime.x,&rv->x,&flipdir);
                    *((float4*)(p))=float4(htime.x,htime.y,htime.z,p->w);
                    idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
                    GPUDEBUG(("entry p=[%f %f %f] flipdir=%d\n",p->x,p->y,p->z,flipdir));

		    if(count++>3){
		       GPUDEBUG(("fail to find entry point after 3 iterations, something is wrong, abort!!"));
		       break;
		    }
		}
                f->t= (gcfg->voidtime) ? f->t : 0.f;
                updateproperty<islabel, issvmc>((Medium *)&htime,media[idx1d],t,idx1d,media,(float3*)p,nuvox);
		if(gcfg->isspecular && htime.w!=gproperty[0].w){
	            p->w*=1.f-reflectcoeff(v, gproperty[0].w,htime.w,flipdir);
                    GPUDEBUG(("transmitted intensity w=%e\n",p->w));
	            if(p->w>EPS){
		        transmit(v, gproperty[0].w,htime.w,flipdir);
                        GPUDEBUG(("transmit into volume v=<%f %f %f>\n",v->x,v->y,v->z));
                    }
		}
		GPUDEBUG(("entry from voxel [%d]\n",idx1d));
		return idx1d;
	    }
          }
	  if( (p->x<0.f) && (v->x<=0.f) || (p->x >= gcfg->maxidx.x) && (v->x>=0.f)
	   || (p->y<0.f) && (v->y<=0.f) || (p->y >= gcfg->maxidx.y) && (v->y>=0.f)
	   || (p->z<0.f) && (v->z<=0.f) || (p->z >= gcfg->maxidx.z) && (v->z>=0.f))
	      return -1;
	  *((float4*)(p))=float4(p->x+v->x,p->y+v->y,p->z+v->z,p->w);
          GPUDEBUG(("inside void [%f %f %f]\n",p->x,p->y,p->z));
          f->t+=gcfg->minaccumtime;
	  if(count++>gcfg->maxvoidstep)
	      return -1;
      }
}

/**
 * @brief Compute 2D-scattering if the domain has a dimension of 1 in x/y or z
 *
 * This function performs 2D scattering calculation if the domain is only a sheet of voxels
 *
 * @param[in,out] v: the direction vector of the photon
 * @param[in] stheta: the sine of the rotation angle
 * @param[in] ctheta: the cosine of the rotation angle
 */

__device__ inline void rotatevector2d(MCXdir *v, float stheta, float ctheta){
      if(gcfg->is2d==1)
   	  *((float4*)v)=float4(
   	       0.f,
	       v->y*ctheta - v->z*stheta,
   	       v->y*stheta + v->z*ctheta,
   	       v->nscat
   	  );
      else if(gcfg->is2d==2)
  	  *((float4*)v)=float4(
	       v->x*ctheta - v->z*stheta,
   	       0.f,
   	       v->x*stheta + v->z*ctheta,
   	       v->nscat
   	  );
      else if(gcfg->is2d==3)
  	  *((float4*)v)=float4(
	       v->x*ctheta - v->y*stheta,
   	       v->x*stheta + v->y*ctheta,
   	       0.f,
   	       v->nscat
   	  );
      GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
}

/**
 * @brief Compute 3D-scattering direction vector
 *
 * This function updates the direction vector after a 3D scattering event
 *
 * @param[in,out] v: the direction vector of the photon
 * @param[in] stheta: the sine of the azimuthal angle
 * @param[in] ctheta: the cosine of the azimuthal angle
 * @param[in] sphi: the sine of the zenith angle
 * @param[in] cphi: the cosine of the zenith angle
 */

__device__ inline void rotatevector(MCXdir *v, float stheta, float ctheta, float sphi, float cphi){
      if( v->z>-1.f+EPS && v->z<1.f-EPS ) {
   	  float tmp0=1.f-v->z*v->z;
   	  float tmp1=stheta*rsqrtf(tmp0);
   	  *((float4*)v)=float4(
   	       tmp1*(v->x*v->z*cphi - v->y*sphi) + v->x*ctheta,
   	       tmp1*(v->y*v->z*cphi + v->x*sphi) + v->y*ctheta,
   	      -tmp1*tmp0*cphi                    + v->z*ctheta,
   	       v->nscat
   	  );
      }else{
   	  *((float4*)v)=float4(stheta*cphi,stheta*sphi,(v->z>0.f)?ctheta:-ctheta,v->nscat);
      }
      GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
}

/**
 * @brief Terminate a photon and launch a new photon according to specified source form
 *
 * This function terminates the current photon and launches a new photon according 
 * to the source type selected. Currently, over a dozen source types are supported, 
 * including pencil, isotropic, planar, fourierx, gaussian, zgaussian etc.
 *
 * @param[in,out] p: the 3D position and weight of the photon
 * @param[in,out] v: the direction vector of the photon
 * @param[in,out] f: the parameter vector of the photon
 * @param[in,out] s: the Stokes vector of the photon
 * @param[in,out] rv: the reciprocal direction vector of the photon (rv[i]=1/v[i])
 * @param[out] prop: the optical properties of the voxel the photon is launched into
 * @param[in,out] idx1d: the linear index of the voxel containing the photon at launch
 * @param[in] field: the 3D array to store photon weights
 * @param[in,out] mediaid: the medium index at the voxel at launch
 * @param[in,out] w0: initial weight, reset here after launch
 * @param[in] isdet: whether the previous photon being terminated lands at a detector
 * @param[in,out] ppath: pointer to the shared-mem buffer to store photon partial-path data
 * @param[in,out] n_det: array in the constant memory where detector positions are stored
 * @param[in,out] dpnum: global-mem variable where the count of detected photons are stored
 * @param[in] t: RNG state
 * @param[in,out] photonseed: RNG state stored at photon's launch time if replay is needed
 * @param[in] media: domain medium index array, read-only
 * @param[in] srcpattern: user-specified source pattern array if pattern source is used
 * @param[in] threadid: the global index of the current thread
 * @param[in] rngseed: in the replay mode, pointer to the saved detected photon seed data
 * @param[in,out] seeddata: pointer to the buffer to save detected photon seeds
 * @param[in,out] gdebugdata: pointer to the buffer to save photon trajectory positions
 * @param[in,out] gprogress: pointer to the host variable to update progress bar
 */

template <const int ispencil, const int isreflect, const int islabel, const int issvmc>
__device__ inline int launchnewphoton(MCXpos *p,MCXdir *v,Stokes *s,MCXtime *f,float3* rv,Medium *prop,uint *idx1d, OutputType *field,
           uint *mediaid,OutputType *w0,uint isdet, float ppath[],float n_det[],uint *dpnum,
	   RandType t[RAND_BUF_LEN],RandType photonseed[RAND_BUF_LEN],
	   uint media[],float srcpattern[],int threadid,RandType rngseed[],RandType seeddata[],float gdebugdata[],volatile int gprogress[],
	   float photontof[],MCXsp *nuvox){
      *w0=1.f;     //< reuse to count for launchattempt
      int canfocus=1; //< non-zero: focusable, zero: not focusable

      /**
       * First, let's terminate the current photon and perform detection calculations
       */
      if(p->w>=0.f){
          ppath[gcfg->partialdata]+=p->w;  //< sum all the remaining energy

          if(gcfg->debuglevel & MCX_DEBUG_MOVE)
              savedebugdata(p,((uint)f->ndone)+threadid*gcfg->threadphoton+umin(threadid,(threadid<gcfg->oddphotons)*threadid),gdebugdata);

          if(*mediaid==0 && *idx1d!=OUTSIDE_VOLUME_MIN && *idx1d!=OUTSIDE_VOLUME_MAX && gcfg->issaveref){
            if(gcfg->issaveref==1){
	      int tshift=MIN(gcfg->maxgate-1,(int)(floorf((f->t-gcfg->twin0)*gcfg->Rtstep)));
	      if(gcfg->srctype!=MCX_SRC_PATTERN && gcfg->srctype!=MCX_SRC_PATTERN3D){
#ifdef USE_ATOMIC
                  atomicAdd(& field[*idx1d+tshift*gcfg->dimlen.z],-p->w);	       
#else
	          field[*idx1d+tshift*gcfg->dimlen.z]+=-p->w;
#endif
	      }else{
		  for(int i=0;i<gcfg->srcnum;i++){
		    if(ppath[gcfg->w0offset+i]>0.f){
#ifdef USE_ATOMIC
                        atomicAdd(& field[(*idx1d+tshift*gcfg->dimlen.z)*gcfg->srcnum+i],-((gcfg->srcnum==1)?p->w:p->w*ppath[gcfg->w0offset+i]));
#else
	                field[(*idx1d+tshift*gcfg->dimlen.z)*gcfg->srcnum+i]+=-((gcfg->srcnum==1)?p->w:p->w*ppath[gcfg->w0offset+i]);
#endif
		    }
		  }
	      }
            }else{
	      saveexitppath(n_det,ppath,p,idx1d);
	    }
	  }
#ifdef SAVE_DETECTORS
      // let's handle detectors here
          if(gcfg->savedet){
             if((isdet&DET_MASK)==DET_MASK && (*mediaid==0 || (issvmc && 
	        (nuvox->sv.isupper ? nuvox->sv.upper : nuvox->sv.lower)==0)) && gcfg->issaveref<2)
	         savedetphoton(n_det,dpnum,ppath,p,v,s,photonseed,seeddata,isdet);
             clearpath(ppath,gcfg->partialdata);
          }
#endif
      }

      /**
       * If the thread completes all assigned photons, terminate this thread.
       */
      if((int)(f->ndone)>=(gcfg->threadphoton+(threadid<gcfg->oddphotons))-1){
          return 1; // all photos complete
      }

      ppath+= gcfg->partialdata;
     
      /**
       * If this is a replay of a detected photon, initilize the RNG with the stored seed here.
       */
      if(gcfg->seed==SEED_FROM_FILE){
          int seedoffset=(threadid*gcfg->threadphoton+min(threadid,gcfg->oddphotons-1)+max(0,(int)f->ndone+1))*RAND_BUF_LEN;
          for(int i=0;i<RAND_BUF_LEN;i++)
	      t[i]=rngseed[seedoffset+i];
      }

      /**
       * Attempt to launch a new photon until success
       */
      do{
	  *((float4*)p)=gcfg->ps;
	  *((float4*)v)=gcfg->c0;
	  *((float4*)f)=float4(0.f,0.f,gcfg->minaccumtime,f->ndone);
          *idx1d=gcfg->idx1dorig;
          *mediaid=gcfg->mediaidorig;
	  if(gcfg->issaveseed)
              copystate(t,photonseed);
          *rv=float3(gcfg->ps.x,gcfg->ps.y,gcfg->ps.z); //< reuse as the origin of the src, needed for focusable sources

          if(issvmc){
              nuvox->sv.issplit=0;   //< initialize the tissue type indicator under SVMC mode
              nuvox->sv.lower  =0;
              nuvox->sv.upper  =0;
              nuvox->sv.isupper=0;
	  }

          if(gcfg->maxpolmedia){
              *((float4*)s)=gcfg->s0;
          }

          /**
           * Only one branch is taken because of template, this can reduce thread divergence
           */
          if(!ispencil){
	    switch(gcfg->srctype) {
		case(MCX_SRC_PLANAR):        // a uniform square/rectangular/quadrilateral shaped area light source
		case(MCX_SRC_PATTERN):       // a square/rectangular/quadrilateral shaped area light source with intensity determined by a 2D array (0-1)
		case(MCX_SRC_PATTERN3D):     // a cubic/brick shaped 3D volumetric light source with intensity determined by a 3D array (0-1)
		case(MCX_SRC_FOURIER):       // SFDI Fourier patterns
		case(MCX_SRC_PENCILARRAY): { /*a rectangular grid over a plane*/
		      float rx=rand_uniform01(t);
		      float ry=rand_uniform01(t);
		      float rz;
		      if(gcfg->srctype==MCX_SRC_PATTERN3D){
		            rz=rand_uniform01(t);
		            *((float4*)p)=float4(p->x+rx*gcfg->srcparam1.x,
			 		         p->y+ry*gcfg->srcparam1.y,
					         p->z+rz*gcfg->srcparam1.z,
					         p->w);
		      }else{
		          *((float4*)p)=float4(p->x+rx*gcfg->srcparam1.x+ry*gcfg->srcparam2.x,
					       p->y+rx*gcfg->srcparam1.y+ry*gcfg->srcparam2.y,
					       p->z+rx*gcfg->srcparam1.z+ry*gcfg->srcparam2.z,
					       p->w);
		      }
		      if(gcfg->srctype==MCX_SRC_PATTERN){ // need to prevent rx/ry=1 here
		          if(gcfg->srcnum<=1){
			      p->w=srcpattern[(int)(ry*JUST_BELOW_ONE*gcfg->srcparam2.w)*(int)(gcfg->srcparam1.w)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.w)];
			      ppath[3]=p->w;
			  }else{
			    *((uint *)(ppath+2))=((int)(ry*JUST_BELOW_ONE*gcfg->srcparam2.w)*(int)(gcfg->srcparam1.w)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.w));
		            for(int i=0;i<gcfg->srcnum;i++)
			      ppath[i+3]=srcpattern[(*((uint *)(ppath+2)))*gcfg->srcnum+i];
			    p->w=1.f;
                          }
		      }else if(gcfg->srctype==MCX_SRC_PATTERN3D){
		          if(gcfg->srcnum<=1){
		              p->w=srcpattern[(int)(rz*JUST_BELOW_ONE*gcfg->srcparam1.z)*(int)(gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+
		                              (int)(ry*JUST_BELOW_ONE*gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.x)];
			      ppath[3]=p->w;
			  }else{
			      *((uint *)(ppath+2))=((int)(rz*JUST_BELOW_ONE*gcfg->srcparam1.z)*(int)(gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+
		                              (int)(ry*JUST_BELOW_ONE*gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.x));
		              for(int i=0;i<gcfg->srcnum;i++)
		                ppath[i+3]=srcpattern[(*((uint *)(ppath+2)))*gcfg->srcnum+i];
			    p->w=1.f;
			  }
		      }else if(gcfg->srctype==MCX_SRC_FOURIER)
			  p->w=(cosf((floorf(gcfg->srcparam1.w)*rx+floorf(gcfg->srcparam2.w)*ry
				  +gcfg->srcparam1.w-floorf(gcfg->srcparam1.w))*TWO_PI)*(1.f-gcfg->srcparam2.w+floorf(gcfg->srcparam2.w))+1.f)*0.5f; //between 0 and 1
		      else if(gcfg->srctype==MCX_SRC_PENCILARRAY){
			  p->x=gcfg->ps.x+ floorf(rx*gcfg->srcparam1.w)*gcfg->srcparam1.x/(gcfg->srcparam1.w-1.f)+floorf(ry*gcfg->srcparam2.w)*gcfg->srcparam2.x/(gcfg->srcparam2.w-1.f);
			  p->y=gcfg->ps.y+ floorf(rx*gcfg->srcparam1.w)*gcfg->srcparam1.y/(gcfg->srcparam1.w-1.f)+floorf(ry*gcfg->srcparam2.w)*gcfg->srcparam2.y/(gcfg->srcparam2.w-1.f);
			  p->z=gcfg->ps.z+ floorf(rx*gcfg->srcparam1.w)*gcfg->srcparam1.z/(gcfg->srcparam1.w-1.f)+floorf(ry*gcfg->srcparam2.w)*gcfg->srcparam2.z/(gcfg->srcparam2.w-1.f);
		      }
		      *idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
		      if(p->x<0.f || p->y<0.f || p->z<0.f || p->x>=gcfg->maxidx.x || p->y>=gcfg->maxidx.y || p->z>=gcfg->maxidx.z){
			  *mediaid=0;
		      }else{
			  *mediaid=media[*idx1d];
		      }
                      *rv=float3(rv->x+(gcfg->srcparam1.x+gcfg->srcparam2.x)*0.5f,
		                 rv->y+(gcfg->srcparam1.y+gcfg->srcparam2.y)*0.5f,
				 rv->z+(gcfg->srcparam1.z+gcfg->srcparam2.z)*0.5f);
		      break;
		}
		case(MCX_SRC_FOURIERX):
		case(MCX_SRC_FOURIERX2D): { // [v1x][v1y][v1z][|v2|]; [kx][ky][phi0][M], unit(v0) x unit(v1)=unit(v2)
		      float rx=rand_uniform01(t);
		      float ry=rand_uniform01(t);
		      float4 v2=gcfg->srcparam1;
		      // calculate v2 based on v2=|v2| * unit(v0) x unit(v1)
		      v2.w*=rsqrtf(gcfg->srcparam1.x*gcfg->srcparam1.x+gcfg->srcparam1.y*gcfg->srcparam1.y+gcfg->srcparam1.z*gcfg->srcparam1.z);
		      v2.x=v2.w*(gcfg->c0.y*gcfg->srcparam1.z - gcfg->c0.z*gcfg->srcparam1.y);
		      v2.y=v2.w*(gcfg->c0.z*gcfg->srcparam1.x - gcfg->c0.x*gcfg->srcparam1.z); 
		      v2.z=v2.w*(gcfg->c0.x*gcfg->srcparam1.y - gcfg->c0.y*gcfg->srcparam1.x);
		      *((float4*)p)=float4(p->x+rx*gcfg->srcparam1.x+ry*v2.x,
					   p->y+rx*gcfg->srcparam1.y+ry*v2.y,
					   p->z+rx*gcfg->srcparam1.z+ry*v2.z,
					   p->w);
		      if(gcfg->srctype==MCX_SRC_FOURIERX2D)
			 p->w=(sinf((gcfg->srcparam2.x*rx+gcfg->srcparam2.z)*TWO_PI)*sinf((gcfg->srcparam2.y*ry+gcfg->srcparam2.w)*TWO_PI)+1.f)*0.5f; //between 0 and 1
		      else
			 p->w=(cosf((gcfg->srcparam2.x*rx+gcfg->srcparam2.y*ry+gcfg->srcparam2.z)*TWO_PI)*(1.f-gcfg->srcparam2.w)+1.f)*0.5f; //between 0 and 1
   
		      *idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
		      if(p->x<0.f || p->y<0.f || p->z<0.f || p->x>=gcfg->maxidx.x || p->y>=gcfg->maxidx.y || p->z>=gcfg->maxidx.z){
			  *mediaid=0;
		      }else{
			  *mediaid=media[*idx1d];
		      }
                      *rv=float3(rv->x+(gcfg->srcparam1.x+v2.x)*0.5f,
		                 rv->y+(gcfg->srcparam1.y+v2.y)*0.5f,
				 rv->z+(gcfg->srcparam1.z+v2.z)*0.5f);
		      break;
		}
		case(MCX_SRC_DISK):
		case(MCX_SRC_GAUSSIAN): { // uniform disk distribution or collimated Gaussian-beam
		      // Uniform disk point picking
		      // http://mathworld.wolfram.com/DiskPointPicking.html
		      float sphi, cphi;
		      float phi=TWO_PI*rand_uniform01(t);
		      sincosf(phi,&sphi,&cphi);
		      float r;
		      if(gcfg->srctype==MCX_SRC_DISK)
			  r=sqrtf(rand_uniform01(t)*fabs(gcfg->srcparam1.x*gcfg->srcparam1.x-gcfg->srcparam1.y*gcfg->srcparam1.y) + gcfg->srcparam1.y*gcfg->srcparam1.y);
		      else if(fabs(gcfg->c0.w) < 1e-5f || fabs(gcfg->srcparam1.y) < 1e-5f)
		          r=sqrtf(-0.5f*logf(rand_uniform01(t)))*gcfg->srcparam1.x;
		      else{
		          r=gcfg->srcparam1.x*gcfg->srcparam1.x*M_PI/gcfg->srcparam1.y; //Rayleigh range
		          r=sqrtf(-0.5f*logf(rand_uniform01(t))*(1.f+(gcfg->c0.w*gcfg->c0.w/(r*r))))*gcfg->srcparam1.x;
                      }
		      if( v->z>-1.f+EPS && v->z<1.f-EPS ) {
			  float tmp0=1.f-v->z*v->z;
			  float tmp1=r*rsqrtf(tmp0);
			  *((float4*)p)=float4(
			       p->x+tmp1*(v->x*v->z*cphi - v->y*sphi),
			       p->y+tmp1*(v->y*v->z*cphi + v->x*sphi),
			       p->z-tmp1*tmp0*cphi                   ,
			       p->w
			  );
			  GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
		      }else{
			  p->x+=r*cphi;
			  p->y+=r*sphi;
			  GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
		      }
		      *idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
		      if(p->x<0.f || p->y<0.f || p->z<0.f || p->x>=gcfg->maxidx.x || p->y>=gcfg->maxidx.y || p->z>=gcfg->maxidx.z){
			  *mediaid=0;
		      }else{
			  *mediaid=media[*idx1d];
		      }
		      break;
		  }
		case(MCX_SRC_CONE):        // uniform cone beam
		case(MCX_SRC_ISOTROPIC):   // isotropic source
		case(MCX_SRC_ARCSINE): {   // uniform distribution in zenith angle, arcsine distribution if projected in orthogonal plane
		      // Uniform point picking on a sphere 
		      // http://mathworld.wolfram.com/SpherePointPicking.html
		      float ang,stheta,ctheta,sphi,cphi;
		      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
		      sincosf(ang,&sphi,&cphi);
		      if(gcfg->srctype==MCX_SRC_CONE){  // a solid-angle section of a uniform sphere
		          ang=cosf(gcfg->srcparam1.x);
		          ang=(gcfg->srcparam1.y>0.f) ? rand_uniform01(t)*gcfg->srcparam1.x : acos(rand_uniform01(t)*(1.0-ang)+ang); //sine distribution
		      }else{
			  if(gcfg->srctype==MCX_SRC_ISOTROPIC) // uniform sphere
			      ang=acosf(2.f*rand_uniform01(t)-1.f); //sine distribution
			  else
			      ang=ONE_PI*rand_uniform01(t); //uniform distribution in zenith angle, arcsine
		      }
		      sincosf(ang,&stheta,&ctheta);
		      rotatevector(v,stheta,ctheta,sphi,cphi);
                      canfocus=0;
		      break;
		}
		case(MCX_SRC_ZGAUSSIAN): {  // angular Gaussian (zenith angle of the incident beam)
		      float ang,stheta,ctheta,sphi,cphi;
		      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
		      sincosf(ang,&sphi,&cphi);
		      ang=sqrtf(-2.f*logf(rand_uniform01(t)))*(1.f-2.f*rand_uniform01(t))*gcfg->srcparam1.x;
		      sincosf(ang,&stheta,&ctheta);
		      rotatevector(v,stheta,ctheta,sphi,cphi);
                      canfocus=0;
		      break;
		}
		case(MCX_SRC_HYPERBOLOID_GAUSSIAN): { // hyperboloid gaussian beam, patch submitted by Gijs Buist (https://groups.google.com/g/mcx-users/c/wauKd1IbEJE/m/_7AQPgFYAAAJ)
		      float sphi, cphi;
		      float phi=TWO_PI*rand_uniform01(t);
		      sincosf(phi,&sphi,&cphi);

		      float r=sqrtf(-0.5f*logf(rand_uniform01(t)))*gcfg->srcparam1.x;

		      /** parameter to generate photon path from coordinates at focus (depends on focal distance and rayleigh range) */
		      float tt=-gcfg->srcparam1.y/gcfg->srcparam1.z;
		      float l=rsqrtf(r*r+gcfg->srcparam1.z*gcfg->srcparam1.z);

		      /** if beam direction is along +z or -z direction */
		      float3 pd=float3(r*(cphi-tt*sphi), r*(sphi+tt*cphi), 0.f); // position displacement from srcpos
		      float3 v0=float3(-r*sphi*l, r*cphi*l, gcfg->srcparam1.z*l); // photon dir. w.r.t the beam dir. v

		      /** if beam dir. is not +z or -z, compute photon position and direction after rotation */
		      if( v->z>-1.f+EPS && v->z<1.f-EPS ) {
		          float tmp0=1.f-v->z*v->z;
		          float tmp1=rsqrtf(tmp0);
		          float ctheta=v->z;
		          float stheta=sqrtf(tmp0);
		          cphi=v->x*tmp1;
		          sphi=v->y*tmp1;

			  /** photon position displacement after rotation */
			  pd=float3(pd.x*cphi*ctheta - pd.y*sphi, pd.x*sphi*ctheta + pd.y*cphi, -pd.x*stheta);

			  /** photon direction after rotation */
			  *((float4*)v)=float4(v0.x*cphi*ctheta - v0.y*sphi + v0.z*cphi*stheta,
					       v0.x*sphi*ctheta + v0.y*cphi + v0.z*sphi*stheta,
					       -v0.x*stheta + v0.z*ctheta,
					       v->nscat);
			  GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
		      } else {
		          *((float4*)v)=float4(v0.x, v0.y, (v->z > 0.f) ? v0.z : -v0.z, v->nscat);
		          GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
		      }

		      /** compute final launch position and update medium label */
		      *((float4*)p)=float4(p->x+pd.x, p->y+pd.y, p->z+pd.z, p->w);
		      *idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
		      if(p->x<0.f || p->y<0.f || p->z<0.f || p->x>=gcfg->maxidx.x || p->y>=gcfg->maxidx.y || p->z>=gcfg->maxidx.z){
		          *mediaid=0;
		      }else{
		          *mediaid=media[*idx1d];
		      }
		      break;
		}
		case(MCX_SRC_LINE):   // uniformally emitting line source, emitting cylindrically
		case(MCX_SRC_SLIT): { // a line source emitting only along a specified direction, like a light sheet
		      float r=rand_uniform01(t);
		      *((float4*)p)=float4(p->x+r*gcfg->srcparam1.x,
					   p->y+r*gcfg->srcparam1.y,
					   p->z+r*gcfg->srcparam1.z,
					   p->w);
		      if(gcfg->srctype==MCX_SRC_LINE){
			      float s,q;
			      r=1.f-2.f*rand_uniform01(t);
			      s=1.f-2.f*rand_uniform01(t);
			      q=sqrtf(1.f-v->x*v->x-v->y*v->y)*(rand_uniform01(t)>0.5f ? 1.f : -1.f);
			      *((float4*)v)=float4(v->y*q-v->z*s,v->z*r-v->x*q,v->x*s-v->y*r,v->nscat);
		      }
                      *rv=float3(rv->x+(gcfg->srcparam1.x)*0.5f,
		                 rv->y+(gcfg->srcparam1.y)*0.5f,
				 rv->z+(gcfg->srcparam1.z)*0.5f);
                      canfocus=(gcfg->srctype==MCX_SRC_SLIT);
		      break;
		}
	    }

	    if(p->w<=gcfg->minenergy)
	        continue;

            /**
             * If beam focus is set, determine the incident angle
             */
            if(canfocus){
	      if(isnan(gcfg->c0.w)){ // isotropic if focal length is nan
                float ang,stheta,ctheta,sphi,cphi;
                ang=TWO_PI*rand_uniform01(t); //next arimuth angle
                sincosf(ang,&sphi,&cphi);
                ang=acosf(2.f*rand_uniform01(t)-1.f); //sine distribution
                sincosf(ang,&stheta,&ctheta);
                rotatevector(v,stheta,ctheta,sphi,cphi);
	      }else if(gcfg->c0.w!=0.f){
	        float Rn2=(gcfg->c0.w > 0.f) - (gcfg->c0.w < 0.f);
	        rv->x+=gcfg->c0.w*v->x;
		rv->y+=gcfg->c0.w*v->y;
		rv->z+=gcfg->c0.w*v->z;
                v->x=Rn2*(rv->x-p->x);
                v->y=Rn2*(rv->y-p->y);
                v->z=Rn2*(rv->z-p->z);
		Rn2=rsqrtf(v->x*v->x+v->y*v->y+v->z*v->z); // normalize
                v->x*=Rn2;
                v->y*=Rn2;
                v->z*=Rn2;
	      }
	    }
	  }

          /**
           * Compute the reciprocal of the velocity vector
           */
          *rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));

          /**
           * If a photon is launched outside of the box, or inside a zero-voxel, move it until it hits a non-zero voxel
           */
	  if((*mediaid & MED_MASK)==0){
		 int idx=skipvoid<islabel, issvmc>(p, v, f, rv, media,t,nuvox); /** specular reflection of the bbx is taken care of here*/
		 if(idx>=0){
		     *idx1d=idx;
		     *mediaid=media[*idx1d];
		 }
	  }
	  *w0+=1.f;
	  GPUDEBUG(("retry %f: mediaid=%d idx=%d w0=%e\n",*w0, *mediaid, *idx1d, p->w));

	  /**
           * if launch attempted for over 1000 times, stop trying and return
           */
	  if(*w0>gcfg->maxvoidstep)
	     return -1;  // launch failed
      }while((*mediaid & MED_MASK)==0 || p->w<=gcfg->minenergy);
      
      /**
       * Now a photon is successfully launched, perform necssary initialization for a new trajectory
       */
      f->ndone++;
      updateproperty<islabel, issvmc>(prop,*mediaid,t,*idx1d,media,(float3*)p,nuvox);
      if(gcfg->debuglevel & MCX_DEBUG_MOVE)
          savedebugdata(p,(uint)f->ndone+threadid*gcfg->threadphoton+umin(threadid,(threadid<gcfg->oddphotons)*threadid),gdebugdata);

      /**
        total energy enters the volume. for diverging/converting 
        beams, this is less than nphoton due to specular reflection 
        loss. This is different from the wide-field MMC, where the 
        total launched energy includes the specular reflection loss
       */
      ppath[1]+=p->w;
      *w0=p->w;
      ppath[2]=((gcfg->srcnum>1)? ppath[2] : p->w); // store initial weight
      v->nscat=EPS;
      if(gcfg->outputtype==otRF){ // if run RF replay
          f->pathlen=photontof[(threadid*gcfg->threadphoton+min(threadid,gcfg->oddphotons-1)+(int)f->ndone)];
          sincosf(TWO_PI*gcfg->omega*f->pathlen, ppath+4+gcfg->srcnum, ppath+3+gcfg->srcnum);
      }
      f->pathlen=0.f;
      
      /**
       * If a progress bar is needed, only sum completed photons from the 1st, last and middle threads to determine progress bar
       */
      if((gcfg->debuglevel & MCX_DEBUG_PROGRESS) && ((int)(f->ndone) & 1) && (threadid==0 || threadid==blockDim.x * gridDim.x - 1 
          || threadid==((blockDim.x * gridDim.x)>>2) || threadid==(((blockDim.x * gridDim.x)>>1) + ((blockDim.x * gridDim.x)>>2))
	  || threadid==((blockDim.x * gridDim.x)>>1))) { //< use the 1st, middle and last thread for progress report
          gprogress[0]++;
      }
      return 0;
}


/**
 * @brief A stand-alone kernel to test random number generators
 *
 * This function fills the domain with the random numbers generated from 
 * the GPU. One can use this function to dump RNG values and test for quality.
 * use -D R in the command line to enable this output.
 *
 * @param[out] field: the array to be filled with RNG values
 * @param[in] n_seed: the seed to the RNG
 */

kernel void mcx_test_rng(OutputType field[],uint n_seed[]){
     int idx= blockDim.x * blockIdx.x + threadIdx.x;
     int i;
     int len=gcfg->maxidx.x*gcfg->maxidx.y*gcfg->maxidx.z*(int)((gcfg->twin1-gcfg->twin0)*gcfg->Rtstep+0.5f);
     RandType t[RAND_BUF_LEN];

     gpu_rng_init(t,n_seed,idx);

     for(i=0;i<len;i++){
	   field[i]=rand_uniform01(t);
     }
}

/**
 * @brief The core Monte Carlo photon simulation kernel (!!!Important!!!)
 *
 * This is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009.
 * everything in the GPU kernels is in grid-unit. To convert back to length, use
 * cfg->unitinmm (scattering/absorption coeff, T, speed etc)
 *
 * @param[in] media: domain medium index array, read-only
 * @param[out] field: the 3D/4D array where the fluence/energy-deposit are accummulated
 * @param[in,out] genergy: the array storing the total launched and escaped energy for each thread
 * @param[in] n_seed: the seed to the RNG of this thread
 * @param[in,out] n_pos: the initial position state of the photon for each thread
 * @param[in,out] n_dir: the initial direction state of the photon for each thread
 * @param[in,out] n_len: the initial parameter state of the photon for each thread
 * @param[in,out] detectedphoton: the buffer where the detected photon data are stored
 * @param[in] srcpattern: user-specified source pattern array if pattern source is used
 * @param[in] replayweight: the pre-computed detected photon weight for replay
 * @param[in] photontof: the pre-computed detected photon time-of-fly for replay
 * @param[in,out] seeddata: pointer to the buffer to save detected photon seeds
 * @param[in,out] gdebugdata: pointer to the buffer to save photon trajectory positions
 * @param[in,out] gprogress: pointer to the host variable to update progress bar
 */

template <const int ispencil, const int isreflect, const int islabel, const int issvmc>
kernel void mcx_main_loop(uint media[],OutputType field[],float genergy[],uint n_seed[],
     float4 n_pos[],float4 n_dir[],float4 n_len[],float n_det[], uint detectedphoton[], 
     float srcpattern[],float replayweight[],float photontof[],int photondetid[], 
     RandType *seeddata,float *gdebugdata, float *ginvcdf, float4 *gsmatrix, volatile int *gprogress){

     /** the 1D index of the current thread */
     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     if(idx>=gcfg->threadphoton*(blockDim.x * gridDim.x)+gcfg->oddphotons)
         return;
     MCXpos  p={0.f,0.f,0.f,-1.f};   //< Photon position state: {x,y,z}: coordinates in grid unit, w:packet weight
     MCXdir  v={0.f,0.f,0.f, 0.f};   //< Photon direction state: {x,y,z}: unitary direction vector in grid unit, nscat:total scat event
     MCXtime f={0.f,0.f,0.f,-1.f};   //< Photon parameter state: pscat: remaining scattering probability,t: photon elapse time, pathlen: total pathlen in one voxel, ndone: completed photons

     MCXsp nuvox;
     Stokes s;

     unsigned char testint=0;  //< flag used under SVMC mode: if a ray-interface intersection test is needed along current photon path
     unsigned char hitintf=0;  //< flag used under SVMC mode: if a photon path hit the intra-voxel interface inside a mixed voxel
     
     uint idx1d, idx1dold;    //< linear index to the current voxel in the media array

     uint  mediaid=gcfg->mediaidorig;
     uint  mediaidold=0;
     int   isdet=0;
     float  n1;               //< reflection var
     float3 htime;            //< time-of-flight for collision test
     float3 rv;               //< reciprocal velocity

     RandType t[RAND_BUF_LEN];
     Medium prop;

     float len, slen;
     OutputType w0;
     int   flipdir=-1;
 
     float *ppath=(float *)(sharedmem);

     /**
         Load use-defined phase function (inversion of CDF) to the shared memory (first gcfg->nphase floats)
      */
     if(gcfg->nphase){
         idx1d=gcfg->nphase/blockDim.x;
         for(idx1dold=0;idx1dold<idx1d;idx1dold++)
	     ppath[threadIdx.x*idx1d+idx1dold]=ginvcdf[threadIdx.x*idx1d+idx1dold];
	 if(gcfg->nphase-(idx1d*blockDim.x) > 0 && threadIdx.x==0){
	     for(idx1dold=0;idx1dold<gcfg->nphase-(idx1d*blockDim.x) ;idx1dold++)
	         ppath[blockDim.x*idx1d+idx1dold]=ginvcdf[blockDim.x*idx1d+idx1dold];
	 }
	 __threadfence_block();
     }

     ppath=(float *)(sharedmem+sizeof(float)*gcfg->nphase+blockDim.x*(gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)));
     ppath+=threadIdx.x*(gcfg->w0offset + gcfg->srcnum + 2*(gcfg->outputtype==otRF)); // block#2: maxmedia*thread number to store the partial
     clearpath(ppath,gcfg->w0offset + gcfg->srcnum);
     ppath[gcfg->partialdata]  =genergy[idx<<1];
     ppath[gcfg->partialdata+1]=genergy[(idx<<1)+1];

     *((float4*)(&prop))=gproperty[1];

     /**
	 Initialize RNG states
      */

     gpu_rng_init(t,n_seed,idx);

     /**
	 Launch the first photon
      */

     if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,0,ppath,
       n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),media,srcpattern,
       idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox)){
         GPUDEBUG(("thread %d: fail to launch photon\n",idx));
	 n_pos[idx]=*((float4*)(&p));
	 n_dir[idx]=*((float4*)(&v));
	 n_len[idx]=*((float4*)(&f));
         return;
     }

     /**
	 The following lines initialize photon state variables, RNG states and 
      */
     rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
     isdet=mediaid & DET_MASK;
     mediaid &= MED_MASK; // keep isdet to 0 to avoid launching photon ina 

     /**
         @brief The main photon movement loop
	 Each pass of this loop, we advance photon by one step - meaning that it is either
	 moved from one voxel to the immediately next voxel when the scattering path continues, 
	 or moved to the end of the scattering path if it ends within the current voxel.
      */

     while(f.ndone<(gcfg->threadphoton+(idx<gcfg->oddphotons))) {

          GPUDEBUG(("photonid [%d] L=%f w=%e medium=%d\n",(int)f.ndone,f.pscat,p.w,mediaid));

          /**
              @brief A scattering event
	      When a photon arrives at a scattering site, 3 values are regenrated
	      1 - a random unitless scattering length f.pscat,
	      2 - a 0-2pi uniformly random arimuthal angle
	      3 - a 0-pi random zenith angle based on the Henyey-Greenstein Phase Function
           */
	  if(f.pscat<=0.f) {  //< if this photon has finished his current scattering path, calculate next scat length & angles
   	       f.pscat=rand_next_scatlen(t); //< random scattering probability, unit-less, exponential distribution
		  
               GPUDEBUG(("scat L=%f RNG=[%0lX %0lX] \n",f.pscat,t[0],t[1]));
	       if(v.nscat!=EPS){ //< if v.nscat is EPS, this means it is the initial launch direction, no need to change direction
                       //< random arimuthal angle
	               float cphi=1.f,sphi=0.f,theta,stheta,ctheta;
                       float tmp0=0.f;
                       if(gcfg->maxpolmedia && !gcfg->is2d){
                           uint i=(uint)NANGLES*((mediaid & MED_MASK)-1);

                           /** Rejection method to choose azimuthal angle phi and deflection angle theta */
                           float I0,I,sin2phi,cos2phi;
                           do{
                               theta=acosf(2.f*rand_next_zangle(t)-1.f);
                               tmp0=TWO_PI*rand_next_aangle(t);
                               sincosf(2.f*tmp0,&sin2phi,&cos2phi);
                               I0=gsmatrix[i].x*s.i+gsmatrix[i].y*(s.q*cos2phi+s.u*sin2phi);
                               uint ithedeg=floorf(theta*NANGLES*(R_PI-EPS));
                               I=gsmatrix[i+ithedeg].x*s.i+gsmatrix[i+ithedeg].y*(s.q*cos2phi+s.u*sin2phi);
                           }while(rand_uniform01(t)*I0>=I);

                           sincosf(tmp0,&sphi,&cphi);
                           sincosf(theta,&stheta,&ctheta);

                           GPUDEBUG(("scat phi=%f\n",tmp0));
                           GPUDEBUG(("scat theta=%f\n",theta));
                       }else{
                           if(!gcfg->is2d){
                               tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                               sincosf(tmp0,&sphi,&cphi);
                           }
                           GPUDEBUG(("scat phi=%f\n",tmp0));

                           if(gcfg->nphase>2){ // after padding the left/right ends, nphase must be 3 or more
                               tmp0=rand_uniform01(t)*(gcfg->nphase-1);
                               theta=tmp0-((int)tmp0);
                               tmp0=(1.f-theta)*((float *)(sharedmem))[(int)tmp0   >= gcfg->nphase ? gcfg->nphase-1 : (int)(tmp0)  ]+
                                         theta *((float *)(sharedmem))[(int)tmp0+1 >= gcfg->nphase ? gcfg->nphase-1 : (int)(tmp0)+1];
                               theta=acosf(tmp0);
                               stheta=sinf(theta);
                               ctheta=tmp0;
                           }else{
                               tmp0=(v.nscat > gcfg->gscatter) ? 0.f : prop.g;
                               /** Here we use Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002 */
                               if(fabs(tmp0)>EPS){  //< if prop.g is too small, the distribution of theta is bad
                                   tmp0=(1.f-prop.g*prop.g)/(1.f-prop.g+2.f*prop.g*rand_next_zangle(t));
                                   tmp0*=tmp0;
                                   tmp0=(1.f+prop.g*prop.g-tmp0)/(2.f*prop.g);

                                   // in early CUDA, when ran=1, CUDA gives 1.000002 for tmp0 which produces nan later
                                   // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                                   tmp0=fmax(-1.f, fmin(1.f, tmp0));

                                   theta=acosf(tmp0);
                                   stheta=sinf(theta);
                                   ctheta=tmp0;
                               }else{
                                   theta=acosf(2.f*rand_next_zangle(t)-1.f);
                                   sincosf(theta,&stheta,&ctheta);
                               }
                           }
                           GPUDEBUG(("scat theta=%f\n",theta));
                       }
#ifdef SAVE_DETECTORS
                       if(gcfg->savedet){
                           if(SAVE_NSCAT(gcfg->savedetflag)){
			       if(issvmc){  //< SVMC mode
			           if((nuvox.sv.isupper ? nuvox.sv.upper:nuvox.sv.lower)>0)
		                       ppath[((nuvox.sv.isupper) ? nuvox.sv.upper:nuvox.sv.lower)-1]++;
			       }else
			           ppath[(mediaid & MED_MASK)-1]++;
			   }
			   /** accummulate momentum transfer */
			   if(SAVE_MOM(gcfg->savedetflag)){
			       if(issvmc){  //< SVMC mode
			           if((nuvox.sv.isupper ? nuvox.sv.upper:nuvox.sv.lower)>0)
				       ppath[gcfg->maxmedia*(SAVE_NSCAT(gcfg->savedetflag)+SAVE_PPATH(gcfg->savedetflag))+
				             ((nuvox.sv.isupper) ? nuvox.sv.upper:nuvox.sv.lower)-1]+=1.f-ctheta;
			       }else
			           ppath[gcfg->maxmedia*(SAVE_NSCAT(gcfg->savedetflag)+SAVE_PPATH(gcfg->savedetflag))+
				         (mediaid & MED_MASK)-1]+=1.f-ctheta;
			   }
		       }
#endif
                       /** Store old direction cosines for polarized photon simulation */
                       if(gcfg->maxpolmedia){
                           rv=float3(v.x,v.y,v.z);
                       }

                       /** Update direction vector with the two random angles */
		       if(gcfg->is2d)
		           rotatevector2d(&v,(rand_next_aangle(t)>0.5f ? stheta: -stheta),ctheta);
		       else
                           rotatevector(&v,stheta,ctheta,sphi,cphi);
                       v.nscat++;

                       /** Update stokes parameters */
                       if(gcfg->maxpolmedia) updatestokes(&s, theta, tmp0, (float3*)&rv, (float3*)&v, &mediaid, gsmatrix);

		       /** Only compute the reciprocal vector when v is changed, this saves division calculations, which are very expensive on the GPU */
                       rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
                       if(gcfg->outputtype==otWP || gcfg->outputtype==otDCS){
                            //< photontof[] and replayweight[] should be cached using local mem to avoid global read
                            int tshift=(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone);
			    tmp0=(gcfg->outputtype==otDCS)? (1.f-ctheta) : 1.f;
                            tshift=(int)(floorf((photontof[tshift]-gcfg->twin0)*gcfg->Rtstep)) + 
                                 ( (gcfg->replaydet==-1)? ((photondetid[tshift]-1)*gcfg->maxgate) : 0);
#ifdef USE_ATOMIC
                            if(!gcfg->isatomic){
#endif
                                field[idx1d+tshift*gcfg->dimlen.z]+=tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)];
#ifdef USE_ATOMIC
                            }else{
    #ifdef USE_DOUBLE
			        atomicAdd(& field[idx1d+tshift*gcfg->dimlen.z], tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]);
    #else
			        float oldval=atomicadd(& field[idx1d+tshift*gcfg->dimlen.z], tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]);
				    if(oldval>MAX_ACCUM){
				        if(atomicadd(& field[idx1d+tshift*gcfg->dimlen.z], -oldval)<0.f)
					    atomicadd(& field[idx1d+tshift*gcfg->dimlen.z], oldval);
					else
					    atomicadd(& field[idx1d+tshift*gcfg->dimlen.z+gcfg->dimlen.w], oldval);
				    }
    #endif
                                GPUDEBUG(("atomic write to [%d] %e, w=%f\n",idx1d,tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)],p.w));
                            }
#endif
                       }
                       if(gcfg->debuglevel & MCX_DEBUG_MOVE)
                           savedebugdata(&p,(uint)f.ndone+idx*gcfg->threadphoton+umin(idx,(idx<gcfg->oddphotons)*idx),gdebugdata);
	       }
	       v.nscat=(int)v.nscat;
	       if(issvmc) testint=1;  //< new propagation direction after scattering, enable ray-interface intersection test
	  }

          /** Read the optical property of the current voxel */
          n1=prop.n;
	  if(islabel)
	    *((float4*)(&prop))=gproperty[mediaid & MED_MASK];
	  else if(issvmc){
	    if(!nuvox.sv.issplit)
	      updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox);
	  }else
	    updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox);

	  /** Advance photon 1 step to the next voxel */
	  len=(gcfg->faststep) ? gcfg->minstep : hitgrid((float3*)&p,(float3*)&v,&(htime.x),&rv.x,&flipdir); // propagate the photon to the first intersection to the grid
	  
	  /** convert photon movement length to unitless scattering length by multiplying with mus */
	  slen=len*prop.mus*(v.nscat+1.f > gcfg->gscatter ? (1.f-prop.g) : 1.f); //unitless (minstep=grid, mus=1/grid)

          GPUDEBUG(("p=[%f %f %f] -> <%f %f %f>*%f -> hit=[%f %f %f] flip=%d\n",p.x,p.y,p.z,v.x,v.y,v.z,len,htime.x,htime.y,htime.z,flipdir));

	  /** if the consumed unitless scat length is less than what's left in f.pscat, keep moving; otherwise, stop in this voxel */
	  slen=fmin(slen,f.pscat);
	  
	  /** final length that the photon moves - either the length to move to the next voxel, or the remaining scattering length */
	  len=slen/(prop.mus*(v.nscat+1.f > gcfg->gscatter ? (1.f-prop.g) : 1.f));
	  
	  /** perform ray-interface intersection test to consider intra-voxel curvature (SVMC mode) */
	  if(issvmc){
	    if(nuvox.sv.issplit && testint)
	      hitintf=ray_plane_intersect((float3*)&p,&v,&prop,len,slen,&nuvox,f,htime,flipdir);
	    else
	      hitintf=0;
	  }
	  
	  /** if photon moves to the next voxel, use the precomputed intersection coord. htime which are assured to be outside of the current voxel */
	  *((float3*)(&p)) = (gcfg->faststep || slen==f.pscat || hitintf) ? float3(p.x+len*v.x,p.y+len*v.y,p.z+len*v.z) : float3(htime.x,htime.y,htime.z);
	  
	  /** calculate photon energy loss */
#ifdef USE_MORE_DOUBLE
	  p.w*=exp(-(OutputType)prop.mua*len);
#else
	  p.w*=expf(-prop.mua*len);
#endif

	  /** remaining unitless scattering length: sum(s_i*mus_i), unit-less */
	  f.pscat-=slen;

	  /** update photon timer to add time-of-flight (unit = s) */
	  f.t+=len*prop.n*gcfg->oneoverc0;
	  f.pathlen+=len;

          GPUDEBUG(("update p=[%f %f %f] -> len=%f\n",p.x,p.y,p.z,len));

#ifdef SAVE_DETECTORS
	  /** accummulate partial path of the current medium */
	  if(gcfg->savedet)
	    if(SAVE_PPATH(gcfg->savedetflag))
	      if(issvmc){
	        if((nuvox.sv.isupper ? nuvox.sv.upper:nuvox.sv.lower)>0)
		  ppath[gcfg->maxmedia*(SAVE_NSCAT(gcfg->savedetflag))+(nuvox.sv.isupper ? nuvox.sv.upper:nuvox.sv.lower)-1]+=len; //(unit=grid)
	      }else
	        ppath[gcfg->maxmedia*(SAVE_NSCAT(gcfg->savedetflag))+(mediaid & MED_MASK)-1]+=len; //(unit=grid)
#endif

          mediaidold=mediaid | isdet;
          idx1dold=idx1d;
          idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
          GPUDEBUG(("idx1d [%d]->[%d]\n",idx1dold,idx1d));

	  /** read the medium index of the new voxel (current or next) */
          if(p.x<0.f||p.y<0.f||p.z<0.f||p.x>=gcfg->maxidx.x||p.y>=gcfg->maxidx.y||p.z>=gcfg->maxidx.z){
              /** if photon moves outside of the volume, set mediaid to 0 */
	      mediaid=0;
	      idx1d=(p.x<0.f||p.y<0.f||p.z<0.f) ? OUTSIDE_VOLUME_MIN : OUTSIDE_VOLUME_MAX;
	      isdet=gcfg->bc[(idx1d==OUTSIDE_VOLUME_MAX)*3+flipdir];  /** isdet now stores the boundary condition flag, this will be overwriten before the end of the loop */
              GPUDEBUG(("moving outside: [%f %f %f], idx1d [%d]->[out], bcflag %d\n",p.x,p.y,p.z,idx1d,isdet));
	  }else{
              /** otherwise, read the optical property index */
	      mediaid=media[idx1d];
	      isdet=mediaid & DET_MASK;  /** upper 16bit is the mask of the covered detector */
	      mediaid &= MED_MASK;       /** lower 16bit is the medium index */
          }
          GPUDEBUG(("medium [%d]->[%d]\n",mediaidold,mediaid));

          /**  save fluence to the voxel when photon moves out */
	  if((idx1d!=idx1dold || (issvmc && hitintf)) && mediaidold){

             /**  if t is within the time window, which spans cfg->maxgate*cfg->tstep.wide */
             if(gcfg->save2pt && f.t>=gcfg->twin0 && f.t<gcfg->twin1){
#ifdef USE_MORE_DOUBLE
	          OutputType weight=ZERO;
#else
	          float weight=0.f;
#endif
                  int tshift=(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep));
		  
		  /** calculate the quality to be accummulated */
		  if(gcfg->outputtype==otEnergy)
		      weight=w0-p.w;
		  else if(gcfg->outputtype==otFluence || gcfg->outputtype==otFlux)
		      weight=(prop.mua==0.f) ? 0.f : ((w0-p.w)/(prop.mua));
		  else if(gcfg->seed==SEED_FROM_FILE){
		      if(gcfg->outputtype==otJacobian || gcfg->outputtype==otRF){
		        weight=replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]*f.pathlen;
			if(gcfg->outputtype==otRF)
			    weight=-weight*ppath[gcfg->w0offset+gcfg->srcnum];
			tshift=(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone);
			tshift=(int)(floorf((photontof[tshift]-gcfg->twin0)*gcfg->Rtstep)) + 
			   ( (gcfg->replaydet==-1)? ((photondetid[tshift]-1)*gcfg->maxgate) : 0);
		      }
		  }

                  GPUDEBUG(("deposit to [%d] %e, w=%f\n",idx1dold,weight,p.w));

              if(weight>0.f){
#ifdef USE_ATOMIC
                if(!gcfg->isatomic){
#endif
                  /** accummulate the quality to the volume using non-atomic operations  */
                  field[idx1dold+tshift*gcfg->dimlen.z]+=weight;
#ifdef USE_ATOMIC
               }else{
	          /** accummulate the quality to the volume using atomic operations  */
                  // ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
		  if(gcfg->srctype!=MCX_SRC_PATTERN && gcfg->srctype!=MCX_SRC_PATTERN3D){
    #ifdef USE_DOUBLE
		      atomicAdd(& field[idx1dold+tshift*gcfg->dimlen.z], weight);
    #else
		      float oldval=atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], weight);
		      if(oldval>MAX_ACCUM && gcfg->outputtype!=otRF){
			if(atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], -oldval)<0.f)
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], oldval);
			else
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z+gcfg->dimlen.w], oldval);
		      }else if(gcfg->outputtype==otRF && gcfg->omega>0.f){
			oldval=-p.w*f.pathlen*ppath[gcfg->w0offset+gcfg->srcnum+1];
		        atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z+gcfg->dimlen.w], oldval);
		      }
    #endif
		  }else{
		      for(int i=0;i<gcfg->srcnum;i++){
		        if(ppath[gcfg->w0offset+i]>0.f){
    #ifdef USE_DOUBLE
		          atomicAdd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i], weight*ppath[gcfg->w0offset+i]);
    #else
		          float oldval=atomicadd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i], weight*ppath[gcfg->w0offset+i]);
		          if(oldval>MAX_ACCUM && gcfg->outputtype!=otRF){
			      if(atomicadd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i], -oldval)<0.f)
			        atomicadd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i], oldval);
			      else
			        atomicadd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i+gcfg->dimlen.w], oldval);
			  }else if(gcfg->outputtype==otRF){
			      oldval=p.w*f.pathlen*ppath[gcfg->w0offset+gcfg->srcnum+1];
		              atomicadd(& field[(idx1dold+tshift*gcfg->dimlen.z)*gcfg->srcnum+i+gcfg->dimlen.w], oldval);
		          }
    #endif
			}
		      }
		  }
                  GPUDEBUG(("atomic write to [%d] %e, w=%f\n",idx1dold,weight,p.w));
               }
#endif
              }
	     }
	     w0=p.w;
	     f.pathlen=0.f;
	  }else
	       mediaid = mediaidold;

	  /** in SVMC mode, update tissue type when photons cross voxel or intra-voxel boundary */
	  if(issvmc){
	      if(idx1d!=idx1dold){
		  updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox);
		  testint=1; // re-enable ray-interface intesection test after launching a new photon under SVMC mode
	      }else if(hitintf){
		  nuvox.nv=-nuvox.nv;  // flip normal vector for transmission
		  nuvox.sv.isupper=!nuvox.sv.isupper;
		  testint=0; // disable ray-interafece intersection test immediately after an intersection event
	      }
	  }

	  /** launch new photon when exceed time window or moving from non-zero voxel to zero voxel without reflection */
          if((mediaid==0 && (((isdet & 0xF)==0 && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[0].w))) || (isdet==bcAbsorb || isdet==bcCyclic) )) || 
	      (issvmc && (idx1d!=idx1dold || hitintf) && !nuvox.sv.isupper && !nuvox.sv.lower && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[0].w))) ||
	      f.t>gcfg->twin1){
	      if(isdet==bcCyclic){
                 if(flipdir==0)  p.x=mcx_nextafterf(roundf(p.x+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.x: -gcfg->maxidx.x)),(v.x > 0.f)-(v.x < 0.f));
                 if(flipdir==1)  p.y=mcx_nextafterf(roundf(p.y+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.y: -gcfg->maxidx.y)),(v.y > 0.f)-(v.y < 0.f));
                 if(flipdir==2)  p.z=mcx_nextafterf(roundf(p.z+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.z: -gcfg->maxidx.z)),(v.z > 0.f)-(v.z < 0.f));
		 if(!(p.x<0.f||p.y<0.f||p.z<0.f||p.x>=gcfg->maxidx.x||p.y>=gcfg->maxidx.y||p.z>=gcfg->maxidx.z)){
                     idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
	             mediaid=media[idx1d];
	             isdet=mediaid & DET_MASK;  /** upper 16bit is the mask of the covered detector */
	             mediaid &= MED_MASK;       /** lower 16bit is the medium index */
                     GPUDEBUG(("Cyclic boundary condition, moving photon in dir %d at %d flag, new pos=[%f %f %f]\n",flipdir,isdet,p.x,p.y,p.z));
	             continue;
		 }
	      }
              GPUDEBUG(("direct relaunch at idx=[%d] mediaid=[%d], ref=[%d] bcflag=%d timegate=%d\n",idx1d,mediaid,gcfg->doreflect,isdet,f.t>gcfg->twin1));
	      if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,
	          (((idx1d==OUTSIDE_VOLUME_MAX && gcfg->bc[9+flipdir]) || (idx1d==OUTSIDE_VOLUME_MIN && gcfg->bc[6+flipdir]))? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
	          ppath, n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
		  media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox))
                   break;
              isdet=mediaid & DET_MASK;
              mediaid &= MED_MASK;
	      if(issvmc) testint=1; // re-enable ray-interface intesection test after launching a new photon under SVMC mode
	      continue;
	  }

          /** perform Russian Roulette*/
          if(p.w < gcfg->minenergy){
                if(rand_do_roulette(t)*ROULETTE_SIZE<=1.f)
                   p.w*=ROULETTE_SIZE;
                else{
                   GPUDEBUG(("relaunch after Russian roulette at idx=[%d] mediaid=[%d], ref=[%d]\n",idx1d,mediaid,gcfg->doreflect));
                   if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),ppath,
	                n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
			media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox))
                        break;
                   isdet=mediaid & DET_MASK;
                   mediaid &= MED_MASK;
		   if(issvmc) testint=1;
                   continue;
               }
          }

          /** do boundary reflection/transmission */
          if(isreflect){
              if(gcfg->mediaformat<100 && !issvmc)
                  updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox); //< optical property across the interface
              if(issvmc && hitintf){
                  if(gproperty[nuvox.sv.lower].w != gproperty[nuvox.sv.upper].w){
                      nuvox.nv=-nuvox.nv; // flip normal vector back for reflection/refraction computation
                      if(reflectray(n1,(float3*)&(v),&rv,&nuvox,&prop,t)){ // true if photon transmits to background media
                          if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),
                              ppath,n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
                              media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox))
                              break;
                          isdet=mediaid & DET_MASK;
                          mediaid &= MED_MASK;
                          testint=1; //< launch new photon, enable ray-interafece inter. test for next step
                          continue;
                      }
                  }else{
                      *((float4*)(&prop))=gproperty[nuvox.sv.isupper ? nuvox.sv.upper : nuvox.sv.lower];
                  }
              }else{
                  if(((isreflect && (isdet & 0xF)==0) || (isdet & 0x1)) && ((isdet & 0xF)==bcMirror || n1!=((gcfg->mediaformat<100)? (prop.n):(gproperty[(mediaid>0 && gcfg->mediaformat>=100)?1:mediaid].w)))){
                      float Rtotal=1.f;
                      float cphi,sphi,stheta,ctheta,tmp0,tmp1;

                      if(!issvmc) updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox);

                      tmp0=n1*n1;
                      tmp1=prop.n*prop.n;
                      cphi=fabs( (flipdir==0) ? v.x : (flipdir==1 ? v.y : v.z)); // cos(si)
                      sphi=1.f-cphi*cphi;            // sin(si)^2

                      len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2 = cos(ti)^2
                      GPUDEBUG(("ref total ref=%f\n",len));

                      if(len>0.f) { //< if no total internal reflection
                          ctheta=tmp0*cphi*cphi+tmp1*len;
                          stheta=2.f*n1*prop.n*cphi*sqrtf(len);
                          Rtotal=(ctheta-stheta)/(ctheta+stheta);
                          ctheta=tmp1*cphi*cphi+tmp0*len;
                          Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
                          GPUDEBUG(("Rtotal=%f\n",Rtotal));
                      } //< else, total internal reflection
                      if(Rtotal<1.f && (((isdet & 0xF)==0 && ((gcfg->mediaformat<100) ? prop.n:gproperty[mediaid].w) >= 1.f) || isdet==bcReflect) && (isdet!=bcMirror) && rand_next_reflect(t)>Rtotal){ // do transmission
                          transmit(&v,n1,prop.n,flipdir);
                          if(mediaid==0 || (issvmc && (nuvox.sv.isupper ? nuvox.sv.upper : nuvox.sv.lower)==0)) { // transmission to external boundary
                              GPUDEBUG(("transmit to air, relaunch\n"));
                              if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,
                                  (((idx1d==OUTSIDE_VOLUME_MAX && gcfg->bc[9+flipdir]) || (idx1d==OUTSIDE_VOLUME_MIN && gcfg->bc[6+flipdir]))? OUTSIDE_VOLUME_MIN : (mediaidold & DET_MASK)),
                                  ppath,n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
                                  media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox))
                                  break;
                              isdet=mediaid & DET_MASK;
                              mediaid &= MED_MASK;
                              if(issvmc) testint=1;
                              continue;
                          }
                          GPUDEBUG(("do transmission\n"));
                          rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
                      }else{ //< do reflection
                          GPUDEBUG(("ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n",flipdir,p.x,p.y,p.z,v.x,v.y,v.z));
                          (flipdir==0) ? (v.x=-v.x) : ((flipdir==1) ? (v.y=-v.y) : (v.z=-v.z)) ;
                          rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
                          (flipdir==0) ?
                              (p.x=mcx_nextafterf(__float2int_rn(p.x), (v.x > 0.f)-(v.x < 0.f))) :
                              ((flipdir==1) ?
                                  (p.y=mcx_nextafterf(__float2int_rn(p.y), (v.y > 0.f)-(v.y < 0.f))) :
                                  (p.z=mcx_nextafterf(__float2int_rn(p.z), (v.z > 0.f)-(v.z < 0.f))) );
                          GPUDEBUG(("ref p_new=[%f %f %f] v_new=[%f %f %f]\n",p.x,p.y,p.z,v.x,v.y,v.z));
                          idx1d=idx1dold;
                          mediaid=(media[idx1d] & MED_MASK);
                          updateproperty<islabel, issvmc>(&prop,mediaid,t,idx1d,media,(float3*)&p,&nuvox); //< optical property across the interface
                          if(issvmc && (nuvox.sv.isupper?nuvox.sv.upper:nuvox.sv.lower)==0){ // terminate photon if photon is reflected to background medium
                              if(launchnewphoton<ispencil, isreflect, islabel, issvmc>(&p,&v,&s,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),
                                  ppath,n_det,detectedphoton,t,(RandType*)(sharedmem+sizeof(float)*gcfg->nphase+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
                                  media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress,photontof,&nuvox))
                                  break;
                              isdet=mediaid & DET_MASK;
                              mediaid &= MED_MASK;
                              testint=1;
                              continue;
                          }
                          n1=prop.n;
                      }
                  }else if(gcfg->mediaformat<100 && !issvmc){
                      updateproperty<islabel, issvmc>(&prop,mediaidold,t,idx1d,media,(float3*)&p,&nuvox);
                  }
              }
          }else{
              if(issvmc) *((float4*)(&prop))=gproperty[nuvox.sv.isupper ? nuvox.sv.upper : nuvox.sv.lower];
          }
     }

     /** return the accumulated total energyloss and launched energy back to the host */
     genergy[idx<<1]    =ppath[gcfg->partialdata];
     genergy[(idx<<1)+1]=ppath[gcfg->partialdata+1];
     
     if(gcfg->issaveref>1)
         *detectedphoton=gcfg->maxdetphoton;

     /** for debugging purposes, we also pass the last photon states back to the host for printing */
     n_pos[idx]=*((float4*)(&p));
     n_dir[idx]=*((float4*)(&v));
     n_len[idx]=*((float4*)(&f));
}

/**
   assert cuda memory allocation result
*/
void mcx_cu_assess(cudaError_t cuerr,const char *file, const int linenum){
     if(cuerr!=cudaSuccess){
         CUDA_ASSERT(cudaDeviceReset());
         mcx_error(-(int)cuerr,(char *)cudaGetErrorString(cuerr),file,linenum);
     }
}

/**
 * @brief Utility function to calculate the GPU stream processors (cores) per SM
 *
 * Obtain GPU core number per MP, this replaces 
 * ConvertSMVer2Cores() in libcudautils to avoid 
 * extra dependency.
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_corecount(int v1, int v2){
     int v=v1*10+v2;
     if(v<20)      return 8;
     else if(v<21) return 32;
     else if(v<30) return 48;
     else if(v<50) return 192;
     else if(v<60 || v==61) return 128;
     else          return 64;
}

/**
 * @brief Utility function to calculate the maximum blocks per SM
 *
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_smxblock(int v1, int v2){
     int v=v1*10+v2;
     if(v<30)      return 8;
     else if(v<50) return 16;
     else          return 32;
}

/**
 * @brief Utility function to calculate the maximum blocks per SM
 *
 *
 * @param[in] v1: the major version of an NVIDIA GPU
 * @param[in] v2: the minor version of an NVIDIA GPU
 */

int mcx_threadmultiplier(int v1, int v2){
     int v=v1*10+v2;
     if(v<=75)      return 1;
     else           return 2;
}


/**
 * @brief Utility function to query GPU info and set active GPU
 *
 * This function query and list all available GPUs on the system and print
 * their parameters. This is used when -L or -I is used.
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[out] info: the GPU information structure
 */
 
int mcx_list_gpu(Config *cfg, GPUInfo **info){

#if __DEVICE_EMULATION__
    return 1;
#else
    int dev;
    int deviceCount,activedev=0;
    
    cudaError_t cuerr=cudaGetDeviceCount(&deviceCount);
    if(cuerr!=cudaSuccess){
	if(cuerr==(cudaError_t)30)
            mcx_error(-(int)cuerr,"A CUDA-capable GPU is not found or configured",__FILE__,__LINE__);
        CUDA_ASSERT(cuerr);
    }

    if (deviceCount == 0){
        MCX_FPRINTF(stderr,S_RED "ERROR: No CUDA-capable GPU device found\n" S_RESET);
        return 0;
    }
    *info=(GPUInfo *)calloc(deviceCount,sizeof(GPUInfo));
    if (cfg->gpuid && cfg->gpuid > deviceCount){
        MCX_FPRINTF(stderr,S_RED "ERROR: Specified GPU ID is out of range\n" S_RESET);
        return 0;
    }
    // scan from the first device
    for (dev = 0; dev<deviceCount; dev++) {
        cudaDeviceProp dp;
        CUDA_ASSERT(cudaGetDeviceProperties(&dp, dev));

	if(cfg->isgpuinfo==3)
	   activedev++;
        else if(cfg->deviceid[dev]=='1'){
           cfg->deviceid[dev]='\0';
           cfg->deviceid[activedev]=dev+1;
           activedev++;
        }
        strncpy((*info)[dev].name,dp.name,MAX_SESSION_LENGTH);
        (*info)[dev].id=dev+1;
	(*info)[dev].devcount=deviceCount;
	(*info)[dev].major=dp.major;
	(*info)[dev].minor=dp.minor;
	(*info)[dev].globalmem=dp.totalGlobalMem;
	(*info)[dev].constmem=dp.totalConstMem;
	(*info)[dev].sharedmem=dp.sharedMemPerBlock;
	(*info)[dev].regcount=dp.regsPerBlock;
	(*info)[dev].clock=dp.clockRate;
	(*info)[dev].sm=dp.multiProcessorCount;
	(*info)[dev].core=dp.multiProcessorCount*mcx_corecount(dp.major,dp.minor);
	(*info)[dev].maxmpthread=dp.maxThreadsPerMultiProcessor;
        (*info)[dev].maxgate=cfg->maxgate;
        (*info)[dev].autoblock=MAX((*info)[dev].maxmpthread / mcx_smxblock(dp.major,dp.minor),64);
        if((*info)[dev].autoblock==0){
             MCX_FPRINTF(stderr,S_RED "WARNING: maxThreadsPerMultiProcessor can not be detected\n" S_RESET);
             (*info)[dev].autoblock=64;
        }
        (*info)[dev].autothread=(*info)[dev].autoblock * mcx_smxblock(dp.major,dp.minor) * (*info)[dev].sm * mcx_threadmultiplier(dp.major,dp.minor);

        if (strncmp(dp.name, "Device Emulation", 16)) {
	  if(cfg->isgpuinfo){
	    MCX_FPRINTF(stdout,S_BLUE"=============================   GPU Information  ================================\n" S_RESET);
	    MCX_FPRINTF(stdout,"Device %d of %d:\t\t%s\n",(*info)[dev].id,(*info)[dev].devcount,(*info)[dev].name);
	    MCX_FPRINTF(stdout,"Compute Capability:\t%u.%u\n",(*info)[dev].major,(*info)[dev].minor);
	    MCX_FPRINTF(stdout,"Global Memory:\t\t%.0f B\nConstant Memory:\t%.0f B\n"
				"Shared Memory:\t\t%.0f B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
               (double)(*info)[dev].globalmem,(double)(*info)[dev].constmem,
               (double)(*info)[dev].sharedmem,(unsigned int)(*info)[dev].regcount,(*info)[dev].clock*1e-6f);
	  #if CUDART_VERSION >= 2000
	       MCX_FPRINTF(stdout,"Number of SMs:\t\t%u\nNumber of Cores:\t%u\n",
	          (*info)[dev].sm,(*info)[dev].core);
	  #endif
            MCX_FPRINTF(stdout,"Auto-thread:\t\t%d\n",(*info)[dev].autothread);
            MCX_FPRINTF(stdout,"Auto-block:\t\t%d\n", (*info)[dev].autoblock);
	  }
	}
    }
    if(cfg->isgpuinfo==2 && cfg->parentid==mpStandalone){ //list GPU info only
          exit(0);
    }
    if(activedev<MAX_DEVICE)
        cfg->deviceid[activedev]='\0';

    return activedev;
#endif
}


/**
 * @brief Master host code for the MCX simulation kernel (!!!Important!!!)
 *
 * This function is the master host code for the MCX kernel. It is responsible
 * for initializing all GPU variables, copy data from host to GPU, launch the
 * kernel for simulation, wait for competion, and retrieve the results.
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[in] gpu: the GPU information structure
 */

void mcx_run_simulation(Config *cfg,GPUInfo *gpu){

     int i,iter;
     float  minstep=1.f; //MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     
     /** \c p0 - initial photon positions for pencil/isotropic/cone beams, used to initialize \c{p={x,y,z,w}} state in the GPU kernel */
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     
     /** \c c0 - initial photon direction state, used to initialize \c{MCXdir v={vx,vy,vz,}} state in the GPU kernel */
     float4 c0=cfg->srcdir;
     
     /** \c stokesvec - initial photon polarization state, described by Stokes Vector */
     float4 s0=(float4)cfg->srciquv;
     
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     int timegate=0, totalgates, gpuid, threadid=0;
     
     /** \c gpuphoton - number of photons to be simulated per thread, determined by total workload and thread number */
     size_t gpuphoton=0;
     
     /** \c photoncount - number of completed photons returned by all thread, output */
     size_t photoncount=0;

     unsigned int printnum;
     unsigned int tic,tic0,tic1,toc=0,debuglen=MCX_DEBUG_REC_LEN;
     size_t fieldlen;
     uint3 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint4 dimlen;
     float Vvox,fullload=0.f;

     /** \c mcgrid - GPU grid size, only use 1D grid, used when launching the kernel in cuda <<<>>> operator */
     dim3 mcgrid;
     
     /** \c mcblock - GPU block size, only use 1D block, used when launching the kernel in cuda <<<>>> operator */
     dim3 mcblock;
     
     /** \c sharedbuf - shared memory buffer length to be requested, used when launching the kernel in cuda <<<>>> operator */
     uint sharedbuf=0;

     /** \c dimxyz - output volume variable \c field voxel count, Nx*Ny*Nz*Ns where Ns=cfg.srcnum is the pattern number for photon sharing */
     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z*((cfg->srctype==MCX_SRC_PATTERN || cfg->srctype==MCX_SRC_PATTERN3D) ? cfg->srcnum : 1);

     /** \c media - input volume representing the simulation domain, format specified in cfg.mediaformat, read-only */
     uint  *media=(uint *)(cfg->vol);
     
     /** \c field - output volume to store GPU computed fluence, length is \c dimxyz */
     float  *field;

     /** \c rfimag - imaginary part of the RF Jacobian, length is \c dimxyz */
     OutputType  *rfimag=NULL;
     
     /** \c Ppos - per-thread photon state initialization host buffers */
     float4 *Ppos,*Pdir,*Plen,*Plen0;
     
     /** \c Pseed - per-thread RNG seed initialization host buffers */
     uint   *Pseed;
     
     /** \c Pdet - output host buffer to store detected photon partial-path and other per-photon data */
     float  *Pdet;

     /** \c energy - output host buffers to accummulate total launched/absorbed energy per thread, needed for normalization */
     float  *energy;

     /** \c srcpw, \c energytot, \c energyabs - output host buffers to accummulate total launched/absorbed energy per pattern in photon sharing, needed for normalization of multi-pattern simulations */
     float  *srcpw=NULL,*energytot=NULL,*energyabs=NULL; // for multi-srcpattern
     
     /** \c seeddata - output buffer to store RNG initial seeds for each detected photon for replay */
     RandType *seeddata=NULL;

     /** \c detected - total number of detected photons, output */
     uint    detected=0;

     /** \c progress - pinned-memory variable as the progress bar during simulation, updated in GPU and visible from host */
     volatile int *progress;

#ifdef _WIN32
     /** \c updateprogress - CUDA event needed to avoid hanging on Windows, see https://forums.developer.nvidia.com/t/solved-how-to-update-host-memory-variable-from-device-during-opencl-kernel-execution/59409/5 */
     cudaEvent_t updateprogress;
#endif

     /** all pointers start with g___ are the corresponding GPU buffers to read/write host variables defined above */
     uint *gmedia;
     float4 *gPpos,*gPdir,*gPlen,*gsmatrix;
     uint   *gPseed,*gdetected;
     int    *greplaydetid=NULL;
     float  *gPdet,*gsrcpattern=NULL,*genergy,*greplayw=NULL,*greplaytof=NULL,*gdebugdata=NULL,*ginvcdf=NULL;
     OutputType *gfield;
     RandType *gseeddata=NULL;
     volatile int *gprogress;

     /**
      *                            |----------------------------------------------->  hostdetreclen  <--------------------------------------|
      *                                      |------------------------>    partialdata   <-------------------|                               
      *host detected photon buffer: detid (1), partial_scat (#media), partial_path (#media), momontum (#media), p_exit (3), v_exit(3), w0 (1) 
      *                                      |--------------------------------------------->    w0offset   <-------------------------------------||<----- w0 (#srcnum) ----->||<- RF replay (2)->|
      *gpu detected photon buffer:            partial_scat (#media), partial_path (#media), momontum (#media), E_escape (1), E_launch (1), w0 (1), w0_photonsharing (#srcnum)   cos(w*T),sin(w*T)
      */

     //< \c partialdata: per-photon buffer length for media-specific data, copy from GPU to host
     unsigned int partialdata=(cfg->medianum-1)*(SAVE_NSCAT(cfg->savedetflag)+SAVE_PPATH(cfg->savedetflag)+SAVE_MOM(cfg->savedetflag)); 
     
     //< \c w0offset - offset in the per-photon buffer to the start of the photon sharing related data
     unsigned int w0offset=partialdata+3;
     
     //< \c hostdetreclen - host-side det photon data buffer per-photon length
     unsigned int hostdetreclen=partialdata+SAVE_DETID(cfg->savedetflag)+3*(SAVE_PEXIT(cfg->savedetflag)+SAVE_VEXIT(cfg->savedetflag))+SAVE_W0(cfg->savedetflag)+4*SAVE_IQUV(cfg->savedetflag);
     
     //< \c is2d - flag to tell mcx if the simulation domain is 2D, set to 1 if any of the x/y/z dimensions has a length of 1
     unsigned int is2d=(cfg->dim.x==1 ? 1 : (cfg->dim.y==1 ? 2 : (cfg->dim.z==1 ? 3 : 0)));

     /** \c param - constants to be used in the GPU, copied to GPU as \c gcfg, stored in the constant memory */
     MCXParam param={cfg->steps,minstep,0,0,cfg->tend,R_C0*cfg->unitinmm,
                     (uint)cfg->issave2pt,(uint)cfg->isreflect,(uint)cfg->isrefint,(uint)cfg->issavedet,1.f/cfg->tstep,
		     p0,c0,s0,maxidx,uint4(0,0,0,0),cp0,cp1,uint2(0,0),cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->srctype,
		     cfg->srcparam1,cfg->srcparam2,cfg->voidtime,cfg->maxdetphoton,
		     cfg->medianum-1,cfg->detnum,cfg->polmedianum,cfg->maxgate,0,0,ABS(cfg->sradius+2.f)<EPS /*isatomic*/,
		     (uint)cfg->maxvoidstep,cfg->issaveseed>0,(uint)cfg->issaveref,cfg->isspecular>0,
		     cfg->maxdetphoton*hostdetreclen,cfg->seed,(uint)cfg->outputtype,0,0,cfg->faststep,
		     cfg->debuglevel,cfg->savedetflag,hostdetreclen,partialdata,w0offset,cfg->mediabyte,
		     (uint)cfg->maxjumpdebug,cfg->gscatter,is2d,cfg->replaydet,cfg->srcnum,cfg->nphase,cfg->omega};
     if(param.isatomic)
         param.skipradius2=0.f;
	 
     /** Start multiple CPU threads using OpenMP, one thread for each GPU device to run simultaneously, \c threadid returns the current thread ID */
#ifdef _OPENMP
     threadid=omp_get_thread_num();
#endif
     if(threadid<MAX_DEVICE && cfg->deviceid[threadid]=='\0')
           return;

     /** Use \c threadid and cfg.deviceid, the compressed list of active GPU devices, to get the desired GPU ID for this thread */
     gpuid=cfg->deviceid[threadid]-1;
     if(gpuid<0)
          mcx_error(-1,"GPU ID must be non-zero",__FILE__,__LINE__);
	  
     /** Activate the corresponding GPU device */
     CUDA_ASSERT(cudaSetDevice(gpuid));

     /** Use the specified GPU's parameters, stored in gpu[gpuid] to determine the maximum time gates that it can hold */
     if(gpu[gpuid].maxgate==0 && dimxyz>0){
         int needmem=dimxyz+cfg->nthread*sizeof(float4)*4+sizeof(float)*cfg->maxdetphoton*hostdetreclen+10*1024*1024; /*keep 10M for other things*/
         gpu[gpuid].maxgate=(gpu[gpuid].globalmem-needmem)/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
         gpu[gpuid].maxgate=MIN(((cfg->tend-cfg->tstart)/cfg->tstep+0.5),gpu[gpuid].maxgate);     
     }
     /** Updating host simulation configuration \c cfg, only allow the master thread to modify cfg, others are read-only */
#pragma omp master
{
     if(cfg->exportfield==NULL){
         if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
	     cfg->exportfield=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*(1+(cfg->outputtype==otRF))*cfg->detnum);
	 }else{
             cfg->exportfield=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*(1+(cfg->outputtype==otRF)));
	 }
     }
     if(cfg->exportdetected==NULL)
         cfg->exportdetected=(float*)malloc(hostdetreclen*cfg->maxdetphoton*sizeof(float));
     if(cfg->issaveseed && cfg->seeddata==NULL)
         cfg->seeddata=malloc(cfg->maxdetphoton*sizeof(RandType)*RAND_BUF_LEN);
     cfg->detectedcount=0;
     cfg->his.detected=0;
     cfg->his.respin=cfg->respin;
     cfg->his.colcount=hostdetreclen;
     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     cfg->runtime=0;
}
#pragma omp barrier

     /** If domain is 2D, the initial photon launch direction in the singleton dimension is expected to be 0 */
     if(is2d){
         float *vec=&(param.c0.x);
         if(ABS(vec[is2d-1])>EPS)
             mcx_error(-1,"input domain is 2D, the initial direction can not have non-zero value in the singular dimension",__FILE__,__LINE__);
     }

     /** If autopilot mode is not used, the thread and block sizes are determined by user input from cfg.nthread and cfg.nblocksize */
     if(!cfg->autopilot){
	uint gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
	gpu[gpuid].autothread=cfg->nthread;
	gpu[gpuid].autoblock=cfg->nblocksize;
	if(cfg->maxgate==0)
	    cfg->maxgate=gates;
	else if(cfg->maxgate>gates)
	    cfg->maxgate=gates;
	gpu[gpuid].maxgate=cfg->maxgate;
     }
     
     /** If total thread number is not integer multiples of block size, round it to the largest block size multiple */
     if(gpu[gpuid].autothread%gpu[gpuid].autoblock)
     	gpu[gpuid].autothread=(gpu[gpuid].autothread/gpu[gpuid].autoblock)*gpu[gpuid].autoblock;

     param.maxgate=gpu[gpuid].maxgate;

     /** If cfg.respin is positive, the output data have to be accummulated, so we use a double-buffer to retrieve and then accummulate */
     if(ABS(cfg->respin)>1){
         if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
             field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*2*cfg->detnum);
	 }else{
             field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*2);	 
	 }
     }else{
         if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
             field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*cfg->detnum); //the second half will be used to accumulate
	 }else{
             field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate); //the second half will be used to accumulate
	 }
     }

#pragma omp master
{
     /** Master thread computes total workloads, specified by users (or equally by default) for all active devices, stored as cfg.workload[gpuid] */
     fullload=0.f;
     for(i=0;cfg->deviceid[i];i++)
        fullload+=cfg->workload[i];

     if(fullload<EPS){
        for(i=0;cfg->deviceid[i];i++)
            cfg->workload[i]=gpu[cfg->deviceid[i]-1].core;
     }
}
#pragma omp barrier

     fullload=0.f;
     for(i=0;cfg->deviceid[i];i++)
        if(cfg->workload[i]>0.f)
            fullload+=cfg->workload[i];
        else
            mcx_error(-1,"workload was unspecified for an active device",__FILE__,__LINE__);

     /** Now we can determine how many photons to be simualated by multiplying the total photon by the relative ratio of per-device workload divided by the total workload */
     gpuphoton=(double)cfg->nphoton*cfg->workload[threadid]/fullload;

     if(gpuphoton==0)
        return;

     /** Once per-thread photon number \c param.threadphoton is known, we distribute the remaiders, if any, to a subset of threads, one extra photon per thread, so we can get the exact total photon number */
     if(cfg->respin>=1){
         param.threadphoton=gpuphoton/gpu[gpuid].autothread;
         param.oddphotons=gpuphoton-param.threadphoton*gpu[gpuid].autothread;
     }else if(cfg->respin<0){
         param.threadphoton=-gpuphoton/gpu[gpuid].autothread/cfg->respin;
         param.oddphotons=-gpuphoton/cfg->respin-param.threadphoton*gpu[gpuid].autothread;     
     }else{
         mcx_error(-1,"respin number can not be 0, check your -r/--repeat input or cfg.respin value",__FILE__,__LINE__);
     }
     
     /** Total time gate number is computed */
     totalgates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
#pragma omp master

     /** Here we determine if the GPU memory of the current device can store all time gates, if not, disabling normalization */
     if(totalgates>gpu[gpuid].maxgate && cfg->isnormalized){
         MCX_FPRINTF(stderr,S_RED "WARNING: GPU memory can not hold all time gates, disabling normalization to allow multiple runs\n" S_RESET);
         cfg->isnormalized=0;
     }
#pragma omp barrier

     /** Here we decide the total output buffer, field's length. it is Nx*Ny*Nz*Nt*Ns */
     if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1)
         fieldlen=dimxyz*gpu[gpuid].maxgate*cfg->detnum;
     else
         fieldlen=dimxyz*gpu[gpuid].maxgate;

     /** A 1D grid is determined by the total thread number and block size */
     mcgrid.x=gpu[gpuid].autothread/gpu[gpuid].autoblock;
     
     /** A 1D block is determined by the user specified block size, or by default, 64, determined emperically to get best performance */
     mcblock.x=gpu[gpuid].autoblock;

     /** If "-D R" is used, we launch a special RNG testing kernel to retrieve random numbers from RNG and test for randomness */
     if(cfg->debuglevel & MCX_DEBUG_RNG){
#pragma omp master
{
           param.twin0=cfg->tstart;
           param.twin1=cfg->tend;
           Pseed=(uint*)malloc(sizeof(RandType)*RAND_BUF_LEN);
           for (i=0; i<(int)(((sizeof(RandType)*RAND_BUF_LEN)>>2)); i++){
		Pseed[i]=((rand() << 16) | (rand() << 1) | (rand() >> 14));
	   }
           CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(RandType)*RAND_BUF_LEN));
	   CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(RandType)*RAND_BUF_LEN,  cudaMemcpyHostToDevice));
           CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(OutputType)*fieldlen));
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(OutputType)*fieldlen)); // cost about 1 ms
           CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

           tic=StartTimer();
           MCX_FPRINTF(cfg->flog,"generating %lu random numbers ... \t",fieldlen); fflush(cfg->flog);
           mcx_test_rng<<<1,1>>>(gfield,gPseed);
           tic1=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving random numbers ... \t",tic1-tic);
           CUDA_ASSERT(cudaGetLastError());

           CUDA_ASSERT(cudaMemcpy(field, gfield,sizeof(OutputType)*dimxyz*gpu[gpuid].maxgate,cudaMemcpyDeviceToHost));
           MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	   if(cfg->exportfield){
	       memcpy(cfg->exportfield,field,fieldlen*sizeof(float));
	   }
	   if(cfg->issave2pt && cfg->parentid==mpStandalone){
               MCX_FPRINTF(cfg->flog,"saving data to file ...\t");
	       mcx_savedata(field,fieldlen,cfg);
               MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
               fflush(cfg->flog);
           }
	   CUDA_ASSERT(cudaFree(gfield));
	   CUDA_ASSERT(cudaFree(gPseed));
	   free(field);
	   free(Pseed);

           CUDA_ASSERT(cudaDeviceReset());
}
#pragma omp barrier

	   return;
     }

     /** 
       * Allocate all host buffers to store input or output data 
       */

     Ppos=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread); /** \c Ppos: host buffer for initial photon position+weight */
     Pdir=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread); /** \c Pdir: host buffer for initial photon direction */
     Plen=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread); /** \c Plen: host buffer for initial additional photon states */
     Plen0=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread);
     energy=(float*)calloc(gpu[gpuid].autothread<<1,sizeof(float)); /** \c energy: host buffer for retrieving total launched and escaped energy of each thread */
     Pdet=(float*)calloc(cfg->maxdetphoton,sizeof(float)*(hostdetreclen)); /** \c Pdet: host buffer for retrieving all detected photon information */
     if(cfg->seed!=SEED_FROM_FILE)
         Pseed=(uint*)malloc(sizeof(RandType)*gpu[gpuid].autothread*RAND_BUF_LEN); /** \c Pseed: RNG seed for each thread in non-replay mode, or */
     else
         Pseed=(uint*)malloc(sizeof(RandType)*cfg->nphoton*RAND_BUF_LEN); /** \c Pseed: RNG seeds for photon replay in GPU threads */

     /** 
       * Allocate all GPU buffers to store input or output data
       */
     if(cfg->mediabyte!=MEDIA_2LABEL_SPLIT)
         CUDA_ASSERT(cudaMalloc((void **) &gmedia, sizeof(uint)*(cfg->dim.x*cfg->dim.y*cfg->dim.z)));
     else
         CUDA_ASSERT(cudaMalloc((void **) &gmedia, sizeof(uint)*(2*cfg->dim.x*cfg->dim.y*cfg->dim.z)));
     //CUDA_ASSERT(cudaBindTexture(0, texmedia, gmedia));
     CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(OutputType)*fieldlen*SHADOWCOUNT));
     CUDA_ASSERT(cudaMalloc((void **) &gPpos, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdir, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPlen, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdet, sizeof(float)*cfg->maxdetphoton*(hostdetreclen)));
     CUDA_ASSERT(cudaMalloc((void **) &gdetected, sizeof(uint)));
     CUDA_ASSERT(cudaMalloc((void **) &genergy, sizeof(float)*(gpu[gpuid].autothread<<1)));

     /** 
       * Allocate pinned memory variable, progress, for real-time update during kernel run-time
       */
     CUDA_ASSERT(cudaHostAlloc((void **)&progress, sizeof(int), cudaHostAllocMapped));
     CUDA_ASSERT(cudaHostGetDevicePointer((int **)&gprogress, (int *)progress, 0));
     *progress = 0;

     if(cfg->debuglevel & MCX_DEBUG_MOVE){
         CUDA_ASSERT(cudaMalloc((void **) &gdebugdata, sizeof(float)*(debuglen*cfg->maxjumpdebug)));
     }
     if(cfg->issaveseed){
         seeddata=(RandType*)malloc(sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN);
	 CUDA_ASSERT(cudaMalloc((void **) &gseeddata, sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN));
     }
     if(cfg->nphase){
         CUDA_ASSERT(cudaMalloc((void **) &ginvcdf, sizeof(float)*cfg->nphase));
	 CUDA_ASSERT(cudaMemcpy(ginvcdf,cfg->invcdf,sizeof(float)*cfg->nphase, cudaMemcpyHostToDevice));
     }
     if(cfg->polmedianum){
         CUDA_ASSERT(cudaMalloc((void **) &gsmatrix, cfg->polmedianum*NANGLES*sizeof(float4)));
	 CUDA_ASSERT(cudaMemcpy(gsmatrix, cfg->smatrix, cfg->polmedianum*NANGLES*sizeof(float4), cudaMemcpyHostToDevice));
     }
     /** 
       * Allocate and copy data needed for photon replay, the needed variables include
       * \c gPseed per-photon seed to be replayed
       * \c greplayw per-photon initial weight
       * \c greplaytof per-photon time-of-flight time in s
       * \c greplaydetid per-photon index for each replayed photon
       */
     if(cfg->seed==SEED_FROM_FILE){
         CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(RandType)*cfg->nphoton*RAND_BUF_LEN));
	 CUDA_ASSERT(cudaMemcpy(gPseed,cfg->replay.seed,sizeof(RandType)*cfg->nphoton*RAND_BUF_LEN, cudaMemcpyHostToDevice));
	 if(cfg->replay.weight){
	     CUDA_ASSERT(cudaMalloc((void **) &greplayw, sizeof(float)*cfg->nphoton));
	     CUDA_ASSERT(cudaMemcpy(greplayw,cfg->replay.weight,sizeof(float)*cfg->nphoton, cudaMemcpyHostToDevice));
	 }
         if(cfg->replay.tof){
	     CUDA_ASSERT(cudaMalloc((void **) &greplaytof, sizeof(float)*cfg->nphoton));
	     CUDA_ASSERT(cudaMemcpy(greplaytof,cfg->replay.tof,sizeof(float)*cfg->nphoton, cudaMemcpyHostToDevice));
	 }
         if(cfg->replay.detid){
	     CUDA_ASSERT(cudaMalloc((void **) &greplaydetid, sizeof(int)*cfg->nphoton));
	     CUDA_ASSERT(cudaMemcpy(greplaydetid,cfg->replay.detid,sizeof(int)*cfg->nphoton, cudaMemcpyHostToDevice));
	 }
     }else
         CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(RandType)*gpu[gpuid].autothread*RAND_BUF_LEN));

     /** 
       * Allocate and copy source pattern buffer for 2D and 3D pattern sources
       */
     if(cfg->srctype==MCX_SRC_PATTERN)
         CUDA_ASSERT(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w*cfg->srcnum)));
     else if(cfg->srctype==MCX_SRC_PATTERN3D)
         CUDA_ASSERT(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z*cfg->srcnum)));
	 
#ifndef SAVE_DETECTORS
#pragma omp master
     /** 
       * Saving detected photon is enabled by default, but in case if a user disabled this feature, a warning is printed
       */
     if(cfg->issavedet){
           MCX_FPRINTF(stderr,S_RED "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc\n" S_RESET);
           cfg->issavedet=0;
     }
#pragma omp barrier
#endif

     /** 
       * Pre-compute array dimension strides to move in +-x/y/z dimension quickly in the GPU, stored in the constant memory
       */
     /** Inside the GPU kernel, volume is always assumbed to be col-major (like those generated by MATLAB or FORTRAN) */
     cachebox.x=(cp1.x-cp0.x+1);
     cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
     dimlen.x=cfg->dim.x;
     dimlen.y=cfg->dim.y*cfg->dim.x;

     dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     dimlen.w=fieldlen;

     param.dimlen=dimlen;
     param.cachebox=cachebox;

     /** 
       * Additional constants to avoid repeated computation inside GPU
       */
     if(p0.x<0.f || p0.y<0.f || p0.z<0.f || p0.x>=cfg->dim.x || p0.y>=cfg->dim.y || p0.z>=cfg->dim.z){
         param.idx1dorig=0;
         param.mediaidorig=0;
     }else{
         param.idx1dorig=(int(floorf(p0.z))*dimlen.y+int(floorf(p0.y))*dimlen.x+int(floorf(p0.x)));
         param.mediaidorig=(cfg->vol[param.idx1dorig] & MED_MASK);
     }
     memcpy(&(param.bc),cfg->bc,12);
     Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z; /*Vvox: voxel volume in mm^3*/
     if(cfg->seed>0)
     	srand(cfg->seed+threadid);
     else
        srand(time(0));

     for (i=0; i<gpu[gpuid].autothread; i++) {
	   Ppos[i]=p0;  // initial position
           Pdir[i]=c0;
           Plen[i]=float4(0.f,0.f,param.minaccumtime,0.f);
     }

     /** 
       * Get ready to start GPU simulation here, clock is now ticking ...
       */
     tic=StartTimer();
#pragma omp master
{
     mcx_printheader(cfg);

#ifdef MCX_TARGET_NAME
     MCX_FPRINTF(cfg->flog,"- variant name: [%s] compiled by nvcc [%d.%d] with CUDA [%d]\n",
         "Fermi",__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,CUDART_VERSION);
#else
     MCX_FPRINTF(cfg->flog,"- code name: [Vanilla MCX] compiled by nvcc [%d.%d] with CUDA [%d]\n",
         __CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,CUDART_VERSION);
#endif
     MCX_FPRINTF(cfg->flog,"- compiled with: RNG [%s] with Seed Length [%d]\n",MCX_RNG_NAME,(int)((sizeof(RandType)*RAND_BUF_LEN)>>2));
     fflush(cfg->flog);
}
#pragma omp barrier

     /** 
       * Copy all host buffers to the GPU
       */
     MCX_FPRINTF(cfg->flog,"\nGPU=%d (%s) threadph=%d extra=%d np=%ld nthread=%d maxgate=%d repetition=%d\n",gpuid+1,gpu[gpuid].name,param.threadphoton,param.oddphotons,
           gpuphoton,gpu[gpuid].autothread,gpu[gpuid].maxgate,ABS(cfg->respin));
     MCX_FPRINTF(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);

     mcx_flush(cfg);

     if(cfg->mediabyte!=MEDIA_2LABEL_SPLIT)
         CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uint)*cfg->dim.x*cfg->dim.y*cfg->dim.z, cudaMemcpyHostToDevice));
     else
         CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uint)*2*cfg->dim.x*cfg->dim.y*cfg->dim.z, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpy(genergy,energy,sizeof(float) *(gpu[gpuid].autothread<<1), cudaMemcpyHostToDevice));
     if(cfg->srcpattern)
        if(cfg->srctype==MCX_SRC_PATTERN)
           CUDA_ASSERT(cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w*cfg->srcnum), cudaMemcpyHostToDevice));
	else if(cfg->srctype==MCX_SRC_PATTERN3D)
	   CUDA_ASSERT(cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z*cfg->srcnum), cudaMemcpyHostToDevice));
     
     /** 
       * Copy constants to the constant memory on the GPU
       */
     CUDA_ASSERT(cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpyToSymbol(gproperty, cfg->detpos,  cfg->detnum*sizeof(float4), cfg->medianum*sizeof(Medium), cudaMemcpyHostToDevice));

     MCX_FPRINTF(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

     /**
         If one has to simulate a lot of time gates, using the GPU global memory
	 requires extra caution. If the total global memory is bigger than the total
	 memory to save all the snapshots, i.e. size(field)*(tend-tstart)/tstep, one
	 simply sets gpu[gpuid].maxgate to the total gate number; this will run GPU kernel
	 once. If the required memory is bigger than the video memory, set gpu[gpuid].maxgate
	 to a number which fits, and the snapshot will be saved with an increment of 
	 gpu[gpuid].maxgate snapshots. In this case, the later simulations will restart from
	 photon launching and exhibit redundancies.

	 The calculation of the energy conservation will only reflect the last simulation.
     */
     sharedbuf=cfg->nphase*sizeof(float)+gpu[gpuid].autoblock*(cfg->issaveseed*(RAND_BUF_LEN*sizeof(RandType))+sizeof(float)*(param.w0offset+cfg->srcnum+2*(cfg->outputtype==otRF)));

     MCX_FPRINTF(cfg->flog,"requesting %d bytes of shared memory\n",sharedbuf);

     /** 
       * Outer loop: loop over each time-gate-group, determined by the capacity of the global memory to hold the output data, in most cases, \c totalgates is 1
       */
     for(timegate=0;timegate<totalgates;timegate+=gpu[gpuid].maxgate){

       /** Determine the start and end time of the current time-gate-group */
       param.twin0=cfg->tstart+cfg->tstep*timegate;
       param.twin1=param.twin0+cfg->tstep*gpu[gpuid].maxgate;

       /** Copy param to the constant memory variable gcfg */
       CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param,     sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

       MCX_FPRINTF(cfg->flog,S_CYAN"launching MCX simulation for time window [%.2ens %.2ens] ...\n" S_RESET
           ,param.twin0*1e9,param.twin1*1e9);

       /**
         * Inner loop: loop over total number of repetitions specified by cfg.respin, results will be accumulated to \c field
         */
       for(iter=0;iter<ABS(cfg->respin);iter++){
           /**
             * Each repetition, we have to reset the output buffers, including \c gfield and \c gPdet
             */
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(OutputType)*fieldlen*SHADOWCOUNT)); // cost about 1 ms
           CUDA_ASSERT(cudaMemset(gPdet,0,sizeof(float)*cfg->maxdetphoton*(hostdetreclen)));
           if(cfg->issaveseed)
	       CUDA_ASSERT(cudaMemset(gseeddata,0,sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN));
           CUDA_ASSERT(cudaMemset(gdetected,0,sizeof(float)));
           if(cfg->debuglevel & MCX_DEBUG_MOVE){
	       uint jumpcount=0;
               CUDA_ASSERT(cudaMemcpyToSymbol(gjumpdebug, &jumpcount, sizeof(uint), 0, cudaMemcpyHostToDevice));
           }
 	   CUDA_ASSERT(cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));
	   CUDA_ASSERT(cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));
	   CUDA_ASSERT(cudaMemcpy(gPlen,  Plen,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));

           if(cfg->seed!=SEED_FROM_FILE){
             for (i=0; i<gpu[gpuid].autothread*((int)(sizeof(RandType)*RAND_BUF_LEN)>>2); i++)
               Pseed[i]=((rand() << 16) | (rand() << 1) | (rand() >> 14));
	     CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(RandType)*gpu[gpuid].autothread*RAND_BUF_LEN,  cudaMemcpyHostToDevice));
           }
           /**
             * Start the clock for GPU-kernel only run-time here
             */
           tic0=GetTimeMillis();
#ifdef _WIN32
#pragma omp master
{
           /**
             * To avoid hanging, we need to use cudaEvent to force GPU to update the pinned memory for progress bar on Windows WHQL driver
             */
           if(cfg->debuglevel & MCX_DEBUG_PROGRESS)
               CUDA_ASSERT(cudaEventCreate(&updateprogress));
}
#endif
           MCX_FPRINTF(cfg->flog,"simulation run#%2d ... \n",iter+1); fflush(cfg->flog);
           mcx_flush(cfg);

           /**
             * Determine template constants for compilers to build specialized binary instances to reduce branching 
	     * and thread-divergence. If not using template, the performance can take a 20% drop.
             */

	   /** \c ispencil: template constant, if 1, launch photon code is dramatically simplified */
           int ispencil=(cfg->srctype==MCX_SRC_PENCIL);
	   
	   /** \c isref: template constant, if 1, perform boundary reflection, if 0, total-absorbion boundary, can simplify kernel */
	   int isref=cfg->isreflect;
	   
	   /** \c issvmc: template constant, if 1, consider the input volume containing split-voxel data, see Yan2020 for details */
	   int issvmc=(cfg->mediabyte==MEDIA_2LABEL_SPLIT);

	   /** Enable reflection flag when c or m flags are used in the cfg.bc boundary condition flags */
	   for(i=0;i<6;i++)
	       if(cfg->bc[i]==bcReflect || cfg->bc[i]==bcMirror)
	           isref=1;
           /**
             * Launch GPU kernel using template constants. Here, the compiler will create 2^4=16 individually compiled
	     * kernel PTX binaries for each combination of template variables. This creates bigger binary and slower 
	     * compilation time, but brings up to 20%-30% speed improvement on certain simulations.
             */
	   switch(ispencil*1000 + (isref>0)*100 + (cfg->mediabyte<=4)*10 + issvmc){
	       case 0:   mcx_main_loop<0,0,0,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1:   mcx_main_loop<0,0,0,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 10:  mcx_main_loop<0,0,1,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 11:  mcx_main_loop<0,0,1,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 100: mcx_main_loop<0,1,0,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 101: mcx_main_loop<0,1,0,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 110: mcx_main_loop<0,1,1,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 111: mcx_main_loop<0,1,1,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1000:mcx_main_loop<1,0,0,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1001:mcx_main_loop<1,0,0,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1010:mcx_main_loop<1,0,1,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1011:mcx_main_loop<1,0,1,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1100:mcx_main_loop<1,1,0,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1101:mcx_main_loop<1,1,0,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1110:mcx_main_loop<1,1,1,0> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	       case 1111:mcx_main_loop<1,1,1,1> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,ginvcdf,gsmatrix,gprogress);break;
	   }
#pragma omp master
{
           /**
             * By now, the GPU kernel has been launched asynchronously, the master thread on the host starts
	     * reading a pinned memory variable, \c gprogress, to realtimely read the completed photon count
	     * updated inside the GPU kernel while it is running.
             */
           if((param.debuglevel & MCX_DEBUG_PROGRESS)){
	     int p0 = 0, ndone=-1;
#ifdef _WIN32
             CUDA_ASSERT(cudaEventRecord(updateprogress));
#endif
	     mcx_progressbar(-0.f,cfg);
	     do{
#ifdef _WIN32
               cudaEventQuery(updateprogress);
#endif
               /**
                 * host variable \c progress is pinned with the GPU variable \c gprogress, and can be
		 * updated by the GPU kernel from the device. We can read this variable to see how many
		 * photons are simulated.
                 */
               ndone = *progress;
	       if (ndone > p0){
                  /**
                    * Here we use the below formula to compute the 0-100% completion ratio. 
		    * Only half of the threads updates the progress, and each thread only update
		    * the counter 5 times at 0%/25%/50%/75%/100% progress to minimize overhead while
		    * still providing a smooth progress bar.
                    */
		  mcx_progressbar(ndone/((param.threadphoton>>1)*4.5f),cfg);
		  p0 = ndone;
	       }
               sleep_ms(100);
	     }while (p0 < (param.threadphoton>>1)*4.5f);
             mcx_progressbar(1.0f,cfg);
             MCX_FPRINTF(cfg->flog,"\n");
             *progress=0;
           }
}
           /**
             * By calling \c cudaDeviceSynchronize, the host thread now waits for the completion of
	     * the kernel, then start retrieving all GPU output data
             */
           CUDA_ASSERT(cudaDeviceSynchronize());
           /** Here, the GPU kernel is completely executed and returned */
	   CUDA_ASSERT(cudaMemcpy(&detected, gdetected,sizeof(uint),cudaMemcpyDeviceToHost));
	   
	   /** now we can estimate and print the GPU-kernel-only runtime */
           tic1=GetTimeMillis();
	   toc+=tic1-tic0;
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",tic1-tic);

	   /**
	     * If the GPU kernel crashed or terminated by error during execution, we need
	     * to capture it by calling \c cudaGetLastError and terminate mcx if error happens
	     */
           CUDA_ASSERT(cudaGetLastError());

	   /**
	     * Now, we start retrieving all output variables, and copy those to the corresponding host buffers
	     */

           /** \c photoncount returns the actual completely simulated photons returned by GPU threads, no longer used */
           CUDA_ASSERT(cudaMemcpy(Plen0,  gPlen,  sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
           for(i=0;i<gpu[gpuid].autothread;i++)
	      photoncount+=int(Plen0[i].w+0.5f);

	   /**
	     * If '-D M' is specified, we retrieve photon trajectory data and store those to \c cfg.exportdebugdata and \c cfg.debugdatalen
	     */
           if(cfg->debuglevel & MCX_DEBUG_MOVE){
               uint debugrec=0;
	       CUDA_ASSERT(cudaMemcpyFromSymbol(&debugrec, gjumpdebug,sizeof(uint),0,cudaMemcpyDeviceToHost));
#pragma omp critical
{
	       if(debugrec>0){
		   if(debugrec>cfg->maxjumpdebug){
			MCX_FPRINTF(cfg->flog,S_RED "WARNING: the saved trajectory positions (%d) \
are more than what your have specified (%d), please use the --maxjumpdebug option to specify a greater number\n" S_RESET
                           ,debugrec,cfg->maxjumpdebug);
		   }else{
			MCX_FPRINTF(cfg->flog,"saved %u trajectory positions, total: %d\t",debugrec,cfg->debugdatalen+debugrec);
		   }
                   debugrec=min(debugrec,cfg->maxjumpdebug);
	           cfg->exportdebugdata=(float*)realloc(cfg->exportdebugdata,(cfg->debugdatalen+debugrec)*debuglen*sizeof(float));
                   CUDA_ASSERT(cudaMemcpy(cfg->exportdebugdata+cfg->debugdatalen, gdebugdata,sizeof(float)*debuglen*debugrec,cudaMemcpyDeviceToHost));
                   cfg->debugdatalen+=debugrec;
	       }
}
           }

	   /**
	     * If photon detection is enabled and detectors are defined, we retrieve partial-path length data, among others, to \c cfg.exportdetected and \c detected
	     */
#ifdef SAVE_DETECTORS
           if(cfg->issavedet){
           	CUDA_ASSERT(cudaMemcpy(Pdet, gPdet,sizeof(float)*cfg->maxdetphoton*(hostdetreclen),cudaMemcpyDeviceToHost));
	        CUDA_ASSERT(cudaGetLastError());
	        /**
	          * If photon seeds are needed for replay, here we retrieve the seed data
	          */
		if(cfg->issaveseed)
		    CUDA_ASSERT(cudaMemcpy(seeddata, gseeddata,sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN,cudaMemcpyDeviceToHost));
		if(detected>cfg->maxdetphoton){
			MCX_FPRINTF(cfg->flog,S_RED "WARNING: the detected photon (%d) \
is more than what your have specified (%d), please use the -H option to specify a greater number\t" S_RESET
                           ,detected,cfg->maxdetphoton);
		}else{
			MCX_FPRINTF(cfg->flog,"detected " S_BOLD "" S_BLUE "%d photons" S_RESET", total: " S_BOLD "" S_BLUE "%ld" S_RESET"\t",detected,cfg->detectedcount+detected);
		}

	        /**
	          * The detected photon dat retrieved from each thread/device are now concatenated to store in a single host buffer
	          */
#pragma omp atomic
                cfg->his.detected+=detected;
                detected=MIN(detected,cfg->maxdetphoton);
		if(cfg->exportdetected){
#pragma omp critical
{
                        cfg->exportdetected=(float*)realloc(cfg->exportdetected,(cfg->detectedcount+detected)*hostdetreclen*sizeof(float));
			if(cfg->issaveseed && cfg->seeddata)
			    cfg->seeddata=(RandType*)realloc(cfg->seeddata,(cfg->detectedcount+detected)*sizeof(RandType)*RAND_BUF_LEN);
	                memcpy(cfg->exportdetected+cfg->detectedcount*(hostdetreclen),Pdet,detected*(hostdetreclen)*sizeof(float));
			if(cfg->issaveseed && cfg->seeddata)
			    memcpy(((RandType*)cfg->seeddata)+cfg->detectedcount*RAND_BUF_LEN,seeddata,detected*sizeof(RandType)*RAND_BUF_LEN);
                        cfg->detectedcount+=detected;
}
		}
	   }
#endif
           mcx_flush(cfg);

	   /**
	     * Accumulate volumetric fluence from all threads/devices
	     */
           if(cfg->issave2pt){
	       OutputType *rawfield=(OutputType*)malloc(sizeof(OutputType)*fieldlen*SHADOWCOUNT);
               CUDA_ASSERT(cudaMemcpy(rawfield, gfield,sizeof(OutputType)*fieldlen*SHADOWCOUNT,cudaMemcpyDeviceToHost));
               MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	       /**
	         * If double-precision is used for output, we do not need two buffers; however, by default, we use
		 * single-precision output, we need to copy and accumulate two separate floating-point buffers
		 * to minimize round-off errors near the source
	         */
	       for(i=0;i<(int)fieldlen;i++){  //accumulate field, can be done in the GPU
	           field[i]=rawfield[i];
#ifndef USE_DOUBLE
                   if(cfg->outputtype!=otRF)
                       field[i]+=rawfield[i+fieldlen];
#endif
	       }
	       if(cfg->outputtype==otRF && cfg->omega>0.f && SHADOWCOUNT==2){
	           rfimag=(OutputType*)malloc(fieldlen*sizeof(OutputType));
	           memcpy(rfimag, rawfield+fieldlen, fieldlen*sizeof(OutputType));
	       }
	       free(rawfield);
	       /**
	         * If respin is used, each repeatition is accumulated to the 2nd half of the buffer
	         */
               if(ABS(cfg->respin)>1){
                   for(i=0;i<(int)fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
           }
       } /** Here is the end of the inner-loop (respin) */

#pragma omp critical
       if(cfg->runtime<toc)
           cfg->runtime=toc;

       /**
	 * If respin is used, copy the accumulated buffer in the 2nd half to the first half
	 */
       if(ABS(cfg->respin)>1)  //copy the accumulated fields back
           memcpy(field,field+fieldlen,sizeof(float)*fieldlen);

       CUDA_ASSERT(cudaMemcpy(energy,genergy,sizeof(float)*(gpu[gpuid].autothread<<1),cudaMemcpyDeviceToHost));
#pragma omp critical
{
       /**
	 * Retrieve accumulated total launched and residual energy from each thread
	 */
       for(i=0;i<gpu[gpuid].autothread;i++){
           cfg->energyesc+=energy[i<<1];
           cfg->energytot+=energy[(i<<1)+1];
       }
       for(i=0;i<gpu[gpuid].autothread;i++)
           cfg->energyabs+=Plen0[i].z;  // the accumulative absorpted energy near the source
}
       /**
	 * For MATLAB mex file, the data is copied to a pre-allocated buffer \c cfg->export* as a return variable
	 */
       if(cfg->exportfield){
	       for(i=0;i<(int)fieldlen;i++)
#pragma omp atomic
                   cfg->exportfield[i]+=field[i];
               if(cfg->outputtype==otRF && rfimag){
	           for(i=0;i<(int)fieldlen;i++)
#pragma omp atomic
                       cfg->exportfield[i+fieldlen]+=rfimag[i];
               }
	       free(rfimag);
	       rfimag=NULL;
       }

       if(param.twin1<cfg->tend){
            CUDA_ASSERT(cudaMemset(genergy,0,sizeof(float)*(gpu[gpuid].autothread<<1)));
       }
     } /** Here is the end of the outer-loop, over time-gate groups */
#pragma omp barrier

     /**
       * Let the master thread to deal with the normalization and file IO
       * First, if multi-pattern simulation, i.e. photon sharing, is used, we normalize each pattern first
       */
#pragma omp master
{
     if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcnum>1){// post-processing only for multi-srcpattern
         srcpw=(float *)calloc(cfg->srcnum,sizeof(float));
	 energytot=(float *)calloc(cfg->srcnum,sizeof(float));
	 energyabs=(float *)calloc(cfg->srcnum,sizeof(float));
	 int psize=(int)cfg->srcparam1.w*(int)cfg->srcparam2.w;
	 for(i=0;i<int(cfg->srcnum);i++){
	     float kahanc=0.f;
	     for(iter=0;iter<psize;iter++)   
	         mcx_kahanSum(&srcpw[i],&kahanc,cfg->srcpattern[iter*cfg->srcnum+i]);
	     energytot[i]=cfg->nphoton*srcpw[i]/(float)psize;
	     kahanc=0.f;
	     if(cfg->outputtype==otEnergy){
	         int fieldlenPsrc=fieldlen/cfg->srcnum;
	         for(iter=0;iter<fieldlenPsrc;iter++)
		     mcx_kahanSum(&energyabs[i],&kahanc,cfg->exportfield[iter*cfg->srcnum+i]);
	     }else{
	         int j;
	         for(iter=0;iter<gpu[gpuid].maxgate;iter++)
		     for(j=0;j<(int)dimlen.z;j++)
		         mcx_kahanSum(&energyabs[i],&kahanc,cfg->exportfield[iter*dimxyz+(j*cfg->srcnum+i)]*mcx_updatemua((uint)cfg->vol[j],cfg));
	     }
	 }
     }
     /**
       * Now we normalize the fluence so that the default output is fluence rate in joule/(s*mm^2)
       * generated by a unitary source (1 joule total).
       *
       * The raw output directly from GPU is the accumulated energy-loss per photon moving step 
       * in joule when cfg.outputtype='fluence', or energy-loss multiplied by mua (1/mm) per voxel
       * (joule/mm) when cfg.outputtype='flux' (default). 
       */
     if(cfg->isnormalized){
	   float *scale=(float *)calloc(cfg->srcnum,sizeof(float));
	   scale[0]=1.f;
	   int isnormalized=0;
           MCX_FPRINTF(cfg->flog,"normalizing raw data ...\t");
           cfg->energyabs+=cfg->energytot-cfg->energyesc;
           /**
	     * If output is flux (J/(s*mm^2), default), raw data (joule*mm) is multiplied by (1/(Nphoton*Vvox*dt))
	     * If output is fluence (J/mm^2), raw data (joule*mm) is multiplied by (1/(Nphoton*Vvox))
	     */
           if(cfg->outputtype==otFlux || cfg->outputtype==otFluence){
               scale[0]=cfg->unitinmm/(cfg->energytot*Vvox*cfg->tstep); /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

               if(cfg->outputtype==otFluence)
		   scale[0]*=cfg->tstep;
	   }else if(cfg->outputtype==otEnergy) /** If output is energy (joule), raw data is simply multiplied by 1/Nphoton */
	       scale[0]=1.f/cfg->energytot;
	   else if(cfg->outputtype==otJacobian || cfg->outputtype==otWP || cfg->outputtype==otDCS || cfg->outputtype==otRF){
	       if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
                   int detid;
		   for(detid=1;detid<=(int)cfg->detnum;detid++){
	               scale[0]=0.f; // the cfg->normalizer and cfg.his.normalizer are inaccurate in this case, but this is ok
		       for(size_t i=0;i<cfg->nphoton;i++)
		           if(cfg->replay.detid[i]==detid)
	                       scale[0]+=cfg->replay.weight[i];
	               if(scale[0]>0.f)
	                   scale[0]=cfg->unitinmm/scale[0];
                       MCX_FPRINTF(cfg->flog,"normalization factor for detector %d alpha=%f\n",detid, scale[0]);  fflush(cfg->flog);
                       mcx_normalize(cfg->exportfield+(detid-1)*dimxyz*gpu[gpuid].maxgate,scale[0],dimxyz*gpu[gpuid].maxgate,cfg->isnormalized,0,1);
		   }
		   isnormalized=1;
	       }else{
	           scale[0]=0.f;
	           for(size_t i=0;i<cfg->nphoton;i++)
	               scale[0]+=cfg->replay.weight[i];
	           if(scale[0]>0.f)
                       scale[0]=cfg->unitinmm/scale[0];
                   MCX_FPRINTF(cfg->flog,"normalization factor for detector %d alpha=%f\n",cfg->replaydet, scale[0]);  fflush(cfg->flog);
	       }
           }
           /**
	     * In photon sharing mode, where multiple pattern sources are simulated, each solution is normalized separately
	     */
	   if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcnum>1){// post-processing only for multi-srcpattern
	       float scaleref=scale[0];
	       int psize=(int)cfg->srcparam1.w*(int)cfg->srcparam2.w;
	       for(i=0;i<int(cfg->srcnum);i++){
		   scale[i]=psize/srcpw[i]*scaleref;
	       }
	   }
         cfg->normalizer=scale[0];
	 cfg->his.normalizer=scale[0];
         if(!isnormalized){
	     for(i=0;i<(int)cfg->srcnum;i++){
                 MCX_FPRINTF(cfg->flog,"source %d, normalization factor alpha=%f\n",(i+1),scale[i]);  fflush(cfg->flog);
	         mcx_normalize(cfg->exportfield,scale[i],fieldlen/cfg->srcnum,cfg->isnormalized,i,cfg->srcnum);
	     }
	 }
	 free(scale);
         MCX_FPRINTF(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);
     }
     /**
       * If not running as a mex file, we need to save volumetric output data, if enabled, as 
       * a file, with suffix specifed by cfg.outputformat (mc2,nii, or .jdat or .jbat)
       */
     if(cfg->issave2pt && cfg->parentid==mpStandalone){
         MCX_FPRINTF(cfg->flog,"saving data to file ...\t");
         mcx_savedata(cfg->exportfield,fieldlen,cfg);
         MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
         fflush(cfg->flog);
     }
     /**
       * If not running as a mex file, we need to save detected photon data, if enabled, as 
       * a file, either as a .mch file, or a .jdat/.jbat file
       */
     if(cfg->issavedet && cfg->parentid==mpStandalone && cfg->exportdetected){
         cfg->his.unitinmm=cfg->unitinmm;
         cfg->his.savedphoton=cfg->detectedcount;
	 cfg->his.totalphoton=cfg->nphoton;
         if(cfg->issaveseed)
             cfg->his.seedbyte=sizeof(RandType)*RAND_BUF_LEN;

         cfg->his.detected=cfg->detectedcount;
         mcx_savedetphoton(cfg->exportdetected,cfg->seeddata,cfg->detectedcount,0,cfg);
     }
     /**
       * If not running as a mex file, we need to save photon trajectory data, if enabled, as 
       * a file, either as a .mct file, or a .jdat/.jbat file
       */
     if((cfg->debuglevel & MCX_DEBUG_MOVE) && cfg->parentid==mpStandalone && cfg->exportdebugdata){
         cfg->his.colcount=MCX_DEBUG_REC_LEN;
         cfg->his.savedphoton=cfg->debugdatalen;
	 cfg->his.totalphoton=cfg->nphoton;
         cfg->his.detected=0;
         mcx_savedetphoton(cfg->exportdebugdata,NULL,cfg->debugdatalen,0,cfg);
     }
}
#pragma omp barrier
     /**
       * Copying GPU photon states back to host as Ppos, Pdir and Plen for debugging purpose is depreciated
       */
     CUDA_ASSERT(cudaMemcpy(Ppos,  gPpos, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(Pdir,  gPdir, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(Plen,  gPlen, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     if(cfg->seed!=SEED_FROM_FILE)
        CUDA_ASSERT(cudaMemcpy(Pseed, gPseed,sizeof(RandType)*gpu[gpuid].autothread*RAND_BUF_LEN,   cudaMemcpyDeviceToHost));
     else
        CUDA_ASSERT(cudaMemcpy(Pseed, gPseed,sizeof(RandType)*cfg->nphoton*RAND_BUF_LEN,   cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(energy,genergy,sizeof(float)*(gpu[gpuid].autothread<<1),cudaMemcpyDeviceToHost));

#pragma omp master
{
     printnum=(gpu[gpuid].autothread<(int)cfg->printnum) ? gpu[gpuid].autothread : cfg->printnum;
     for (i=0; i<(int)printnum; i++) {
            MCX_FPRINTF(cfg->flog,"% 4d[A% f % f % f]C%3d J%5d W% 8f(P%.13f %.13f %.13f)T% 5.3e L% 5.3f %.0f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i]);
     }
     /**
       * Report simulation summary, total energy here equals total simulated photons+unfinished photons for all threads
       */
     MCX_FPRINTF(cfg->flog,"simulated %ld photons (%ld) with %d threads (repeat x%d)\nMCX simulation speed: " S_BOLD "" S_BLUE "%.2f photon/ms\n" S_RESET,
             (long int)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1),(long int)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1),
	     gpu[gpuid].autothread,ABS(cfg->respin),(double)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1)/max(1,cfg->runtime)); fflush(cfg->flog);
     if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcnum>1){
         for(i=0;i<(int)cfg->srcnum;i++){
	     MCX_FPRINTF(cfg->flog,"source #%d total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET"\n(loss due to initial specular reflection is excluded in the total)\n",
                 i+1,energytot[i],energyabs[i]/energytot[i]*100.f);fflush(cfg->flog);
	 }
     }else{
         MCX_FPRINTF(cfg->flog,"total simulated energy: %.2f\tabsorbed: " S_BOLD "" S_BLUE "%5.5f%%" S_RESET"\n(loss due to initial specular reflection is excluded in the total)\n",
             cfg->energytot,(cfg->energytot-cfg->energyesc)/cfg->energytot*100.f);fflush(cfg->flog);
         fflush(cfg->flog);
     }
     
     cfg->energyabs=cfg->energytot-cfg->energyesc;
}
#pragma omp barrier

     /**
       * Simulation is complete, now we need clear up all GPU memory buffers
       */
     CUDA_ASSERT(cudaFree(gmedia));
     CUDA_ASSERT(cudaFree(gfield));
     CUDA_ASSERT(cudaFree(gPpos));
     CUDA_ASSERT(cudaFree(gPdir));
     CUDA_ASSERT(cudaFree(gPlen));
     CUDA_ASSERT(cudaFree(gPseed));
     CUDA_ASSERT(cudaFree(genergy));
     CUDA_ASSERT(cudaFree(gPdet));
     CUDA_ASSERT(cudaFree(gdetected));
     if(cfg->nphase)
         CUDA_ASSERT(cudaFree(ginvcdf));
     if(cfg->polmedianum)
         CUDA_ASSERT(cudaFree(gsmatrix));
     if(cfg->debuglevel & MCX_DEBUG_MOVE)
         CUDA_ASSERT(cudaFree(gdebugdata));
     if(cfg->issaveseed){
         CUDA_ASSERT(cudaFree(gseeddata));
	 free(seeddata);
     }
     if(cfg->seed==SEED_FROM_FILE){
         if(cfg->replay.weight)
             CUDA_ASSERT(cudaFree(greplayw));
         if(cfg->replay.tof)
             CUDA_ASSERT(cudaFree(greplaytof));
         if(cfg->replay.detid)
             CUDA_ASSERT(cudaFree(greplaydetid));
     }

     /**
       * The below call in theory is not needed, but it ensures the device is freed for other programs, especially on Windows
       */
     CUDA_ASSERT(cudaDeviceReset());

     /**
       * Lastly, free all host buffers, the simulation is complete.
       */
     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Plen0);
     free(Pseed);
     free(Pdet);
     free(energy);
     free(field);
     free(srcpw);
     free(energytot);
     free(energyabs);
}
