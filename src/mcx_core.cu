/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504, 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_core.cu

@brief   GPU kernel for MC simulations and CUDA host code
*******************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>

#include "br2cu.h"
#include "mcx_core.h"
#include "tictoc.h"
#include "mcx_const.h"

#ifdef USE_HALF                     ///< use half-precision for ray-tracing
    #include "cuda_fp16.h"
#endif

#if defined(USE_XOROSHIRO128P_RAND)
    #include "xoroshiro128p_rand.cu" ///< use xorshift128+ RNG (XORSHIFT128P)
#elif defined(USE_LL5_RAND)
    #include "logistic_rand.cu"     ///< use Logistic Lattice ring 5 RNG (LL5)
#elif defined(USE_POSIX_RAND)
    #include "posix_rand.cu"        ///< use POSIX erand48 RNG (POSIX)
#else                               ///< default RNG method: USE_XORSHIFT128P_RAND
    #include "xorshift128p_rand.cu" ///< use xorshift128+ RNG (XORSHIFT128P)
#endif

#ifdef _OPENMP                      ///< use multi-threading for running simulation on multiple GPUs
    #include <omp.h>
#endif

#define CUDA_ASSERT(a)      mcx_cu_assess((a),__FILE__,__LINE__) ///< macro to report CUDA errors

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
 
__device__ inline float atomicadd(float* address, float value){

#if __CUDA_ARCH__ >= 200 ///< for Fermi, atomicAdd supports floats

  return atomicAdd(address,value);

#elif __CUDA_ARCH__ >= 110

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

/**
 * @brief Reset shared memory buffer for storing fluence near the source (obsolete)
 * @param[in] p: pointer to the cache buffer
 * @param[in] len: length of the buffer to be reset
 */

__device__ inline void clearcache(float *p,int len){
      uint i;
      if(threadIdx.x==0)
        for(i=0;i<len;i++)
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

__device__ inline void savedetphoton(float n_det[],uint *detectedphoton,float *ppath,MCXpos *p0,MCXdir *v,RandType t[RAND_BUF_LEN],RandType *seeddata){
      uint detid=finddetector(p0);
      if(detid){
	 uint baseaddr=atomicAdd(detectedphoton,1);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    for(i=0;i<gcfg->issaveseed*RAND_BUF_LEN;i++)
	        seeddata[baseaddr*RAND_BUF_LEN+i]=t[i]; ///< save photon seed for replay
	    baseaddr*=2+gcfg->maxmedia*(2+gcfg->ismomentum)+(gcfg->issaveexit>0)*6;
	    n_det[baseaddr++]=detid;
	    for(i=0;i<gcfg->maxmedia*(2+gcfg->ismomentum);i++)
		n_det[baseaddr++]=ppath[i]; ///< save partial pathlength to the memory
	    if(gcfg->issaveexit){
	        *((float3*)(n_det+baseaddr))=float3(p0->x,p0->y,p0->z);
		baseaddr+=3;
		*((float3*)(n_det+baseaddr))=float3(v->x,v->y,v->z);
		baseaddr+=3;
	    }
	    n_det[baseaddr++]=ppath[gcfg->maxmedia*(2+gcfg->ismomentum)];
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

      ///< time-of-flight to hit the wall in each direction
      htime[0]=fabs((floorf(p0->x)+(v->x>0.f)-p0->x)*rv[0]); ///< time-of-flight in x
      htime[1]=fabs((floorf(p0->y)+(v->y>0.f)-p0->y)*rv[1]);
      htime[2]=fabs((floorf(p0->z)+(v->z>0.f)-p0->z)*rv[2]);

      ///< get the direction with the smallest time-of-flight
      dist=fminf(fminf(htime[0],htime[1]),htime[2]);
      (*id)=(dist==htime[0]?0:(dist==htime[1]?1:2));

      ///< p0 is inside, htime is the 1st intersection point
      htime[0]=p0->x+dist*v->x;
      htime[1]=p0->y+dist*v->y;
      htime[2]=p0->z+dist*v->z;

      ///< make sure the intersection point htime is immediately outside of the current voxel (i.e. not within the current voxel)
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
          half f;
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
           half2 h2;
           half h[2];
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
      if(tmp2>0.f){ ///< partial reflection
          float Re,Im,Rtotal;
	  Re=tmp0*Icos*Icos+tmp1*tmp2;
	  tmp2=sqrtf(tmp2); /** to save one sqrt*/
	  Im=2.f*n1*n2*Icos*tmp2;
	  Rtotal=(Re-Im)/(Re+Im);     /** Rp*/
	  Re=tmp1*Icos*Icos+tmp0*tmp2*tmp2;
	  Rtotal=(Rtotal+(Re-Im)/(Re+Im))*0.5f; /** (Rp+Rs)/2*/
	  return Rtotal;
      }else{ ///< total reflection
          return 1.f;
      }
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

__device__ inline int skipvoid(MCXpos *p,MCXdir *v,MCXtime *f,float3* rv,uint media[]){
      int count=1,idx1d;
      while(1){
          if(p->x>=0.f && p->y>=0.f && p->z>=0.f && p->x < gcfg->maxidx.x
               && p->y < gcfg->maxidx.y && p->z < gcfg->maxidx.z){
	    idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	    if(media[idx1d] & MED_MASK){ ///< if enters a non-zero voxel
                GPUDEBUG(("inside volume [%f %f %f] v=<%f %f %f>\n",p->x,p->y,p->z,v->x,v->y,v->z));
	        float3 htime;
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

		if(gcfg->isspecular && gproperty[media[idx1d] & MED_MASK].w!=gproperty[0].w){
	            p->w*=1.f-reflectcoeff(v, gproperty[0].w,gproperty[media[idx1d] & MED_MASK].w,flipdir);
                    GPUDEBUG(("transmitted intensity w=%e\n",p->w));
	            if(p->w>EPS){
		        transmit(v, gproperty[0].w,gproperty[media[idx1d] & MED_MASK].w,flipdir);
                        GPUDEBUG(("transmit into volume v=<%f %f %f>\n",v->x,v->y,v->z));
                    }
		}
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
 * @param[in,out] rv: the reciprocal direction vector of the photon (rv[i]=1/v[i])
 * @param[out] prop: the optical properties of the voxel the photon is launched into
 * @param[in,out] idx1d: the linear index of the voxel containing the photon at launch
 * @param[in] field: the 3D array to store photon weights
 * @param[in,out] mediaid: the medium index at the voxel at launch
 * @param[in,out] w0: initial weight, reset here after launch
 * @param[in] isdet: whether the previous photon being terminated lands at a detector
 * @param[in,out] ppath: pointer to the shared-mem buffer to store photon partial-path data
 * @param[in,out] energyloss: register variable to accummulate the escaped photon energy
 * @param[in,out] energylaunched: register variable to accummulate the total launched photon energy
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

template <int mcxsource>
__device__ inline int launchnewphoton(MCXpos *p,MCXdir *v,MCXtime *f,float3* rv,Medium *prop,uint *idx1d, float *field,
           uint *mediaid,float *w0,uint isdet, float ppath[],float energyloss[],float energylaunched[],float n_det[],uint *dpnum,
	   RandType t[RAND_BUF_LEN],RandType photonseed[RAND_BUF_LEN],
	   uint media[],float srcpattern[],int threadid,RandType rngseed[],RandType seeddata[],float gdebugdata[],volatile int gprogress[]){
      *w0=1.f;     ///< reuse to count for launchattempt
      f->pathlen=-1.f; ///< reuse as "canfocus" flag for each source: non-zero: focusable, zero: not focusable
      *rv=float3(gcfg->ps.x,gcfg->ps.y,gcfg->ps.z); ///< reuse as the origin of the src, needed for focusable sources

      /**
       * First, let's terminate the current photon and perform detection calculations
       */
      if(p->w>=0.f){
          *energyloss+=p->w;  ///< sum all the remaining energy
#ifdef SAVE_DETECTORS
      // let's handle detectors here
          if(gcfg->savedet){
             if((isdet&DET_MASK)==DET_MASK && *mediaid==0)
	         savedetphoton(n_det,dpnum,ppath,p,v,photonseed,seeddata);
             clearpath(ppath,gcfg->maxmedia*(2+gcfg->ismomentum)+1);
          }
#endif
          if(*mediaid==0 && *idx1d!=OUTSIDE_VOLUME_MIN && *idx1d!=OUTSIDE_VOLUME_MAX && gcfg->issaveref){
	       int tshift=MIN(gcfg->maxgate-1,(int)(floorf((f->t-gcfg->twin0)*gcfg->Rtstep)));
#ifdef USE_ATOMIC
               atomicadd(& field[*idx1d+tshift*gcfg->dimlen.z],-p->w);	       
#else
	       field[*idx1d+tshift*gcfg->dimlen.z]+=-p->w;
#endif
	  }
      }

      /**
       * If the thread completes all assigned photons, terminate this thread.
       */
      if((int)(f->ndone)>=(gcfg->threadphoton+(threadid<gcfg->oddphotons))-1){
          return 1; // all photos complete
      }
      
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

          /**
           * Only one branch is taken because of template, this can reduce thread divergence
           */
	  switch(mcxsource) {
		case(MCX_SRC_PLANAR):
		case(MCX_SRC_PATTERN):
		case(MCX_SRC_PATTERN3D):
		case(MCX_SRC_FOURIER):
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
		      if(gcfg->srctype==MCX_SRC_PATTERN) // need to prevent rx/ry=1 here
			  p->w=srcpattern[(int)(ry*JUST_BELOW_ONE*gcfg->srcparam2.w)*(int)(gcfg->srcparam1.w)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.w)];
		      else if(gcfg->srctype==MCX_SRC_PATTERN3D)
		          p->w=srcpattern[(int)(rz*JUST_BELOW_ONE*gcfg->srcparam1.z)*(int)(gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+
		                          (int)(ry*JUST_BELOW_ONE*gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.x)];
		      else if(gcfg->srctype==MCX_SRC_FOURIER)
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
		      v2.w*=rsqrt(gcfg->srcparam1.x*gcfg->srcparam1.x+gcfg->srcparam1.y*gcfg->srcparam1.y+gcfg->srcparam1.z*gcfg->srcparam1.z);
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
		case(MCX_SRC_GAUSSIAN): { // uniform disk distribution or Gaussian-beam
		      // Uniform disk point picking
		      // http://mathworld.wolfram.com/DiskPointPicking.html
		      float sphi, cphi;
		      float phi=TWO_PI*rand_uniform01(t);
		      sincosf(phi,&sphi,&cphi);
		      float r;
		      if(gcfg->srctype==MCX_SRC_DISK)
			  r=sqrtf(rand_uniform01(t))*gcfg->srcparam1.x;
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
		case(MCX_SRC_CONE):
		case(MCX_SRC_ISOTROPIC):
		case(MCX_SRC_ARCSINE): {
		      // Uniform point picking on a sphere 
		      // http://mathworld.wolfram.com/SpherePointPicking.html
		      float ang,stheta,ctheta,sphi,cphi;
		      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
		      sincosf(ang,&sphi,&cphi);
		      if(gcfg->srctype==MCX_SRC_CONE){  // a solid-angle section of a uniform sphere
			  do{
			      ang=(gcfg->srcparam1.y>0) ? TWO_PI*rand_uniform01(t) : acosf(2.f*rand_uniform01(t)-1.f); //sine distribution
			  }while(ang>gcfg->srcparam1.x);
		      }else{
			  if(gcfg->srctype==MCX_SRC_ISOTROPIC) // uniform sphere
			      ang=acosf(2.f*rand_uniform01(t)-1.f); //sine distribution
			  else
			      ang=ONE_PI*rand_uniform01(t); //uniform distribution in zenith angle, arcsine
		      }
		      sincosf(ang,&stheta,&ctheta);
		      rotatevector(v,stheta,ctheta,sphi,cphi);
                      f->pathlen=0.f;
		      break;
		}
		case(MCX_SRC_ZGAUSSIAN): {
		      float ang,stheta,ctheta,sphi,cphi;
		      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
		      sincosf(ang,&sphi,&cphi);
		      ang=sqrtf(-2.f*logf(rand_uniform01(t)))*(1.f-2.f*rand_uniform01(t))*gcfg->srcparam1.x;
		      sincosf(ang,&stheta,&ctheta);
		      rotatevector(v,stheta,ctheta,sphi,cphi);
                      f->pathlen=0.f;
		      break;
		}
		case(MCX_SRC_LINE):
		case(MCX_SRC_SLIT): {
		      float r=rand_uniform01(t);
		      *((float4*)p)=float4(p->x+r*gcfg->srcparam1.x,
					   p->y+r*gcfg->srcparam1.y,
					   p->z+r*gcfg->srcparam1.z,
					   p->w);
		      if(gcfg->srctype==MCX_SRC_LINE){
			      float s,q;
			      r=1.f-2.f*rand_uniform01(t);
			      s=1.f-2.f*rand_uniform01(t);
			      q=sqrt(1.f-v->x*v->x-v->y*v->y)*(rand_uniform01(t)>0.5f ? 1.f : -1.f);
			      *((float4*)v)=float4(v->y*q-v->z*s,v->z*r-v->x*q,v->x*s-v->y*r,v->nscat);
		      }
                      *rv=float3(rv->x+(gcfg->srcparam1.x)*0.5f,
		                 rv->y+(gcfg->srcparam1.y)*0.5f,
				 rv->z+(gcfg->srcparam1.z)*0.5f);
                      f->pathlen=(gcfg->srctype==MCX_SRC_SLIT)?-1.f:0.f;
		      break;
		}
	  }
          /**
           * If beam focus is set, determine the incident angle
           */
          if(f->pathlen<0.f){
	    if(gcfg->c0.w!=0.f){
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
	    }else if(__float_as_int(gcfg->c0.w)==0x80000000){ // isotropic if focal length is -0.f
                float ang,stheta,ctheta,sphi,cphi;
                ang=TWO_PI*rand_uniform01(t); //next arimuth angle
                sincosf(ang,&sphi,&cphi);
                ang=acosf(2.f*rand_uniform01(t)-1.f); //sine distribution
                sincosf(ang,&stheta,&ctheta);
                rotatevector(v,stheta,ctheta,sphi,cphi);
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
             int idx=skipvoid(p, v, f, rv, media); /** specular reflection of the bbx is taken care of here*/
             if(idx>=0){
		 *idx1d=idx;
		 *mediaid=media[*idx1d];
	     }
	  }
	  *w0+=1.f;
	  
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
      *((float4*)(prop))=gproperty[*mediaid & MED_MASK]; //always use mediaid to read gproperty[]
      if(gcfg->debuglevel & MCX_DEBUG_MOVE)
          savedebugdata(p,(uint)f->ndone+threadid*gcfg->threadphoton+umin(threadid,(threadid<gcfg->oddphotons)*threadid),gdebugdata);

      /**
        total energy enters the volume. for diverging/converting 
        beams, this is less than nphoton due to specular reflection 
        loss. This is different from the wide-field MMC, where the 
        total launched energy includes the specular reflection loss
       */
      *energylaunched+=p->w;
      *w0=p->w;
      ppath[gcfg->maxmedia*(2+gcfg->ismomentum)]=p->w; // store initial weight
      v->nscat=EPS;
      f->pathlen=0.f;
      
      /**
       * If a progress bar is needed, only sum completed photons from the 1st, last and middle threads to determine progress bar
       */
      if((gcfg->debuglevel & MCX_DEBUG_PROGRESS) && ((int)(f->ndone) & 1) && (threadid==0 || threadid==blockDim.x * gridDim.x - 1 
          || threadid==((blockDim.x * gridDim.x)>>1))) { ///< use the 1st, middle and last thread for progress report
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

kernel void mcx_test_rng(float field[],uint n_seed[]){
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

template <int mcxsource>
kernel void mcx_main_loop(uint media[],float field[],float genergy[],uint n_seed[],
     float4 n_pos[],float4 n_dir[],float4 n_len[],float n_det[], uint detectedphoton[], 
     float srcpattern[],float replayweight[],float photontof[],int photondetid[], 
     RandType *seeddata,float *gdebugdata,volatile int *gprogress){

     /** the 1D index of the current thread */
     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     if(idx>=gcfg->threadphoton*(blockDim.x * gridDim.x)+gcfg->oddphotons)
         return;
     MCXpos  p={0.f,0.f,0.f,-1.f};   ///< Photon position state: {x,y,z}: coordinates in grid unit, w:packet weight
     MCXdir  v={0.f,0.f,0.f, 0.f};   ///< Photon direction state: {x,y,z}: unitary direction vector in grid unit, nscat:total scat event
     MCXtime f={0.f,0.f,0.f,-1.f};   ///< Photon parameter state: pscat: remaining scattering probability,t: photon elapse time, pathlen: total pathlen in one voxel, ndone: completed photons
     float  energyloss=genergy[idx<<1];
     float  energylaunched=genergy[(idx<<1)+1];

     uint idx1d, idx1dold;    ///< linear index to the current voxel in the media array

     uint  mediaid=gcfg->mediaidorig;
     uint  mediaidold=0;
     int   isdet=0;
     float  n1;               ///< reflection var
     float3 htime;            ///< time-of-flight for collision test
     float3 rv;               ///< reciprocal velocity

     RandType t[RAND_BUF_LEN];
     Medium prop;

     float len, slen;
     float w0;
     int   flipdir=-1;
 
     float *ppath=(float *)(sharedmem+blockDim.x*(gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)));

#ifdef  SAVE_DETECTORS
     ppath+=threadIdx.x*(gcfg->maxmedia*(2+gcfg->ismomentum)+1); // block#2: maxmedia*thread number to store the partial
     if(gcfg->savedet) clearpath(ppath,gcfg->maxmedia*(2+gcfg->ismomentum)+1);
#endif

     gpu_rng_init(t,n_seed,idx);

     if(launchnewphoton<mcxsource>(&p,&v,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,0,ppath,&energyloss,
       &energylaunched,n_det,detectedphoton,t,(RandType*)(sharedmem+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),media,srcpattern,
       idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress)){
         GPUDEBUG(("thread %d: fail to launch photon\n",idx));
	 n_pos[idx]=*((float4*)(&p));
	 n_dir[idx]=*((float4*)(&v));
	 n_len[idx]=*((float4*)(&f));
         return;
     }
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
	  if(f.pscat<=0.f) {  ///< if this photon has finished his current scattering path, calculate next scat length & angles
   	       f.pscat=rand_next_scatlen(t); ///< random scattering probability, unit-less, exponential distribution

               GPUDEBUG(("scat L=%f RNG=[%0lX %0lX] \n",f.pscat,t[0],t[1]));
	       if(v.nscat!=EPS){ ///< if v.nscat is EPS, this means it is the initial launch direction, no need to change direction
                       ///< random arimuthal angle
	               float cphi=1.f,sphi=0.f,theta,stheta,ctheta;
                       float tmp0=0.f;
		       if(!gcfg->is2d){
		           tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                           sincosf(tmp0,&sphi,&cphi);
		       }
                       GPUDEBUG(("scat phi=%f\n",tmp0));
		       tmp0=(v.nscat > gcfg->gscatter) ? 0.f : prop.g;

                       /** Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002 */

                       if(tmp0>EPS){  ///< if prop.g is too small, the distribution of theta is bad
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
                       GPUDEBUG(("scat theta=%f\n",theta));
#ifdef SAVE_DETECTORS
                       ppath[(mediaid & MED_MASK)-1]++;
	               /** accummulate momentum transfer */
                       if(gcfg->ismomentum)
	                   ppath[(gcfg->maxmedia<<1)+(mediaid & MED_MASK)-1]+=1.f-ctheta;
#endif
                       /** Update direction vector with the two random angles */
		       if(gcfg->is2d)
		           rotatevector2d(&v,(rand_next_aangle(t)>0.5f ? stheta: -stheta),ctheta);
		       else
                           rotatevector(&v,stheta,ctheta,sphi,cphi);
                       v.nscat++;

		       /** Only compute the reciprocal vector when v is changed, this saves division calculations, which are very expensive on the GPU */
                       rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
                       if(gcfg->outputtype==otWP || gcfg->outputtype==otDCS){
                            ///< photontof[] and replayweight[] should be cached using local mem to avoid global read
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
                                atomicadd(& field[idx1d+tshift*gcfg->dimlen.z], tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]);
                                GPUDEBUG(("atomic write to [%d] %e, w=%f\n",idx1d,tmp0*replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)],p.w));
                            }
#endif
                       }
                       if(gcfg->debuglevel & MCX_DEBUG_MOVE)
                           savedebugdata(&p,(uint)f.ndone+idx*gcfg->threadphoton+umin(idx,(idx<gcfg->oddphotons)*idx),gdebugdata);
	       }
	       v.nscat=(int)v.nscat;
	  }

          /** Read the optical property of the current voxel */
          n1=prop.n;
	  *((float4*)(&prop))=gproperty[mediaid & MED_MASK];
	  
	  /** Advance photon 1 step to the next voxel */
	  len=(gcfg->faststep) ? gcfg->minstep : hitgrid((float3*)&p,(float3*)&v,&(htime.x),&rv.x,&flipdir); // propagate the photon to the first intersection to the grid
	  
	  /** convert photon movement length to unitless scattering length by multiplying with mus */
	  slen=len*prop.mus*(v.nscat+1.f > gcfg->gscatter ? (1.f-prop.g) : 1.f); //unitless (minstep=grid, mus=1/grid)

          GPUDEBUG(("p=[%f %f %f] -> <%f %f %f>*%f -> hit=[%f %f %f] flip=%d\n",p.x,p.y,p.z,v.x,v.y,v.z,len,htime.x,htime.y,htime.z,flipdir));

	  /** if the consumed unitless scat length is less than what's left in f.pscat, keep moving; otherwise, stop in this voxel */
	  slen=fmin(slen,f.pscat);
	  
	  /** final length that the photon moves - either the length to move to the next voxel, or the remaining scattering length */
	  len=slen/(prop.mus*(v.nscat+1.f > gcfg->gscatter ? (1.f-prop.g) : 1.f));
	  
	  /** if photon moves to the next voxel, use the precomputed intersection coord. htime which are assured to be outside of the current voxel */
	  *((float3*)(&p)) = (gcfg->faststep || slen==f.pscat) ? float3(p.x+len*v.x,p.y+len*v.y,p.z+len*v.z) : float3(htime.x,htime.y,htime.z);
	  
	  /** calculate photon energy loss */
	  p.w*=expf(-prop.mua*len);
	  
	  /** remaining unitless scattering length: sum(s_i*mus_i), unit-less */
	  f.pscat-=slen;

	  /** update photon timer to add time-of-flight (unit = s) */
	  f.t+=len*prop.n*gcfg->oneoverc0;
	  f.pathlen+=len;

          GPUDEBUG(("update p=[%f %f %f] -> len=%f\n",p.x,p.y,p.z,len));

#ifdef SAVE_DETECTORS
	  /** accummulate partial path of the current medium */
          if(gcfg->savedet)
	      ppath[gcfg->maxmedia+(mediaid & MED_MASK)-1]+=len; //(unit=grid)
#endif

          mediaidold=mediaid | isdet;
          idx1dold=idx1d;
          idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
          GPUDEBUG(("idx1d [%d]->[%d]\n",idx1dold,idx1d));

	  /** read the medium index of the new voxel (current or next) */
          if(p.x<0.f||p.y<0.f||p.z<0.f||p.x>=gcfg->maxidx.x||p.y>=gcfg->maxidx.y||p.z>=gcfg->maxidx.z){
              /** if photon moves outside of the volume, set mediaid to 0 */
	      mediaid=0;
	      isdet=gcfg->bc[(!(p.x<0.f||p.y<0.f||p.z<0.f))*3+flipdir];  /** isdet now stores the boundary condition flag, this will be overwriten before the end of the loop */
              GPUDEBUG(("moving outside: [%f %f %f], idx1d [%d]->[out], bcflag %d\n",p.x,p.y,p.z,idx1d,isdet));
	      idx1d=(p.x<0.f||p.y<0.f||p.z<0.f) ? OUTSIDE_VOLUME_MIN : OUTSIDE_VOLUME_MAX;
	  }else{
              /** otherwise, read the optical property index */
	      mediaid=media[idx1d];
	      isdet=mediaid & DET_MASK;  /** upper 16bit is the mask of the covered detector */
	      mediaid &= MED_MASK;       /** lower 16bit is the medium index */
          }
          GPUDEBUG(("medium [%d]->[%d]\n",mediaidold,mediaid));

          /**  save fluence to the voxel when photon moves out */
	  if(idx1d!=idx1dold && mediaidold){

             /**  if t is within the time window, which spans cfg->maxgate*cfg->tstep.wide */
             if(gcfg->save2pt && f.t>=gcfg->twin0 && f.t<gcfg->twin1){
	          float weight=0.f;
                  int tshift=(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep));
		  
		  /** calculate the quality to be accummulated */
		  if(gcfg->outputtype==otEnergy)
		      weight=w0-p.w;
		  else if(gcfg->seed==SEED_FROM_FILE){
		      if(gcfg->outputtype==otJacobian){
		        weight=replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]*f.pathlen;
			tshift=(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone);
			tshift=(int)(floorf((photontof[tshift]-gcfg->twin0)*gcfg->Rtstep)) + 
			   ( (gcfg->replaydet==-1)? ((photondetid[tshift]-1)*gcfg->maxgate) : 0);
		      }
		  }else
		      weight=(prop.mua==0.f) ? 0.f : ((w0-p.w)/(prop.mua));

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
		  float oldval=atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], weight);
		  if(oldval>MAX_ACCUM){
			if(atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], -oldval)<0.f)
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], oldval);
			else
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z+gcfg->dimlen.w], oldval);
		  }
                  GPUDEBUG(("atomic write to [%d] %e, w=%f\n",idx1dold,weight,p.w));
               }
#endif
              }
	     }
	     w0=p.w;
	     f.pathlen=0.f;
	  }

	  /** launch new photon when exceed time window or moving from non-zero voxel to zero voxel without reflection */
          if((mediaid==0 && (((isdet & 0xF)==0 && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[0].w))) || (isdet==bcAbsorb || isdet==bcCylic) )) || f.t>gcfg->twin1){
	      if(isdet==bcCylic){
		 if(flipdir==0)  p.x=mcx_nextafterf(roundf(p.x+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.x: -gcfg->maxidx.x)),(v.x > 0.f)-(v.x < 0.f));
		 if(flipdir==1)  p.y=mcx_nextafterf(roundf(p.y+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.y: -gcfg->maxidx.y)),(v.y > 0.f)-(v.y < 0.f));
		 if(flipdir==2)  p.z=mcx_nextafterf(roundf(p.z+((idx1d==OUTSIDE_VOLUME_MIN) ? gcfg->maxidx.z: -gcfg->maxidx.z)),(v.z > 0.f)-(v.z < 0.f));
                 idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
	         mediaid=media[idx1d];
	         isdet=mediaid & DET_MASK;  /** upper 16bit is the mask of the covered detector */
	         mediaid &= MED_MASK;       /** lower 16bit is the medium index */
                 GPUDEBUG(("Cylic boundary condition, moving photon in dir %d at %d flag, new pos=[%f %f %f]\n",flipdir,isdet,p.x,p.y,p.z));
	         continue;
	      }
              GPUDEBUG(("direct relaunch at idx=[%d] mediaid=[%d], ref=[%d] bcflag=%d timegate=%d\n",idx1d,mediaid,gcfg->doreflect,isdet,f.t>gcfg->twin1));
	      if(launchnewphoton<mcxsource>(&p,&v,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),ppath,
	          &energyloss,&energylaunched,n_det,detectedphoton,t,(RandType*)(sharedmem+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
		  media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress))
                   break;
              isdet=mediaid & DET_MASK;
              mediaid &= MED_MASK;
	      continue;
	  }

          /** perform Russian Roulette*/
          if(p.w < gcfg->minenergy){
                if(rand_do_roulette(t)*ROULETTE_SIZE<=1.f)
                   p.w*=ROULETTE_SIZE;
                else{
                   GPUDEBUG(("relaunch after Russian roulette at idx=[%d] mediaid=[%d], ref=[%d]\n",idx1d,mediaid,gcfg->doreflect));
                   if(launchnewphoton<mcxsource>(&p,&v,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),ppath,
	                &energyloss,&energylaunched,n_det,detectedphoton,t,(RandType*)(sharedmem+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
			media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress))
                        break;
                   isdet=mediaid & DET_MASK;
                   mediaid &= MED_MASK;
                   continue;
               }
          }

          /** do boundary reflection/transmission */
	  if(((gcfg->doreflect && (isdet & 0xF)==0) || (isdet & 0x1)) && n1!=gproperty[mediaid].w){
	          float Rtotal=1.f;
	          float cphi,sphi,stheta,ctheta,tmp0,tmp1;

                  *((float4*)(&prop))=gproperty[mediaid]; ///< optical property across the interface

                  tmp0=n1*n1;
                  tmp1=prop.n*prop.n;
		  cphi=fabs( (flipdir==0) ? v.x : (flipdir==1 ? v.y : v.z)); // cos(si)
		  sphi=1.f-cphi*cphi;            // sin(si)^2

                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2 = cos(ti)^2
	          GPUDEBUG(("ref total ref=%f\n",len));

                  if(len>0.f) { ///< if no total internal reflection
                	ctheta=tmp0*cphi*cphi+tmp1*len;
                	stheta=2.f*n1*prop.n*cphi*sqrtf(len);
                	Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       		ctheta=tmp1*cphi*cphi+tmp0*len;
       	       		Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
	        	GPUDEBUG(("Rtotal=%f\n",Rtotal));
                  } ///< else, total internal reflection
	          if(Rtotal<1.f && (((isdet & 0xF)==0 && gproperty[mediaid].w>=1.f) || isdet==bcReflect) && rand_next_reflect(t)>Rtotal){ // do transmission
                        transmit(&v,n1,prop.n,flipdir);
                        if(mediaid==0){ // transmission to external boundary
                            GPUDEBUG(("transmit to air, relaunch\n"));
		    	    if(launchnewphoton<mcxsource>(&p,&v,&f,&rv,&prop,&idx1d,field,&mediaid,&w0,(mediaidold & DET_MASK),
			        ppath,&energyloss,&energylaunched,n_det,detectedphoton,t,(RandType*)(sharedmem+threadIdx.x*gcfg->issaveseed*RAND_BUF_LEN*sizeof(RandType)),
				media,srcpattern,idx,(RandType*)n_seed,seeddata,gdebugdata,gprogress))
                                break;
                            isdet=mediaid & DET_MASK;
                            mediaid &= MED_MASK;
			    continue;
			}
	                GPUDEBUG(("do transmission\n"));
                        rv=float3(__fdividef(1.f,v.x),__fdividef(1.f,v.y),__fdividef(1.f,v.z));
		  }else{ ///< do reflection
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
        	  	*((float4*)(&prop))=gproperty[mediaid];
                  	n1=prop.n;
		  }
	  }
     }

     /** return the tracked total energyloss and launched energy back to the host */
     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energylaunched;

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
     else          return 128;
}

/**
 * @brief Utility function to calculate the maximum blocks per SMX
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

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        MCX_FPRINTF(stderr,"No CUDA-capable GPU device found\n");
        return 0;
    }
    *info=(GPUInfo *)calloc(deviceCount,sizeof(GPUInfo));
    if (cfg->gpuid && cfg->gpuid > deviceCount){
        MCX_FPRINTF(stderr,"Specified GPU ID is out of range\n");
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
        (*info)[dev].autoblock=(*info)[dev].maxmpthread / mcx_smxblock(dp.major,dp.minor);
        (*info)[dev].autothread=(*info)[dev].autoblock * mcx_smxblock(dp.major,dp.minor) * (*info)[dev].sm;

        if (strncmp(dp.name, "Device Emulation", 16)) {
	  if(cfg->isgpuinfo){
	    MCX_FPRINTF(stdout,"=============================   GPU Infomation  ================================\n");
	    MCX_FPRINTF(stdout,"Device %d of %d:\t\t%s\n",(*info)[dev].id,(*info)[dev].devcount,(*info)[dev].name);
	    MCX_FPRINTF(stdout,"Compute Capability:\t%u.%u\n",(*info)[dev].major,(*info)[dev].minor);
	    MCX_FPRINTF(stdout,"Global Memory:\t\t%u B\nConstant Memory:\t%u B\n"
				"Shared Memory:\t\t%u B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
               (unsigned int)(*info)[dev].globalmem,(unsigned int)(*info)[dev].constmem,
               (unsigned int)(*info)[dev].sharedmem,(unsigned int)(*info)[dev].regcount,(*info)[dev].clock*1e-6f);
	  #if CUDART_VERSION >= 2000
	       MCX_FPRINTF(stdout,"Number of MPs:\t\t%u\nNumber of Cores:\t%u\n",
	          (*info)[dev].sm,(*info)[dev].core);
	  #endif
            MCX_FPRINTF(stdout,"SMX count:\t\t%u\n", (*info)[dev].sm);
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
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     float4 c0=cfg->srcdir;
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     float *energy;
     int timegate=0, totalgates, gpuid, threadid=0;
     size_t gpuphoton=0, photoncount=0;

     unsigned int printnum;
     unsigned int tic,tic0,tic1,toc=0,debuglen=MCX_DEBUG_REC_LEN;
     size_t fieldlen;
     uint3 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint4 dimlen;
     float Vvox,fullload=0.f;

     dim3 mcgrid, mcblock;
     dim3 clgrid, clblock;

     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     
     uint  *media=(uint *)(cfg->vol);
     float  *field;
     float4 *Ppos,*Pdir,*Plen,*Plen0;
     uint   *Pseed;
     float  *Pdet;
     RandType *seeddata=NULL;
     uint    detected=0,sharedbuf=0;

     volatile int *progress, *gprogress;
#ifndef WIN32
     cudaEvent_t updateprogress;
#endif

     uint *gmedia;
     float4 *gPpos,*gPdir,*gPlen;
     uint   *gPseed,*gdetected;
     int    *greplaydetid=NULL;
     float  *gPdet,*gsrcpattern,*gfield,*genergy,*greplayw=NULL,*greplaytof=NULL,*gdebugdata=NULL;
     RandType *gseeddata=NULL;
     int detreclen=2+(cfg->medianum-1)*(2+(cfg->ismomentum>0))+(cfg->issaveexit>0)*6;
     unsigned int is2d=(cfg->dim.x==1 ? 1 : (cfg->dim.y==1 ? 2 : (cfg->dim.z==1 ? 3 : 0)));
     MCXParam param={cfg->steps,minstep,0,0,cfg->tend,R_C0*cfg->unitinmm,
                     (uint)cfg->issave2pt,(uint)cfg->isreflect,(uint)cfg->isrefint,(uint)cfg->issavedet,1.f/cfg->tstep,
		     p0,c0,maxidx,uint4(0,0,0,0),cp0,cp1,uint2(0,0),cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->srctype,
		     cfg->srcparam1,cfg->srcparam2,cfg->voidtime,cfg->maxdetphoton,
		     cfg->medianum-1,cfg->detnum,cfg->maxgate,0,0,ABS(cfg->sradius+2.f)<EPS /*isatomic*/,
		     (uint)cfg->maxvoidstep,cfg->issaveseed>0,cfg->issaveexit>0,cfg->issaveref>0,cfg->ismomentum>0,cfg->isspecular>0,
		     cfg->maxdetphoton*detreclen,cfg->seed,(uint)cfg->outputtype,0,0,cfg->faststep,
		     cfg->debuglevel,(uint)cfg->maxjumpdebug,cfg->gscatter,is2d,cfg->replaydet};
     if(param.isatomic)
         param.skipradius2=0.f;
#ifdef _OPENMP
     threadid=omp_get_thread_num();
#endif
     if(threadid<MAX_DEVICE && cfg->deviceid[threadid]=='\0')
           return;

     gpuid=cfg->deviceid[threadid]-1;
     if(gpuid<0)
          mcx_error(-1,"GPU ID must be non-zero",__FILE__,__LINE__);
     CUDA_ASSERT(cudaSetDevice(gpuid));

     if(gpu[gpuid].maxgate==0 && dimxyz>0){
         int needmem=dimxyz+cfg->nthread*sizeof(float4)*4+sizeof(float)*cfg->maxdetphoton*detreclen+10*1024*1024; /*keep 10M for other things*/
         gpu[gpuid].maxgate=(gpu[gpuid].globalmem-needmem)/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
         gpu[gpuid].maxgate=MIN(((cfg->tend-cfg->tstart)/cfg->tstep+0.5),gpu[gpuid].maxgate);     
     }
     /*only allow the master thread to modify cfg, others are read-only*/
#pragma omp master
{
     if(cfg->exportfield==NULL){
         if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
	     cfg->exportfield=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*2*cfg->detnum);
	 }else{
             cfg->exportfield=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*2);
	 }
     }
     if(cfg->exportdetected==NULL)
         cfg->exportdetected=(float*)malloc(detreclen*cfg->maxdetphoton*sizeof(float));
     if(cfg->issaveseed && cfg->seeddata==NULL)
         cfg->seeddata=malloc(cfg->maxdetphoton*sizeof(RandType)*RAND_BUF_LEN);
     cfg->detectedcount=0;
     cfg->his.detected=0;
     cfg->his.respin=cfg->respin;
     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     cfg->runtime=0;
}
#pragma omp barrier

     if(is2d){
         float *vec=&(param.c0.x);
         if(ABS(vec[is2d-1])>EPS)
             mcx_error(-1,"input domain is 2D, the initial direction can not have non-zero value in the singular dimension",__FILE__,__LINE__);
     }
     if(!cfg->autopilot){
	gpu[gpuid].autothread=cfg->nthread;
	gpu[gpuid].autoblock=cfg->nblocksize;
	gpu[gpuid].maxgate=cfg->maxgate;
     }
     if(gpu[gpuid].autothread%gpu[gpuid].autoblock)
     	gpu[gpuid].autothread=(gpu[gpuid].autothread/gpu[gpuid].autoblock)*gpu[gpuid].autoblock;

     param.maxgate=gpu[gpuid].maxgate;

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

     gpuphoton=(double)cfg->nphoton*cfg->workload[threadid]/fullload;

     if(gpuphoton==0)
        return;

     if(cfg->respin>=1){
         param.threadphoton=gpuphoton/gpu[gpuid].autothread;
         param.oddphotons=gpuphoton-param.threadphoton*gpu[gpuid].autothread;
     }else if(cfg->respin<0){
         param.threadphoton=-gpuphoton/gpu[gpuid].autothread/cfg->respin;
         param.oddphotons=-gpuphoton/cfg->respin-param.threadphoton*gpu[gpuid].autothread;     
     }else{
         mcx_error(-1,"respin number can not be 0, check your -r/--repeat input or cfg.respin value",__FILE__,__LINE__);
     }
     totalgates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
#pragma omp master
     if(totalgates>gpu[gpuid].maxgate && cfg->isnormalized){
         MCX_FPRINTF(stderr,"WARNING: GPU memory can not hold all time gates, disabling normalization to allow multiple runs\n");
         cfg->isnormalized=0;
     }
#pragma omp barrier

     if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1)
         fieldlen=dimxyz*gpu[gpuid].maxgate*cfg->detnum;
     else
         fieldlen=dimxyz*gpu[gpuid].maxgate;

     mcgrid.x=gpu[gpuid].autothread/gpu[gpuid].autoblock;
     mcblock.x=gpu[gpuid].autoblock;

     clgrid.x=cfg->dim.x;
     clgrid.y=cfg->dim.y;
     clblock.x=cfg->dim.z;

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
           CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen));
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(float)*fieldlen)); // cost about 1 ms
           CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

           tic=StartTimer();
           MCX_FPRINTF(cfg->flog,"generating %lu random numbers ... \t",fieldlen); fflush(cfg->flog);
           mcx_test_rng<<<1,1>>>(gfield,gPseed);
           tic1=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving random numbers ... \t",tic1-tic);
           CUDA_ASSERT(cudaGetLastError());

           CUDA_ASSERT(cudaMemcpy(field, gfield,sizeof(float)*dimxyz*gpu[gpuid].maxgate,cudaMemcpyDeviceToHost));
           MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	   if(cfg->exportfield)
	       memcpy(cfg->exportfield,field,fieldlen*sizeof(float));
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

     Ppos=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread);
     Pdir=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread);
     Plen=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread);
     Plen0=(float4*)malloc(sizeof(float4)*gpu[gpuid].autothread);
     energy=(float*)calloc(gpu[gpuid].autothread<<1,sizeof(float));
     Pdet=(float*)calloc(cfg->maxdetphoton,sizeof(float)*(detreclen));
     if(cfg->seed!=SEED_FROM_FILE)
         Pseed=(uint*)malloc(sizeof(RandType)*gpu[gpuid].autothread*RAND_BUF_LEN);
     else
         Pseed=(uint*)malloc(sizeof(RandType)*cfg->nphoton*RAND_BUF_LEN);

     CUDA_ASSERT(cudaMalloc((void **) &gmedia, sizeof(uint)*(dimxyz)));
     //CUDA_ASSERT(cudaBindTexture(0, texmedia, gmedia));
     CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen*2));
     CUDA_ASSERT(cudaMalloc((void **) &gPpos, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdir, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPlen, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdet, sizeof(float)*cfg->maxdetphoton*(detreclen)));
     CUDA_ASSERT(cudaMalloc((void **) &gdetected, sizeof(uint)));
     CUDA_ASSERT(cudaMalloc((void **) &genergy, sizeof(float)*(gpu[gpuid].autothread<<1)));

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

     if(cfg->srctype==MCX_SRC_PATTERN)
         CUDA_ASSERT(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w)));
     else if(cfg->srctype==MCX_SRC_PATTERN3D)
         CUDA_ASSERT(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z)));
	 
#ifndef SAVE_DETECTORS
#pragma omp master
     if(cfg->issavedet){
           MCX_FPRINTF(stderr,"WARNING: this MCX binary can not save partial path, please use mcx_det or mcx_det_cached\n");
           cfg->issavedet=0;
     }
#pragma omp barrier
#endif

     /*volume is assumbed to be col-major*/
     cachebox.x=(cp1.x-cp0.x+1);
     cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
     dimlen.x=cfg->dim.x;
     dimlen.y=cfg->dim.y*cfg->dim.x;

     dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     dimlen.w=fieldlen;

     param.dimlen=dimlen;
     param.cachebox=cachebox;
     if(p0.x<0.f || p0.y<0.f || p0.z<0.f || p0.x>=cfg->dim.x || p0.y>=cfg->dim.y || p0.z>=cfg->dim.z){
         param.idx1dorig=0;
         param.mediaidorig=0;
     }else{
         param.idx1dorig=(int(floorf(p0.z))*dimlen.y+int(floorf(p0.y))*dimlen.x+int(floorf(p0.x)));
         param.mediaidorig=(cfg->vol[param.idx1dorig] & MED_MASK);
     }
     memcpy(&(param.bc),cfg->bc,8);
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
     tic=StartTimer();
#pragma omp master
{
     mcx_printheader(cfg);

#ifdef MCX_TARGET_NAME
     MCX_FPRINTF(cfg->flog,"- variant name: [%s] compiled for GPU Capability [%d] with CUDA [%d]\n",
         "Fermi",MCX_CUDA_ARCH,CUDART_VERSION);
#else
     MCX_FPRINTF(cfg->flog,"- code name: [Vanilla MCX] compiled for GPU Capacity [%d] with CUDA [%d]\n",
         MCX_CUDA_ARCH,CUDART_VERSION);
#endif
     MCX_FPRINTF(cfg->flog,"- compiled with: RNG [%s] with Seed Length [%d]\n",MCX_RNG_NAME,(int)((sizeof(RandType)*RAND_BUF_LEN)>>2));
#ifdef SAVE_DETECTORS
     MCX_FPRINTF(cfg->flog,"- this version CAN save photons at the detectors\n\n");
#else
     MCX_FPRINTF(cfg->flog,"- this version CAN NOT save photons at the detectors\n\n");
#endif
     fflush(cfg->flog);
}
#pragma omp barrier

     MCX_FPRINTF(cfg->flog,"\nGPU=%d (%s) threadph=%d extra=%d np=%ld nthread=%d maxgate=%d repetition=%d\n",gpuid+1,gpu[gpuid].name,param.threadphoton,param.oddphotons,
           gpuphoton,gpu[gpuid].autothread,gpu[gpuid].maxgate,ABS(cfg->respin));
     MCX_FPRINTF(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);

     mcx_flush(cfg);

     CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uint)*dimxyz, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpy(genergy,energy,sizeof(float) *(gpu[gpuid].autothread<<1), cudaMemcpyHostToDevice));
     if(cfg->srcpattern)
        if(cfg->srctype==MCX_SRC_PATTERN)
           CUDA_ASSERT(cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w), cudaMemcpyHostToDevice));
	else if(cfg->srctype==MCX_SRC_PATTERN3D)
	   CUDA_ASSERT(cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z), cudaMemcpyHostToDevice));
     

     CUDA_ASSERT(cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpyToSymbol(gproperty, cfg->detpos,  cfg->detnum*sizeof(float4), cfg->medianum*sizeof(Medium), cudaMemcpyHostToDevice));

     MCX_FPRINTF(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

     /*
         if one has to simulate a lot of time gates, using the GPU global memory
	 requires extra caution. If the total global memory is bigger than the total
	 memory to save all the snapshots, i.e. size(field)*(tend-tstart)/tstep, one
	 simply sets gpu[gpuid].maxgate to the total gate number; this will run GPU kernel
	 once. If the required memory is bigger than the video memory, set gpu[gpuid].maxgate
	 to a number which fits, and the snapshot will be saved with an increment of 
	 gpu[gpuid].maxgate snapshots. In this case, the later simulations will restart from
	 photon launching and exhibit redundancies.

	 The calculation of the energy conservation will only reflect the last simulation.
     */
     sharedbuf=gpu[gpuid].autoblock*(cfg->issaveseed*(RAND_BUF_LEN*sizeof(RandType))+sizeof(float)*((cfg->medianum-1)*(2+(cfg->ismomentum>0))+1));

     MCX_FPRINTF(cfg->flog,"requesting %d bytes of shared memory\n",sharedbuf);

     //simulate for all time-gates in maxgate groups per run
     for(timegate=0;timegate<totalgates;timegate+=gpu[gpuid].maxgate){

       param.twin0=cfg->tstart+cfg->tstep*timegate;
       param.twin1=param.twin0+cfg->tstep*gpu[gpuid].maxgate;
       CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param,     sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

       MCX_FPRINTF(cfg->flog,"lauching MCX simulation for time window [%.2ens %.2ens] ...\n"
           ,param.twin0*1e9,param.twin1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<ABS(cfg->respin);iter++){
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(float)*fieldlen*2)); // cost about 1 ms
           CUDA_ASSERT(cudaMemset(gPdet,0,sizeof(float)*cfg->maxdetphoton*(detreclen)));
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
           tic0=GetTimeMillis();
#ifndef WIN32
#pragma omp master
{
           if(cfg->debuglevel & MCX_DEBUG_PROGRESS)
               CUDA_ASSERT(cudaEventCreate(&updateprogress));
}
#endif
           MCX_FPRINTF(cfg->flog,"simulation run#%2d ... \n",iter+1); fflush(cfg->flog);
           mcx_flush(cfg);

	   switch(cfg->srctype) {
		case(MCX_SRC_PENCIL): mcx_main_loop<MCX_SRC_PENCIL> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_ISOTROPIC): mcx_main_loop<MCX_SRC_ISOTROPIC> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_CONE): mcx_main_loop<MCX_SRC_CONE> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_GAUSSIAN): mcx_main_loop<MCX_SRC_GAUSSIAN> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_PLANAR): mcx_main_loop<MCX_SRC_PLANAR> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_PATTERN): mcx_main_loop<MCX_SRC_PATTERN> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_FOURIER): mcx_main_loop<MCX_SRC_FOURIER> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_ARCSINE): mcx_main_loop<MCX_SRC_ARCSINE> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_DISK): mcx_main_loop<MCX_SRC_DISK> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_FOURIERX): mcx_main_loop<MCX_SRC_FOURIERX> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_FOURIERX2D): mcx_main_loop<MCX_SRC_FOURIERX2D> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_ZGAUSSIAN): mcx_main_loop<MCX_SRC_ZGAUSSIAN> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_LINE): mcx_main_loop<MCX_SRC_LINE> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_SLIT): mcx_main_loop<MCX_SRC_SLIT> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_PENCILARRAY): mcx_main_loop<MCX_SRC_PENCILARRAY> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
		case(MCX_SRC_PATTERN3D): mcx_main_loop<MCX_SRC_PATTERN3D> <<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,greplaydetid,gseeddata,gdebugdata,gprogress); break;
	   }

#pragma omp master
{
           if((param.debuglevel & MCX_DEBUG_PROGRESS)){
	     int p0 = 0, ndone=-1;
#ifndef WIN32
             CUDA_ASSERT(cudaEventRecord(updateprogress));
#endif
	     mcx_progressbar(-0.f,cfg);
	     do{
#ifndef WIN32
               cudaEventQuery(updateprogress);
#endif
               ndone = *progress;
	       if (ndone > p0){
		  mcx_progressbar(ndone/(param.threadphoton*1.45f),cfg);
		  p0 = ndone;
	       }
               sleep_ms(100);
	     }while (p0 < (param.threadphoton*1.45f));
             mcx_progressbar(1.0f,cfg);
             MCX_FPRINTF(cfg->flog,"\n");
             *progress=0;
           }
}
           CUDA_ASSERT(cudaThreadSynchronize());
	   CUDA_ASSERT(cudaMemcpy(&detected, gdetected,sizeof(uint),cudaMemcpyDeviceToHost));
           tic1=GetTimeMillis();
	   toc+=tic1-tic0;
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",tic1-tic);
           CUDA_ASSERT(cudaGetLastError());

           CUDA_ASSERT(cudaMemcpy(Plen0,  gPlen,  sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
           for(i=0;i<gpu[gpuid].autothread;i++)
	      photoncount+=int(Plen0[i].w+0.5f);

           if(cfg->debuglevel & MCX_DEBUG_MOVE){
               uint debugrec=0;
	       CUDA_ASSERT(cudaMemcpyFromSymbol(&debugrec, gjumpdebug,sizeof(uint),0,cudaMemcpyDeviceToHost));
#pragma omp critical
{
	       if(debugrec>0){
		   if(debugrec>cfg->maxdetphoton){
			MCX_FPRINTF(cfg->flog,"WARNING: the saved trajectory positions (%d) \
are more than what your have specified (%d), please use the --maxjumpdebug option to specify a greater number\n"
                           ,debugrec,cfg->maxjumpdebug);
		   }else{
			MCX_FPRINTF(cfg->flog,"saved %ud trajectory positions, total: %d\t",debugrec,cfg->maxjumpdebug+debugrec);
		   }
                   debugrec=min(debugrec,cfg->maxjumpdebug);
	           cfg->exportdebugdata=(float*)realloc(cfg->exportdebugdata,(cfg->debugdatalen+debugrec)*debuglen*sizeof(float));
                   CUDA_ASSERT(cudaMemcpy(cfg->exportdebugdata+cfg->debugdatalen, gdebugdata,sizeof(float)*debuglen*debugrec,cudaMemcpyDeviceToHost));
                   cfg->debugdatalen+=debugrec;
	       }
}
           }
#ifdef SAVE_DETECTORS
           if(cfg->issavedet){
           	CUDA_ASSERT(cudaMemcpy(Pdet, gPdet,sizeof(float)*cfg->maxdetphoton*(detreclen),cudaMemcpyDeviceToHost));
	        CUDA_ASSERT(cudaGetLastError());
		if(cfg->issaveseed)
		    CUDA_ASSERT(cudaMemcpy(seeddata, gseeddata,sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN,cudaMemcpyDeviceToHost));
		if(detected>cfg->maxdetphoton){
			MCX_FPRINTF(cfg->flog,"WARNING: the detected photon (%d) \
is more than what your have specified (%d), please use the -H option to specify a greater number\t"
                           ,detected,cfg->maxdetphoton);
		}else{
			MCX_FPRINTF(cfg->flog,"detected %d photons, total: %ld\t",detected,cfg->detectedcount+detected);
		}
#pragma omp atomic
                cfg->his.detected+=detected;
                detected=MIN(detected,cfg->maxdetphoton);
		if(cfg->exportdetected){
#pragma omp critical
{
                        cfg->exportdetected=(float*)realloc(cfg->exportdetected,(cfg->detectedcount+detected)*detreclen*sizeof(float));
			if(cfg->issaveseed && cfg->seeddata)
			    cfg->seeddata=(RandType*)realloc(cfg->seeddata,(cfg->detectedcount+detected)*sizeof(RandType)*RAND_BUF_LEN);
	                memcpy(cfg->exportdetected+cfg->detectedcount*(detreclen),Pdet,detected*(detreclen)*sizeof(float));
			if(cfg->issaveseed && cfg->seeddata)
			    memcpy(((RandType*)cfg->seeddata)+cfg->detectedcount*RAND_BUF_LEN,seeddata,detected*sizeof(RandType)*RAND_BUF_LEN);
                        cfg->detectedcount+=detected;
}
		}
	   }
#endif
           mcx_flush(cfg);

	   //handling the 2pt distributions
           if(cfg->issave2pt){
	       float *rawfield=(float*)malloc(sizeof(float)*fieldlen*2);
               CUDA_ASSERT(cudaMemcpy(rawfield, gfield,sizeof(float)*fieldlen*2,cudaMemcpyDeviceToHost));
               MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	       for(i=0;i<(int)fieldlen;i++)  //accumulate field, can be done in the GPU
	           field[i]=rawfield[i]+rawfield[i+fieldlen];
	       free(rawfield);

               if(ABS(cfg->respin)>1){
                   for(i=0;i<(int)fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
           }
       } /*end of respin loop*/

#pragma omp critical
       if(cfg->runtime<toc)
           cfg->runtime=toc;

       if(ABS(cfg->respin)>1)  //copy the accumulated fields back
           memcpy(field,field+fieldlen,sizeof(float)*fieldlen);

       if(cfg->isnormalized){
           CUDA_ASSERT(cudaMemcpy(energy,genergy,sizeof(float)*(gpu[gpuid].autothread<<1),cudaMemcpyDeviceToHost));
#pragma omp critical
{
           for(i=0;i<gpu[gpuid].autothread;i++){
               cfg->energyesc+=energy[i<<1];
       	       cfg->energytot+=energy[(i<<1)+1];
           }
	   for(i=0;i<gpu[gpuid].autothread;i++)
               cfg->energyabs+=Plen0[i].z;  // the accumulative absorpted energy near the source
}
       }
       MCX_FPRINTF(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);

       if(cfg->exportfield){
	       for(i=0;i<(int)fieldlen;i++)
#pragma omp atomic
                  cfg->exportfield[i]+=field[i];
       }

       if(param.twin1<cfg->tend){
            CUDA_ASSERT(cudaMemset(genergy,0,sizeof(float)*(gpu[gpuid].autothread<<1)));
       }
     } /*end of time-gate group loop*/
#pragma omp barrier

     /*let the master thread to deal with the normalization and file IO*/
#pragma omp master
{
     if(cfg->isnormalized){
	   float scale=1.f;
	   int isnormalized=0;
           MCX_FPRINTF(cfg->flog,"normalizing raw data ...\t");
           cfg->energyabs+=cfg->energytot-cfg->energyesc;
           if(cfg->outputtype==otFlux || cfg->outputtype==otFluence){
               scale=cfg->unitinmm/(cfg->energytot*Vvox*cfg->tstep); /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

               if(cfg->outputtype==otFluence)
		   scale*=cfg->tstep;
	   }else if(cfg->outputtype==otEnergy)
	       scale=1.f/cfg->energytot;
	   else if(cfg->outputtype==otJacobian || cfg->outputtype==otWP || cfg->outputtype==otDCS){
	       if(cfg->seed==SEED_FROM_FILE && cfg->replaydet==-1){
                   int detid;
		   for(detid=1;detid<=(int)cfg->detnum;detid++){
	               scale=0.f; // the cfg->normalizer and cfg.his.normalizer are inaccurate in this case, but this is ok
		       for(size_t i=0;i<cfg->nphoton;i++)
		           if(cfg->replay.detid[i]==detid)
	                       scale+=cfg->replay.weight[i];
	               if(scale>0.f)
	                   scale=cfg->unitinmm/scale;
                       MCX_FPRINTF(cfg->flog,"normalization factor for detector %d alpha=%f\n",detid, scale);  fflush(cfg->flog);
                       mcx_normalize(cfg->exportfield+(detid-1)*dimxyz*gpu[gpuid].maxgate,scale,dimxyz*gpu[gpuid].maxgate,cfg->isnormalized);
		   }
		   isnormalized=1;
	       }else{
	           scale=0.f;
	           for(size_t i=0;i<cfg->nphoton;i++)
	               scale+=cfg->replay.weight[i];
	           if(scale>0.f)
                       scale=cfg->unitinmm/scale;
	       }
           }
         cfg->normalizer=scale;
	 cfg->his.normalizer=scale;
         if(!isnormalized){
             MCX_FPRINTF(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
	     mcx_normalize(cfg->exportfield,scale,fieldlen,cfg->isnormalized);
	 }
     }
     if(cfg->issave2pt && cfg->parentid==mpStandalone){
         MCX_FPRINTF(cfg->flog,"saving data to file ... %lu %d\t",fieldlen,gpu[gpuid].maxgate);
         mcx_savedata(cfg->exportfield,fieldlen,cfg);
         MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
         fflush(cfg->flog);
     }
     if(cfg->issavedet && cfg->parentid==mpStandalone && cfg->exportdetected){
         cfg->his.unitinmm=cfg->unitinmm;
         cfg->his.savedphoton=cfg->detectedcount;
	 cfg->his.totalphoton=cfg->nphoton;
         if(cfg->issaveseed)
             cfg->his.seedbyte=sizeof(RandType)*RAND_BUF_LEN;

         cfg->his.detected=cfg->detectedcount;
         mcx_savedetphoton(cfg->exportdetected,cfg->seeddata,cfg->detectedcount,0,cfg);
     }
     if((cfg->debuglevel & MCX_DEBUG_MOVE) && cfg->parentid==mpStandalone && cfg->exportdebugdata){
         cfg->his.colcount=MCX_DEBUG_REC_LEN;
         cfg->his.savedphoton=cfg->debugdatalen;
	 cfg->his.totalphoton=cfg->nphoton;
         cfg->his.detected=0;
         mcx_savedetphoton(cfg->exportdebugdata,NULL,cfg->debugdatalen,0,cfg);
     }
}
#pragma omp barrier

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
     // total energy here equals total simulated photons+unfinished photons for all threads
     MCX_FPRINTF(cfg->flog,"simulated %ld photons (%ld) with %d threads (repeat x%d)\nMCX simulation speed: %.2f photon/ms\n",
             (long int)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1),(long int)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1),
	     gpu[gpuid].autothread,ABS(cfg->respin),(double)cfg->nphoton*((cfg->respin>1) ? (cfg->respin) : 1)/max(1,cfg->runtime)); fflush(cfg->flog);
     MCX_FPRINTF(cfg->flog,"total simulated energy: %.2f\tabsorbed: %5.5f%%\n(loss due to initial specular reflection is excluded in the total)\n",
             cfg->energytot,(cfg->energytot-cfg->energyesc)/cfg->energytot*100.f);fflush(cfg->flog);
     fflush(cfg->flog);
     
     cfg->energyabs=cfg->energytot-cfg->energyesc;
}
#pragma omp barrier

     CUDA_ASSERT(cudaFree(gmedia));
     CUDA_ASSERT(cudaFree(gfield));
     CUDA_ASSERT(cudaFree(gPpos));
     CUDA_ASSERT(cudaFree(gPdir));
     CUDA_ASSERT(cudaFree(gPlen));
     CUDA_ASSERT(cudaFree(gPseed));
     CUDA_ASSERT(cudaFree(genergy));
     CUDA_ASSERT(cudaFree(gPdet));
     CUDA_ASSERT(cudaFree(gdetected));
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

     CUDA_ASSERT(cudaDeviceReset());

     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Plen0);
     free(Pseed);
     free(Pdet);
     free(energy);
     free(field);
}
