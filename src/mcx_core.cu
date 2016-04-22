////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
//  Author: Qianqian Fang <q.fang at neu.edu>
//
//  Reference (Fang2009):
//        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
//        Migration in 3D Turbid Media Accelerated by Graphics Processing 
//        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
//
//  mcx_core.cu: GPU kernels and CUDA host code
//
//  License: GNU General Public License v3, see LICENSE.txt for details
//
////////////////////////////////////////////////////////////////////////////////

#include "br2cu.h"
#include "mcx_core.h"
#include "tictoc.h"
#include "mcx_const.h"

#if defined(USE_XORSHIFT128P_RAND)
    #include "xorshift128p_rand.cu" // use xorshift128+ RNG (XORSHIFT128P)
#elif defined(USE_POSIX_RAND)
    #include "posix_rand.cu"        // use POSIX erand48 RNG (POSIX)
#elif defined(USE_MT_RAND)
    #include "mt_rand_s.cu"         // use Mersenne Twister RNG (MT), depreciated
#else
    #include "logistic_rand.cu"     // use Logistic Lattice ring 5 RNG (LL5)
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#define CUDA_ASSERT(a)      mcx_cu_assess((a),__FILE__,__LINE__)

// optical properties saved in the constant memory
// {x}:mua,{y}:mus,{z}:anisotropy (g),{w}:refractive index (n)
__constant__ float4 gproperty[MAX_PROP];

__constant__ float4 gdetpos[MAX_DETECTORS];

// kernel constant parameters
__constant__ MCXParam gcfg[1];

extern __shared__ float sharedmem[]; //max 64 tissue types when block size=64

// tested with texture memory for media, only improved 1% speed
// to keep code portable, use global memory for now
// also need to change all media[idx1d] to tex1Dfetch() below
//texture<uchar, 1, cudaReadModeElementType> texmedia;

__device__ inline void atomicadd(float* address, float value){

#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats

  atomicAdd(address,value);

#elif __CUDA_ARCH__ >= 110

// float-atomic-add from 
// http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);

#endif

}

__device__ inline void clearpath(float *p,int maxmediatype){
      uint i;
      for(i=0;i<maxmediatype;i++)
      	   p[i]=0.f;
}

__device__ inline void clearcache(float *p,int len){
      uint i;
      if(threadIdx.x==0)
        for(i=0;i<len;i++)
      	   p[i]=0.f;
}

#ifdef  USE_CACHEBOX
__device__ inline void savecache(float *data,float *cache){
      uint x,y,z;
      if(threadIdx.x==0){
        for(z=gcfg->cp0.z;z<=gcfg->cp1.z;z++)
           for(y=gcfg->cp0.y;y<=gcfg->cp1.y;y++)
              for(x=gcfg->cp0.x;x<=gcfg->cp1.x;x++){
                 atomicadd(data+z*gcfg->dimlen.y+y*gcfg->dimlen.x+x,
		    cache[(z-gcfg->cp0.z)*gcfg->cachebox.y+(y-gcfg->cp0.y)*gcfg->cachebox.x+(x-gcfg->cp0.x)]);
	      }
      }
}
#endif

#ifdef SAVE_DETECTORS
__device__ inline uint finddetector(MCXpos *p0){
      uint i;
      for(i=0;i<gcfg->detnum;i++){
      	if((gdetpos[i].x-p0->x)*(gdetpos[i].x-p0->x)+
	   (gdetpos[i].y-p0->y)*(gdetpos[i].y-p0->y)+
	   (gdetpos[i].z-p0->z)*(gdetpos[i].z-p0->z) < gdetpos[i].w*gdetpos[i].w){
	        return i+1;
	   }
      }
      return 0;
}

__device__ inline void savedetphoton(float n_det[],uint *detectedphoton,float nscat,float *ppath,MCXpos *p0,RandType t[RAND_BUF_LEN],RandType *seeddata){
      uint detid;
      detid=finddetector(p0);
      if(detid){
	 uint baseaddr=atomicAdd(detectedphoton,1);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    for(i=0;i<gcfg->issaveseed*RAND_BUF_LEN;i++)
	        seeddata[baseaddr*RAND_BUF_LEN+i]=t[i]; // save photon seed for replay
	    baseaddr*=gcfg->maxmedia+2;
	    n_det[baseaddr++]=detid;
	    n_det[baseaddr++]=nscat;
	    for(i=0;i<gcfg->maxmedia;i++)
		n_det[baseaddr+i]=ppath[i]; // save partial pathlength to the memory
	 }
      }
}
#endif

__device__ inline float mcx_nextafterf(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a+gcfg->maxvoidstep;
      num.i+=dir ^ (num.i & 0x80000000U);
      return num.f-gcfg->maxvoidstep;
}

__device__ inline float hitgrid(float3 *p0, float3 *v, float *htime,float* rv,int *id){
      float dist;

      //time-of-flight to hit the wall in each direction
      htime[0]=fabs((floorf(p0->x)+(v->x>0.f)-p0->x)*rv[0]); // absolute distance of travel in x/y/z
      htime[1]=fabs((floorf(p0->y)+(v->y>0.f)-p0->y)*rv[1]);
      htime[2]=fabs((floorf(p0->z)+(v->z>0.f)-p0->z)*rv[2]);

      //get the direction with the smallest time-of-flight
      dist=fminf(fminf(htime[0],htime[1]),htime[2]);
      (*id)=(dist==htime[0]?0:(dist==htime[1]?1:2));

      //p0 is inside, p is outside, move to the 1st intersection pt, now in the air side, to be corrected in the else block
      htime[0]=p0->x+dist*v->x;
      htime[1]=p0->y+dist*v->y;
      htime[2]=p0->z+dist*v->z;

      (*id==0) ?
          (htime[0]=mcx_nextafterf(__float2int_rn(htime[0]), (v->x > 0.f)-(v->x < 0.f))) :
	  ((*id==1) ? 
	      (htime[1]=mcx_nextafterf(__float2int_rn(htime[1]), (v->y > 0.f)-(v->y < 0.f))) :
	      (htime[2]=mcx_nextafterf(__float2int_rn(htime[2]), (v->z > 0.f)-(v->z < 0.f))) );

      return dist;
}

__device__ inline void transmit(MCXdir *v, float n1, float n2,int flipdir){
      float tmp0=n1/n2;
      v->x*=tmp0;
      v->y*=tmp0;
      v->z*=tmp0;
      (flipdir==0) ?
          (v->x=sqrtf(1.f - v->y*v->y - v->z*v->z)*((v->x>0.f)-(v->x<0.f))):
	  ((flipdir==1) ? 
	      (v->y=sqrtf(1.f - v->x*v->x - v->z*v->z)*((v->y>0.f)-(v->y<0.f))):
	      (v->z=sqrtf(1.f - v->x*v->x - v->y*v->y)*((v->z>0.f)-(v->z<0.f))));
}

__device__ inline float reflectcoeff(MCXdir *v, float n1, float n2, int flipdir){
      float Icos=fabs((flipdir==0) ? v->x : (flipdir==1 ? v->y : v->z));
      float tmp0=n1*n1;
      float tmp1=n2*n2;
      float tmp2=1.f-tmp0/tmp1*(1.f-Icos*Icos); /*1-[n1/n2*sin(si)]^2 = cos(ti)^2*/
      if(tmp2>0.f){ // partial reflection
          float Re,Im,Rtotal;
	  Re=tmp0*Icos*Icos+tmp1*tmp2;
	  tmp2=sqrtf(tmp2); /*to save one sqrt*/
	  Im=2.f*n1*n2*Icos*tmp2;
	  Rtotal=(Re-Im)/(Re+Im);     /*Rp*/
	  Re=tmp1*Icos*Icos+tmp0*tmp2*tmp2;
	  Rtotal=(Rtotal+(Re-Im)/(Re+Im))*0.5f; /*(Rp+Rs)/2*/
	  return Rtotal;
      }else{ // total reflection
          return 1.f;
      }
}

/* if the source location is outside of the volume or 
in an void voxel, mcx advances the photon in v.{xyz} direction
until it hits an non-zero voxel */
__device__ inline int skipvoid(MCXpos *p,MCXdir *v,MCXtime *f,float3* rv,uchar media[]){
      int count=1,idx1d;
      while(1){
          if(p->x>=0.f && p->y>=0.f && p->z>=0.f && p->x < gcfg->maxidx.x
               && p->y < gcfg->maxidx.y && p->z < gcfg->maxidx.z){
	    idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	    if(media[idx1d] & MED_MASK){ // if inside
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

		if(gproperty[media[idx1d] & MED_MASK].w!=gproperty[0].w){
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

__device__ inline int launchnewphoton(MCXpos *p,MCXdir *v,MCXtime *f,float3* rv,Medium *prop,uint *idx1d,
           uint *mediaid,float *w0,float *Lmove,uint isdet, float ppath[],float energyloss[],float energylaunched[],float n_det[],uint *dpnum,
	   RandType t[RAND_BUF_LEN],RandType photonseed[RAND_BUF_LEN],
	   uchar media[],float srcpattern[],int threadid,RandType rngseed[],RandType seeddata[]){
      int launchattempt=1;

      if(p->w>=0.f){
          *energyloss+=p->w;  // sum all the remaining energy
#ifdef SAVE_DETECTORS
      // let's handle detectors here
          if(gcfg->savedet){
             if(isdet && *mediaid==0)
	         savedetphoton(n_det,dpnum,v->nscat,ppath,p,photonseed,seeddata);
             clearpath(ppath,gcfg->maxmedia);
          }
#endif
      }

      if((int)(f->ndone)>=(gcfg->threadphoton+(threadid<gcfg->oddphotons))){
          return 1; // all photos complete
      }
      if(gcfg->seed==SEED_FROM_FILE){
          int seedoffset=(threadid*gcfg->threadphoton+min(threadid,gcfg->oddphotons-1)+(int)f->ndone)*RAND_BUF_LEN;
          for(int i=0;i<RAND_BUF_LEN;i++)
	      t[i]=rngseed[seedoffset+i];
      }
      do{
	  *((float4*)p)=gcfg->ps;
	  *((float4*)v)=gcfg->c0;
	  *((float4*)f)=float4(0.f,0.f,gcfg->minaccumtime,f->ndone);
          *idx1d=gcfg->idx1dorig;
          *mediaid=gcfg->mediaidorig;
	  if(gcfg->issaveseed)
              copystate(t,photonseed);

	  if(gcfg->srctype==MCX_SRC_PENCIL){ /*source can be outside*/
	      // do nothing
	  }else if(gcfg->srctype==MCX_SRC_PLANAR || gcfg->srctype==MCX_SRC_PATTERN|| gcfg->srctype==MCX_SRC_FOURIER){ /*a rectangular grid over a plane*/
	      float rx=rand_uniform01(t);
	      float ry=rand_uniform01(t);
	      *((float4*)p)=float4(p->x+rx*gcfg->srcparam1.x+ry*gcfg->srcparam2.x,
	                	   p->y+rx*gcfg->srcparam1.y+ry*gcfg->srcparam2.y,
				   p->z+rx*gcfg->srcparam1.z+ry*gcfg->srcparam2.z,
				   p->w);
              if(gcfg->srctype==MCX_SRC_PATTERN) // need to prevent rx/ry=1 here
        	  p->w=srcpattern[(int)(ry*JUST_BELOW_ONE*gcfg->srcparam2.w)*(int)(gcfg->srcparam1.w)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.w)];
	      else if(gcfg->srctype==MCX_SRC_FOURIER){
		  p->w=(cosf((floorf(gcfg->srcparam1.w)*rx+floorf(gcfg->srcparam2.w)*ry
		          +gcfg->srcparam1.w-floorf(gcfg->srcparam1.w))*TWO_PI)*(1.f-gcfg->srcparam2.w+floorf(gcfg->srcparam2.w))+1.f)*0.5f; //between 0 and 1
              }
              *idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
              if(p->x<0.f || p->y<0.f || p->z<0.f || p->x>=gcfg->maxidx.x || p->y>=gcfg->maxidx.y || p->z>=gcfg->maxidx.z){
        	  *mediaid=0;
              }else{
        	  *mediaid=media[*idx1d];
              }
	  }else if(gcfg->srctype==MCX_SRC_FOURIERX||gcfg->srctype==MCX_SRC_FOURIERX2D){ // [v1x][v1y][v1z][|v2|]; [kx][ky][phi0][M], unit(v0) x unit(v1)=unit(v2)
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
	  }else if(gcfg->srctype==MCX_SRC_DISK ||gcfg->srctype==MCX_SRC_GAUSSIAN){ // uniform disk distribution or Gaussian-beam
	      // Uniform disk point picking
	      // http://mathworld.wolfram.com/DiskPointPicking.html
	      float sphi, cphi;
	      float phi=TWO_PI*rand_uniform01(t);
              sincosf(phi,&sphi,&cphi);
	     float r;
	     if(gcfg->srctype==MCX_SRC_DISK)
		 r=sqrtf(rand_uniform01(t))*gcfg->srcparam1.x;
	     else
		 r=sqrtf(-logf(rand_uniform01(t)))*gcfg->srcparam1.x;

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
	  }else if(gcfg->srctype==MCX_SRC_CONE || gcfg->srctype==MCX_SRC_ISOTROPIC || gcfg->srctype==MCX_SRC_ARCSINE){
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
	  }else if(gcfg->srctype==MCX_SRC_ZGAUSSIAN){
              float ang,stheta,ctheta,sphi,cphi;
	      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
	      sincosf(ang,&sphi,&cphi);
              ang=sqrtf(-2.f*logf(rand_uniform01(t)))*(1.f-2.f*rand_uniform01(t))*gcfg->srcparam1.x;
	      sincosf(ang,&stheta,&ctheta);
	      rotatevector(v,stheta,ctheta,sphi,cphi);
	  }else if(gcfg->srctype==MCX_SRC_LINE || gcfg->srctype==MCX_SRC_SLIT){
	      float r=rand_uniform01(t);
	      *((float4*)p)=float4(p->x+r*gcfg->srcparam1.x,
	                	   p->y+r*gcfg->srcparam1.y,
				   p->z+r*gcfg->srcparam1.z,
				   p->w);
              if(gcfg->srctype==MCX_SRC_LINE){
	              float s,p;
		      r=1.f-2.f*rand_uniform01(t);
		      s=1.f-2.f*rand_uniform01(t);
		      p=sqrt(1.f-v->x*v->x-v->y*v->y)*(rand_uniform01(t)>0.5f ? 1.f : -1.f);
		      *((float4*)v)=float4(v->y*p-v->z*s,v->z*r-v->x*p,v->x*s-v->y*r,v->nscat);
	      }
	  }
          *rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));
	  if((*mediaid & MED_MASK)==0){
             int idx=skipvoid(p, v, f, rv, media); /*specular reflection of the bbx is taken care of here*/
             if(idx>=0){
		 *idx1d=idx;
		 *mediaid=media[*idx1d];
	     }
	  }
	  if(launchattempt++>gcfg->maxvoidstep)
	     return -1;  // launch failed
      }while((*mediaid & MED_MASK)==0 || p->w<=gcfg->minenergy);
      f->ndone++; // launch successfully
      *((float4*)(prop))=gproperty[*mediaid & MED_MASK]; //always use mediaid to read gproperty[]
      
      /*total energy enters the volume. for diverging/converting 
      beams, this is less than nphoton due to specular reflection 
      loss. This is different from the wide-field MMC, where the 
      total launched energy includes the specular reflection loss*/
      
      *energylaunched+=p->w;
      *w0=p->w;
      *Lmove=0.f;
      return 0;
}

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
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
   everything in the GPU kernels is in grid-unit. To convert back to length, use
   cfg->unitinmm (scattering/absorption coeff, T, speed etc)
*/
kernel void mcx_main_loop(uchar media[],float field[],float genergy[],uint n_seed[],
     float4 n_pos[],float4 n_dir[],float4 n_len[],float n_det[], uint detectedphoton[], 
     float srcpattern[],float replayweight[],float photontof[],RandType *seeddata){

     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     MCXpos  p={0.f,0.f,0.f,-1.f};//{x,y,z}: coordinates in grid unit, w:packet weight
     MCXdir *v=(MCXdir*)(sharedmem+(threadIdx.x<<2));   //{x,y,z}: unitary direction vector in grid unit, nscat:total scat event
     MCXtime f;   //pscat: remaining scattering probability,t: photon elapse time, 
                  //tnext: next accumulation time, ndone: completed photons
     float  energyloss=genergy[idx<<1];
     float  energylaunched=genergy[(idx<<1)+1];

     uint idx1d, idx1dold;   //idx1dold is related to reflection
     uint moves=0;

#ifdef TEST_RACING
     int cc=0;
#endif
     uint  mediaid=gcfg->mediaidorig;
     uint  mediaidold=0,isdet=0;
     float  n1;   //reflection var
     float3 htime;            //time-of-fly for collision test
     float3 rv;               //reciprocal velocity

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN];
     RandType photonseed[RAND_BUF_LEN];
     Medium prop;    //can become float2 if no reflection (mua/musp is in 1/grid unit)

     float len, slen;
     float w0,Lmove;
     int   flipdir=-1;
 
     float *ppath=sharedmem+(blockDim.x<<2); // first blockDim.x<<2 stores v for all threads
#ifdef  USE_CACHEBOX
  #ifdef  SAVE_DETECTORS
     float *cachebox=ppath+(gcfg->savedet ? blockDim.x*gcfg->maxmedia: 0);
  #else
     float *cachebox=ppath;
  #endif
     if(gcfg->skipradius2>EPS) clearcache(cachebox,(gcfg->cp1.x-gcfg->cp0.x+1)*(gcfg->cp1.y-gcfg->cp0.y+1)*(gcfg->cp1.z-gcfg->cp0.z+1));
#else
     float accumweight=0.f;
#endif

#ifdef  SAVE_DETECTORS
     ppath+=threadIdx.x*gcfg->maxmedia; // block#2: maxmedia*thread number to store the partial
     if(gcfg->savedet) clearpath(ppath,gcfg->maxmedia);
#endif

     gpu_rng_init(t,n_seed,idx);

     if(launchnewphoton(&p,v,&f,&rv,&prop,&idx1d,&mediaid,&w0,&Lmove,0,ppath,&energyloss,
       &energylaunched,n_det,detectedphoton,t,photonseed,media,srcpattern,
       idx,(RandType*)n_seed,seeddata)){
         n_seed[idx]=NO_LAUNCH;
	 n_pos[idx]=*((float4*)(&p));
	 n_dir[idx]=*((float4*)(v));
	 n_len[idx]=*((float4*)(&f));
         return;
     }
     rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));
     isdet=mediaid & DET_MASK;
     mediaid &= MED_MASK; // keep isdet to 0 to avoid launching photon ina 

     /*
      using a while-loop to terminate a thread by np.will cause MT RNG to be 3.5x slower
      LL5 RNG will only be slightly slower than for-loop.with photon-move criterion

      we have switched to while-loop since v0.4.9, as LL5 was only minimally effected
      and we do not use MT as the default RNG.
     */

     while(f.ndone<=(gcfg->threadphoton+(idx<gcfg->oddphotons))) {

          GPUDEBUG(("photonid [%d] L=%f w=%e medium=%d\n",(int)f.ndone,f.pscat,p.w,mediaid));

          // dealing with scattering

	  if(f.pscat<=0.f) {  // if this photon has finished his current jump, get next scat length & angles
               if(moves++>gcfg->reseedlimit){
                  moves=0;
                  gpu_rng_reseed(t,n_seed,idx,(p.x+p.y+p.z+p.w)+f.ndone*(v->x+v->y+v->z));
               }
   	       f.pscat=rand_next_scatlen(t); // random scattering probability, unit-less

               GPUDEBUG(("scat L=%f RNG=[%0lX %0lX] \n",f.pscat,t[0],t[1]));
	       if(p.w<1.f){ // if this is not my first jump
                       //random arimuthal angle
	               float cphi,sphi,theta,stheta,ctheta;
                       float tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                       sincosf(tmp0,&sphi,&cphi);
                       GPUDEBUG(("scat phi=%f\n",tmp0));

                       //Henyey-Greenstein Phase Function, "Handbook of Optical 
                       //Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002

                       if(prop.g>EPS){  //if prop.g is too small, the distribution of theta is bad
		           tmp0=(1.f-prop.g*prop.g)/(1.f-prop.g+2.f*prop.g*rand_next_zangle(t));
		           tmp0*=tmp0;
		           tmp0=(1.f+prop.g*prop.g-tmp0)/(2.f*prop.g);

                           // when ran=1, CUDA gives me 1.000002 for tmp0 which produces nan later
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
                       rotatevector(v,stheta,ctheta,sphi,cphi);
                       v->nscat++;
                       rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));
	       }
	  }

          n1=prop.n;
	  *((float4*)(&prop))=gproperty[mediaid & MED_MASK];
	  
	  len=(gcfg->faststep) ? gcfg->minstep : hitgrid((float3*)&p,(float3*)v,&(htime.x),&rv.x,&flipdir); // propagate the photon to the first intersection to the grid
	  slen=len*prop.mus; //unitless (minstep=grid, mus=1/grid)

          GPUDEBUG(("p=[%f %f %f] -> <%f %f %f>*%f -> hit=[%f %f %f] flip=%d\n",p.x,p.y,p.z,v->x,v->y,v->z,len,htime.x,htime.y,htime.z,flipdir));

          // dealing with absorption
	  slen=min(slen,f.pscat);
	  len=slen/prop.mus;
	  *((float3*)(&p)) = (gcfg->faststep || slen==f.pscat) ? float3(p.x+len*v->x,p.y+len*v->y,p.z+len*v->z) : float3(htime.x,htime.y,htime.z);
	  p.w*=expf(-prop.mua*len);
	  f.pscat-=slen;     //remaining probability: sum(s_i*mus_i), unit-less
	  f.t+=len*prop.n*gcfg->oneoverc0; //propagation time  (unit=s)
	  Lmove+=len;

          GPUDEBUG(("update p=[%f %f %f] -> len=%f\n",p.x,p.y,p.z,len));

#ifdef SAVE_DETECTORS
          if(gcfg->savedet)
	      ppath[(mediaid & MED_MASK)-1]+=len; //(unit=grid)
#endif

          mediaidold=mediaid | isdet;
          idx1dold=idx1d;
          idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
          GPUDEBUG(("idx1d [%d]->[%d]\n",idx1dold,idx1d));
          if(p.x<0||p.y<0||p.z<0||p.x>=gcfg->maxidx.x||p.y>=gcfg->maxidx.y||p.z>=gcfg->maxidx.z){
	      mediaid=0;
	  }else{
	      mediaid=media[idx1d];
	      isdet=mediaid & DET_MASK;
	      mediaid &= MED_MASK;
          }
          GPUDEBUG(("medium [%d]->[%d]\n",mediaidold,mediaid));

          // saving fluence to the voxel when moving out

	  if(idx1d!=idx1dold && idx1dold>0 && mediaidold){
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep.wide
             if(gcfg->save2pt && f.t>=gcfg->twin0 && f.t<gcfg->twin1){
	          float weight;
                  int tshift=(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep));
		  if(gcfg->outputtype==otEnergy)
		      weight=w0-p.w;
		  else if(gcfg->seed==SEED_FROM_FILE && gcfg->outputtype==otJacobian){
		      weight=replayweight[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]*Lmove;
                      tshift=(int)(floorf((photontof[(idx*gcfg->threadphoton+min(idx,gcfg->oddphotons-1)+(int)f.ndone)]-gcfg->twin0)*gcfg->Rtstep));
		  }else
		      weight=(prop.mua==0.f) ? 0.f : ((w0-p.w)/(prop.mua));

                  GPUDEBUG(("deposit to [%d] %e, w=%f\n",idx1dold,weight,p.w));

#ifdef TEST_RACING
                  // enable TEST_RACING to determine how many missing accumulations due to race
                  if( (p.x-gcfg->ps.x)*(p.x-gcfg->ps.x)+(p.y-gcfg->ps.y)*(p.y-gcfg->ps.y)+(p.z-gcfg->ps.z)*(p.z-gcfg->ps.z)>gcfg->skipradius2) {
                      field[idx1dold+tshift*gcfg->dimlen.z]+=1.f;
		      cc++;
                  }
#else
  #ifdef USE_ATOMIC
                if(!gcfg->isatomic){
  #endif
                  // set gcfg->skipradius2 to only start depositing energy when dist^2>gcfg->skipradius2 
                  if(gcfg->skipradius2>EPS){
  #ifdef  USE_CACHEBOX
                      if(p.x<gcfg->cp1.x+1.f && p.x>=gcfg->cp0.x &&
		         p.y<gcfg->cp1.y+1.f && p.y>=gcfg->cp0.y &&
			 p.z<gcfg->cp1.z+1.f && p.z>=gcfg->cp0.z){
                         atomicadd(cachebox+(int(p.z-gcfg->cp0.z)*gcfg->cachebox.y
			      +int(p.y-gcfg->cp0.y)*gcfg->cachebox.x+int(p.x-gcfg->cp0.x)),weight);
  #else
                      if((p.x-gcfg->ps.x)*(p.x-gcfg->ps.x)+(p.y-gcfg->ps.y)*(p.y-gcfg->ps.y)+(p.z-gcfg->ps.z)*(p.z-gcfg->ps.z)<=gcfg->skipradius2){
                          accumweight+=p.w*prop.mua; // weight*absorption
  #endif
                      }else{
                          field[idx1dold+tshift*gcfg->dimlen.z]+=weight;
                      }
                  }else{
                      field[idx1dold+tshift*gcfg->dimlen.z]+=weight;
                  }
  #ifdef USE_ATOMIC
               }else{
                  // ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
		  atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], weight);
                  GPUDEBUG(("atomic write to [%d] %e, w=%f\n",idx1dold,weight,p.w));
               }
  #endif
#endif
	     }
	     w0=p.w;
	     Lmove=0.f;
             //f.tnext+=gcfg->minaccumtime*prop.n; // fluence is a temporal-integration, unit=s
	  }
	  
	  // launch new photon when exceed time window or moving from non-zero voxel to zero voxel without reflection

          if((mediaid==0 && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[mediaid].w))) || f.t>gcfg->twin1){
              GPUDEBUG(("direct relaunch at idx=[%d] mediaid=[%d], ref=[%d]\n",idx1d,mediaid,gcfg->doreflect));
	      if(launchnewphoton(&p,v,&f,&rv,&prop,&idx1d,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),ppath,
	          &energyloss,&energylaunched,n_det,detectedphoton,t,photonseed,media,srcpattern,idx,(RandType*)n_seed,seeddata))
                   break;
              isdet=mediaid & DET_MASK;
              mediaid &= MED_MASK;
	      continue;
	  }
          
          // do boundary reflection/transmission

	  if(gcfg->doreflect && n1!=gproperty[mediaid].w){
	          float Rtotal=1.f;
	          float cphi,sphi,stheta,ctheta,tmp0,tmp1;

                  *((float4*)(&prop))=gproperty[mediaid]; // optical property across the interface

                  tmp0=n1*n1;
                  tmp1=prop.n*prop.n;
		  cphi=fabs( (flipdir==0) ? v->x : (flipdir==1 ? v->y : v->z)); // cos(si)
		  sphi=1.f-cphi*cphi;            // sin(si)^2

                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2 = cos(ti)^2
	          GPUDEBUG(("ref total ref=%f\n",len));

                  if(len>0.f) { // if no total internal reflection
                	ctheta=tmp0*cphi*cphi+tmp1*len;
                	stheta=2.f*n1*prop.n*cphi*sqrtf(len);
                	Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       		ctheta=tmp1*cphi*cphi+tmp0*len;
       	       		Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
	        	GPUDEBUG(("Rtotal=%f\n",Rtotal));
                  } // else, total internal reflection
	          if(Rtotal<1.f && rand_next_reflect(t)>Rtotal){ // do transmission
                        if(mediaid==0){ // transmission to external boundary
                            GPUDEBUG(("transmit to air, relaunch\n"));
		    	    if(launchnewphoton(&p,v,&f,&rv,&prop,&idx1d,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),
			        ppath,&energyloss,&energylaunched,n_det,detectedphoton,t,photonseed,
				media,srcpattern,idx,(RandType*)n_seed,seeddata))
                                break;
                            isdet=mediaid & DET_MASK;
                            mediaid &= MED_MASK;
			    continue;
			}
	                GPUDEBUG(("do transmission\n"));
			transmit(v,n1,prop.n,flipdir);
                        rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));
		  }else{ //do reflection
	                GPUDEBUG(("ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n",flipdir,p.x,p.y,p.z,v->x,v->y,v->z));
			(flipdir==0) ? (v->x=-v->x) : ((flipdir==1) ? (v->y=-v->y) : (v->z=-v->z)) ;
                        rv=float3(__fdividef(1.f,v->x),__fdividef(1.f,v->y),__fdividef(1.f,v->z));
			(flipdir==0) ?
        		    (p.x=mcx_nextafterf(__float2int_rn(p.x), (v->x > 0.f)-(v->x < 0.f))) :
			    ((flipdir==1) ? 
				(p.y=mcx_nextafterf(__float2int_rn(p.y), (v->x > 0.f)-(v->x < 0.f))) :
				(p.z=mcx_nextafterf(__float2int_rn(p.z), (v->x > 0.f)-(v->x < 0.f))) );
	                GPUDEBUG(("ref p_new=[%f %f %f] v_new=[%f %f %f]\n",p.x,p.y,p.z,v->x,v->y,v->z));
                	idx1d=idx1dold;
		 	mediaid=(media[idx1d] & MED_MASK);
        	  	*((float4*)(&prop))=gproperty[mediaid];
                  	n1=prop.n;
		  }
	  }
     }
     // cachebox saves the total absorbed energy of all time in the sphere r<sradius.
     // in non-atomic mode, cachebox is more accurate than saving to the grid
     // as it is not influenced by race conditions.
     // now I borrow f.tnext to pass this value back
#ifdef  USE_CACHEBOX
     if(gcfg->skipradius2>EPS){
     	f.tnext=0.f;
        savecache(field,cachebox);
     }
#else
     f.tnext=accumweight;
#endif

     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energylaunched;

#ifdef TEST_RACING
     n_seed[idx]=cc;
#endif
     n_pos[idx]=*((float4*)(&p));
     n_dir[idx]=*((float4*)(v));
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
  obtain GPU core number per MP, this replaces 
  ConvertSMVer2Cores() in libcudautils to avoid 
  extra dependency.
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
  obtain GPU core number per MP, this replaces 
  ConvertSMVer2Cores() in libcudautils to avoid 
  extra dependency.
*/

int mcx_smxblock(int v1, int v2){
     int v=v1*10+v2;
     if(v<30)      return 8;
     else if(v<50) return 16;
     else          return 32;
}

/**
  query GPU info and set active GPU
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
	    MCX_FPRINTF(stdout,"Global Memory:\t\t%u B\nConstant Memory:\t%u B\n\
Shared Memory:\t\t%u B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
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
#ifdef USE_MT_RAND
    if(cfg->nblocksize>N-M){
        mcx_error(-1,"block size can not be larger than 227 when using MT19937 RNG",__FILE__,__LINE__);
    }
#endif

    return activedev;
#endif
}


/**
   host code for MCX kernels
*/
void mcx_run_simulation(Config *cfg,GPUInfo *gpu){

     int i,iter;
     float  minstep=1.f; //MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     float4 c0=float4(cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f);
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     float *energy;
     int timegate=0, totalgates, gpuid, gpuphoton=0,threadid=0;

     unsigned int photoncount=0,printnum;
     unsigned int tic,tic0,tic1,toc=0,fieldlen;
     uint3 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint3 dimlen;
     float Vvox,fullload=0.f;

     dim3 mcgrid, mcblock;
     dim3 clgrid, clblock;

     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     
     uchar  *media=(uchar *)(cfg->vol);
     float  *field;
     float4 *Ppos,*Pdir,*Plen,*Plen0;
     uint   *Pseed;
     float  *Pdet;
     RandType *seeddata=NULL;
     uint    detected=0,sharedbuf=0;

     uchar *gmedia;
     float4 *gPpos,*gPdir,*gPlen;
     uint   *gPseed,*gdetected;
     float  *gPdet,*gsrcpattern,*gfield,*genergy,*greplayw,*greplaytof;
     RandType *gseeddata=NULL;
     MCXParam param={cfg->steps,minstep,0,0,cfg->tend,R_C0*cfg->unitinmm,
                     (uint)cfg->issave2pt,(uint)cfg->isreflect,(uint)cfg->isrefint,(uint)cfg->issavedet,1.f/cfg->tstep,
		     p0,c0,maxidx,uint3(0,0,0),cp0,cp1,uint2(0,0),cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->srctype,
		     cfg->srcparam1,cfg->srcparam2,cfg->voidtime,cfg->maxdetphoton,
		     cfg->medianum-1,cfg->detnum,0,0,cfg->reseedlimit,ABS(cfg->sradius+2.f)<EPS /*isatomic*/,
		     (uint)cfg->maxvoidstep,cfg->issaveseed>0,cfg->maxdetphoton*(cfg->medianum+1),cfg->seed,
		     (uint)cfg->outputtype,0,0,cfg->faststep};
     int detreclen=cfg->medianum+1;
     if(param.isatomic)
         param.skipradius2=0.f;

#ifdef _OPENMP
     threadid=omp_get_thread_num();
#endif
     if(threadid<MAX_DEVICE && cfg->deviceid[threadid]=='\0')
           return;

     gpuid=cfg->deviceid[threadid]-1;
     CUDA_ASSERT(cudaSetDevice(gpuid));

     if(gpu[gpuid].maxgate==0 && dimxyz>0){
         int needmem=dimxyz+cfg->nthread*sizeof(float4)*4+sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1)+10*1024*1024; /*keep 10M for other things*/
         gpu[gpuid].maxgate=(gpu[gpuid].globalmem-needmem)/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
         gpu[gpuid].maxgate=MIN(((cfg->tend-cfg->tstart)/cfg->tstep+0.5),gpu[gpuid].maxgate);     
     }
     /*only allow the master thread to modify cfg, others are read-only*/
#pragma omp master
{
     if(cfg->exportfield==NULL)
         cfg->exportfield=(float *)calloc(sizeof(float)*cfg->dim.x*cfg->dim.y*cfg->dim.z,gpu[gpuid].maxgate*2);
     if(cfg->exportdetected==NULL)
         cfg->exportdetected=(float*)malloc((cfg->medianum+1)*cfg->maxdetphoton*sizeof(float));
     if(cfg->issaveseed && cfg->seeddata==NULL)
         cfg->seeddata=malloc(cfg->maxdetphoton*sizeof(float)*RAND_BUF_LEN);
     cfg->detectedcount=0;
     cfg->his.detected=0;
     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     cfg->runtime=0;
}
#pragma omp barrier

     if(!cfg->autopilot){
	gpu[gpuid].autothread=cfg->nthread;
	gpu[gpuid].autoblock=cfg->nblocksize;
	gpu[gpuid].maxgate=cfg->maxgate;
     }
     if(gpu[gpuid].autothread%gpu[gpuid].autoblock)
     	gpu[gpuid].autothread=(gpu[gpuid].autothread/gpu[gpuid].autoblock)*gpu[gpuid].autoblock;

     if(cfg->respin>1){
         field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate*2);
     }else{
         field=(float *)calloc(sizeof(float)*dimxyz,gpu[gpuid].maxgate); //the second half will be used to accumulate
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

     param.threadphoton=gpuphoton/gpu[gpuid].autothread/cfg->respin;
     param.oddphotons=gpuphoton/cfg->respin-param.threadphoton*gpu[gpuid].autothread;
     totalgates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
#pragma omp master
     if(totalgates>gpu[gpuid].maxgate && cfg->isnormalized){
         MCX_FPRINTF(stderr,"WARNING: GPU memory can not hold all time gates, disabling normalization to allow multiple runs\n");
         cfg->isnormalized=0;
     }
#pragma omp barrier

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
           Pseed=(uint*)malloc(sizeof(uint)*RAND_SEED_LEN);
           for (i=0; i<RAND_SEED_LEN; i++)
		Pseed[i]=rand();
           CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(uint)*RAND_SEED_LEN));
	   CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(uint)*RAND_SEED_LEN,  cudaMemcpyHostToDevice));
           CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen));
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(float)*fieldlen)); // cost about 1 ms
           CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

           tic=StartTimer();
           MCX_FPRINTF(cfg->flog,"generating %d random numbers ... \t",fieldlen); fflush(cfg->flog);
           mcx_test_rng<<<1,1>>>(gfield,gPseed);
           tic1=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving random numbers ... \t",tic1-tic);
           CUDA_ASSERT(cudaGetLastError());

           CUDA_ASSERT(cudaMemcpy(field, gfield,sizeof(float)*dimxyz*gpu[gpuid].maxgate,cudaMemcpyDeviceToHost));
           MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	   if(cfg->exportfield)
	       memcpy(cfg->exportfield,field,fieldlen*sizeof(float));
	   if(cfg->issave2pt && cfg->parentid==mpStandalone){
               MCX_FPRINTF(cfg->flog,"saving data to file ...\t");
	       mcx_savedata(field,fieldlen,timegate>0,"mc2",cfg);
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
     Pseed=(uint*)malloc(sizeof(uint)*gpu[gpuid].autothread*RAND_SEED_LEN);

     CUDA_ASSERT(cudaMalloc((void **) &gmedia, sizeof(uchar)*(dimxyz)));
     //CUDA_ASSERT(cudaBindTexture(0, texmedia, gmedia));
     CUDA_ASSERT(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen));
     CUDA_ASSERT(cudaMalloc((void **) &gPpos, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdir, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPlen, sizeof(float4)*gpu[gpuid].autothread));
     CUDA_ASSERT(cudaMalloc((void **) &gPdet, sizeof(float)*cfg->maxdetphoton*(detreclen)));
     CUDA_ASSERT(cudaMalloc((void **) &gdetected, sizeof(uint)));
     CUDA_ASSERT(cudaMalloc((void **) &genergy, sizeof(float)*(gpu[gpuid].autothread<<1)));
     if(cfg->issaveseed){
         seeddata=(RandType*)malloc(sizeof(RandType)*cfg->maxdetphoton*RAND_SEED_LEN);
	 CUDA_ASSERT(cudaMalloc((void **) &gseeddata, sizeof(RandType)*cfg->maxdetphoton*RAND_SEED_LEN));
     }
     if(cfg->seed==SEED_FROM_FILE){
         CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(float)*cfg->nphoton*RAND_SEED_LEN));
	 CUDA_ASSERT(cudaMemcpy(gPseed,cfg->replay.seed,sizeof(float)*cfg->nphoton*RAND_SEED_LEN, cudaMemcpyHostToDevice));
	 if(cfg->replay.weight){
	     CUDA_ASSERT(cudaMalloc((void **) &greplayw, sizeof(float)*cfg->nphoton));
	     CUDA_ASSERT(cudaMemcpy(greplayw,cfg->replay.weight,sizeof(float)*cfg->nphoton, cudaMemcpyHostToDevice));
	 }
         if(cfg->replay.tof){
	     CUDA_ASSERT(cudaMalloc((void **) &greplaytof, sizeof(float)*cfg->nphoton));
	     CUDA_ASSERT(cudaMemcpy(greplaytof,cfg->replay.tof,sizeof(float)*cfg->nphoton, cudaMemcpyHostToDevice));
	 }
     }else
         CUDA_ASSERT(cudaMalloc((void **) &gPseed, sizeof(uint)*gpu[gpuid].autothread*RAND_SEED_LEN));

     if(cfg->srctype==MCX_SRC_PATTERN)
         CUDA_ASSERT(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w)));

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

     param.dimlen=dimlen;
     param.cachebox=cachebox;
     if(p0.x<0.f || p0.y<0.f || p0.z<0.f || p0.x>=cfg->dim.x || p0.y>=cfg->dim.y || p0.z>=cfg->dim.z){
         param.idx1dorig=0;
         param.mediaidorig=0;
     }else{
         param.idx1dorig=(int(floorf(p0.z))*dimlen.y+int(floorf(p0.y))*dimlen.x+int(floorf(p0.x)));
         param.mediaidorig=(cfg->vol[param.idx1dorig] & MED_MASK);
     }

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
     MCX_FPRINTF(cfg->flog,"- compiled with: RNG [%s] with Seed Length [%d]\n",MCX_RNG_NAME,RAND_SEED_LEN);
#ifdef SAVE_DETECTORS
     MCX_FPRINTF(cfg->flog,"- this version CAN save photons at the detectors\n\n");
#else
     MCX_FPRINTF(cfg->flog,"- this version CAN NOT save photons at the detectors\n\n");
#endif
     fflush(cfg->flog);
}
#pragma omp barrier

     MCX_FPRINTF(cfg->flog,"\nGPU=%d (%s) threadph=%d extra=%d np=%d nthread=%d maxgate=%d repetition=%d\n",gpuid+1,gpu[gpuid].name,param.threadphoton,param.oddphotons,
           gpuphoton,gpu[gpuid].autothread,gpu[gpuid].maxgate,cfg->respin);
     MCX_FPRINTF(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);

     CUDA_ASSERT(cudaMemcpy(gmedia, media, sizeof(uchar) *dimxyz, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpy(genergy,energy,sizeof(float) *(gpu[gpuid].autothread<<1), cudaMemcpyHostToDevice));
     if(cfg->srcpattern)
         CUDA_ASSERT(cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w), cudaMemcpyHostToDevice));

     CUDA_ASSERT(cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice));
     CUDA_ASSERT(cudaMemcpyToSymbol(gdetpos, cfg->detpos,  cfg->detnum*sizeof(float4), 0, cudaMemcpyHostToDevice));

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
     sharedbuf=gpu[gpuid].autoblock*(sizeof(RandType)*RAND_SEED_LEN+sizeof(MCXdir));
#ifdef  USE_CACHEBOX
     if(cfg->sradius>EPS || ABS(cfg->sradius+1.f)<EPS)
        sharedbuf+=sizeof(float)*((cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1));
#endif
     if(cfg->issavedet)
        sharedbuf+=gpu[gpuid].autoblock*sizeof(float)*(cfg->medianum-1);
#ifdef USE_MT_RAND
     sharedbuf+=(N+2)*sizeof(uint); // MT RNG uses N+2 uint in the shared memory
#endif

     MCX_FPRINTF(cfg->flog,"requesting %d bytes of shared memory\n",sharedbuf);

     //simulate for all time-gates in maxgate groups per run
     for(timegate=0;timegate<totalgates;timegate+=gpu[gpuid].maxgate){

       param.twin0=cfg->tstart+cfg->tstep*timegate;
       param.twin1=param.twin0+cfg->tstep*gpu[gpuid].maxgate;
       CUDA_ASSERT(cudaMemcpyToSymbol(gcfg,   &param,     sizeof(MCXParam), 0, cudaMemcpyHostToDevice));

       MCX_FPRINTF(cfg->flog,"lauching MCX simulation for time window [%.2ens %.2ens] ...\n"
           ,param.twin0*1e9,param.twin1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<(int)cfg->respin;iter++){
           CUDA_ASSERT(cudaMemset(gfield,0,sizeof(float)*fieldlen)); // cost about 1 ms
           CUDA_ASSERT(cudaMemset(gPdet,0,sizeof(float)*cfg->maxdetphoton*(detreclen)));
           if(cfg->issaveseed)
	       CUDA_ASSERT(cudaMemset(gseeddata,0,sizeof(RandType)*cfg->maxdetphoton*RAND_BUF_LEN));
           CUDA_ASSERT(cudaMemset(gdetected,0,sizeof(float)));

 	   CUDA_ASSERT(cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));
	   CUDA_ASSERT(cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));
	   CUDA_ASSERT(cudaMemcpy(gPlen,  Plen,  sizeof(float4)*gpu[gpuid].autothread,  cudaMemcpyHostToDevice));
           if(cfg->seed!=SEED_FROM_FILE){
             for (i=0; i<gpu[gpuid].autothread*RAND_SEED_LEN; i++)
               Pseed[i]=rand();
	     CUDA_ASSERT(cudaMemcpy(gPseed, Pseed, sizeof(uint)*gpu[gpuid].autothread*RAND_SEED_LEN,  cudaMemcpyHostToDevice));
           }
           tic0=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);
           mcx_main_loop<<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern,greplayw,greplaytof,gseeddata);

           CUDA_ASSERT(cudaThreadSynchronize());
	   CUDA_ASSERT(cudaMemcpy(&detected, gdetected,sizeof(uint),cudaMemcpyDeviceToHost));
           tic1=GetTimeMillis();
	   toc+=tic1-tic0;
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",tic1-tic);
           CUDA_ASSERT(cudaGetLastError());

           CUDA_ASSERT(cudaMemcpy(Plen0,  gPlen,  sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
           for(i=0;i<gpu[gpuid].autothread;i++)
	      photoncount+=int(Plen0[i].w+0.5f);

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
			MCX_FPRINTF(cfg->flog,"detected %d photons, total: %d\t",detected,cfg->detectedcount+detected);
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

	   //handling the 2pt distributions
           if(cfg->issave2pt){
               CUDA_ASSERT(cudaMemcpy(field, gfield,sizeof(float) *dimxyz*gpu[gpuid].maxgate,cudaMemcpyDeviceToHost));
               MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

               if(cfg->respin>1){
                   for(i=0;i<(int)fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
           }
       } /*end of respin loop*/

#pragma omp critical
       if(cfg->runtime<toc)
           cfg->runtime=toc;

       if(cfg->respin>1)  //copy the accumulated fields back
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
           MCX_FPRINTF(cfg->flog,"normalizing raw data ...\t");
           cfg->energyabs+=cfg->energytot-cfg->energyesc;
           if(cfg->outputtype==otFlux || cfg->outputtype==otFluence){
               scale=1.f/(cfg->energytot*Vvox*cfg->tstep);
	       if(cfg->unitinmm!=1.f)
		   scale*=cfg->unitinmm; /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

               if(cfg->outputtype==otFluence)
		   scale*=cfg->tstep;
	   }else if(cfg->outputtype==otEnergy || cfg->outputtype==otJacobian)
	       scale=1.f/cfg->energytot;

         cfg->normalizer=scale;
	 MCX_FPRINTF(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
         mcx_normalize(cfg->exportfield,scale,fieldlen);
     }
     if(cfg->issave2pt && cfg->parentid==mpStandalone){
         MCX_FPRINTF(cfg->flog,"saving data to file ... %d %d\t",fieldlen,gpu[gpuid].maxgate);
         mcx_savedata(cfg->exportfield,fieldlen,0,"mc2",cfg);
         MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
         fflush(cfg->flog);
     }
     if(cfg->issavedet && cfg->parentid==mpStandalone && cfg->exportdetected){
         cfg->his.unitinmm=cfg->unitinmm;
         cfg->his.savedphoton=cfg->detectedcount;
         if(cfg->issaveseed)
             cfg->his.seedbyte=sizeof(RandType)*RAND_BUF_LEN;

         cfg->his.detected=cfg->detectedcount;
         mcx_savedetphoton(cfg->exportdetected,cfg->seeddata,cfg->detectedcount,0,cfg);
     }
}
#pragma omp barrier

     CUDA_ASSERT(cudaMemcpy(Ppos,  gPpos, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(Pdir,  gPdir, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(Plen,  gPlen, sizeof(float4)*gpu[gpuid].autothread, cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(Pseed, gPseed,sizeof(uint)  *gpu[gpuid].autothread*RAND_SEED_LEN,   cudaMemcpyDeviceToHost));
     CUDA_ASSERT(cudaMemcpy(energy,genergy,sizeof(float)*(gpu[gpuid].autothread<<1),cudaMemcpyDeviceToHost));

#ifdef TEST_RACING
     {
       float totalcount=0.f,hitcount=0.f;
       for (i=0; i<fieldlen; i++)
          hitcount+=field[i];
       for (i=0; i<gpu[gpuid].autothread; i++)
	  totalcount+=Pseed[i];

       MCX_FPRINTF(cfg->flog,"expected total recording number: %f, got %f, missed %f\n",
          totalcount,hitcount,(totalcount-hitcount)/totalcount);
     }
#endif

#pragma omp master
{
     printnum=(gpu[gpuid].autothread<(int)cfg->printnum) ? gpu[gpuid].autothread : cfg->printnum;
     for (i=0; i<(int)printnum; i++) {
            MCX_FPRINTF(cfg->flog,"% 4d[A% f % f % f]C%3d J%5d W% 8f(P%.13f %.13f %.13f)T% 5.3e L% 5.3f %.0f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i]);
     }
     // total energy here equals total simulated photons+unfinished photons for all threads
     MCX_FPRINTF(cfg->flog,"simulated %d photons (%d) with %d threads (repeat x%d)\nMCX simulation speed: %.2f photon/ms\n",
             cfg->nphoton,cfg->nphoton,gpu[gpuid].autothread,cfg->respin,(double)cfg->nphoton/cfg->runtime); fflush(cfg->flog);
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
         CUDA_ASSERT(cudaFree(greplayw));
         CUDA_ASSERT(cudaFree(greplaytof));
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
