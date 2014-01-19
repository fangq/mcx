////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation                                                                                                
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
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

#ifdef USE_MT_RAND
#include "mt_rand_s.cu"     // use Mersenne Twister RNG (MT)
#else
#include "logistic_rand.cu" // use Logistic Lattice ring 5 RNG (LL5)
#endif

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

__device__ inline void savedetphoton(float n_det[],uint *detectedphoton,float nscat,float *ppath,MCXpos *p0,RandType t[RAND_BUF_LEN]){
      uint detid;
      detid=finddetector(p0);
      if(detid){
	 uint baseaddr=atomicAdd(detectedphoton,1);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    for(i=0;i<gcfg->issaveseed*RAND_BUF_LEN;i++)
	        n_det[gcfg->seedoffset+baseaddr*RAND_BUF_LEN+i]=t[i]; // save photon seed for replay

	    baseaddr*=gcfg->maxmedia+2;
	    n_det[baseaddr++]=detid;
	    n_det[baseaddr++]=nscat;
	    for(i=0;i<gcfg->maxmedia;i++)
		n_det[baseaddr+i]=ppath[i]; // save partial pathlength to the memory
	 }
      }
}
#endif

/**
  the returned position, htime, always stays inside the non-air voxel
*/

__device__ inline void getentrypoint(MCXpos *p0, MCXpos *p, MCXdir *v, float3 *htime, float *flipdir,int idx1d,int idx1dold,uchar media[],uchar mediaidold,float *backmove){
      float tmp0,tmp1;
      //time-of-flight to hit the wall in each direction
      htime->x=(v->x>EPS||v->x<-EPS)?(floorf(p0->x)+(v->x>0.f)-p0->x)/v->x:VERY_BIG;
      htime->y=(v->y>EPS||v->y<-EPS)?(floorf(p0->y)+(v->y>0.f)-p0->y)/v->y:VERY_BIG;
      htime->z=(v->z>EPS||v->z<-EPS)?(floorf(p0->z)+(v->z>0.f)-p0->z)/v->z:VERY_BIG;
      //get the direction with the smallest time-of-flight
      tmp0=fminf(fminf(htime->x,htime->y),htime->z);
      (*flipdir)=(tmp0==htime->x?1.f:(tmp0==htime->y?2.f:(tmp0==htime->z&&idx1d!=idx1dold)?3.f:0.f));

      //p0 is inside, p is outside, move to the 1st intersection pt, now in the air side, to be corrected in the else block
      tmp0*=JUST_ABOVE_ONE;
      htime->x=p0->x+tmp0*v->x;
      htime->y=p0->y+tmp0*v->y;
      htime->z=p0->z+tmp0*v->z;

      GPUDEBUG((" trial 1: [%.1f %.1f,%.1f] %d %f\n",htime->x,htime->y,htime->z,(*flipdir),
            media[int(floorf(htime->z)*gcfg->dimlen.y+floorf(htime->y)*gcfg->dimlen.x+floorf(htime->x))]));

      if(htime->x>=0&&htime->y>=0&&htime->z>=0&&htime->x<gcfg->maxidx.x&&htime->y<gcfg->maxidx.y&&htime->z<gcfg->maxidx.z
	   &&media[int(floorf(htime->z)*gcfg->dimlen.y+floorf(htime->y)*gcfg->dimlen.x+floorf(htime->x))]==mediaidold){ //if the first vox is not air

           htime->x=(v->x>EPS||v->x<-EPS)?(floorf(p->x)+(v->x<0.f)-p->x)/(-v->x):VERY_BIG;
           htime->y=(v->y>EPS||v->y<-EPS)?(floorf(p->y)+(v->y<0.f)-p->y)/(-v->y):VERY_BIG;
           htime->z=(v->z>EPS||v->z<-EPS)?(floorf(p->z)+(v->z<0.f)-p->z)/(-v->z):VERY_BIG;
           tmp0=fminf(fminf(htime->x,htime->y),htime->z);
           tmp1=(*flipdir);   //save the previous ref. interface id
           (*flipdir)=(tmp0==htime->x?1.f:(tmp0==htime->y?2.f:(tmp0==htime->z&&idx1d!=idx1dold)?3.f:0.f));

           //if(gcfg->doreflect3){
             tmp0*=JUST_ABOVE_ONE;
             htime->x=p->x-tmp0*v->x; //move to the last intersection pt, stays in the non-air voxel
             htime->y=p->y-tmp0*v->y;
             htime->z=p->z-tmp0*v->z;
	     *backmove=tmp0;

             GPUDEBUG((" trial 2: [%.1f %.1f,%.1f] %d %f\n",htime->x,htime->y,htime->z,(*flipdir),
                  media[int(floorf(htime->z)*gcfg->dimlen.y+floorf(htime->y)*gcfg->dimlen.x+floorf(htime->x))]));

             if(tmp1!=(*flipdir)&&htime->x>=0&&htime->y>=0&&htime->z>=0&&
		  floorf(htime->x)<gcfg->maxidx.x&&floorf(htime->y)<gcfg->maxidx.y&&floorf(htime->z)<gcfg->maxidx.z){
                 if(media[int(floorf(htime->z)*gcfg->dimlen.y+floorf(htime->y)*gcfg->dimlen.x+floorf(htime->x))]!=mediaidold){ //this is an air voxel

                     GPUDEBUG((" trial 3: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime->x,htime->y,htime->z,
                         media[int(floorf(htime->z)*gcfg->dimlen.y+floorf(htime->y)*gcfg->dimlen.x+floorf(htime->x))], 
			 p->x,p->y,p->z));

                     /*to compute the remaining interface, we used the following fact to accelerate: 
                       if there exist 3 intersections, photon must pass x/y/z interface exactly once,
                       we solve the coeff of the following equation to find the last interface:
                          a*1+b*2+c=3
       	       	       	  a*1+b*3+c=2 -> [a b c]=[-1 -1 6], this will give the remaining interface id
       	       	       	  a*2+b*3+c=1
                     */
                     (*flipdir)=-tmp1-(*flipdir)+6.f;

                     htime->x=(v->x>EPS||v->x<-EPS)?(floorf(htime->x)+(v->x<0.f)-htime->x)/(-v->x):VERY_BIG;
                     htime->y=(v->y>EPS||v->y<-EPS)?(floorf(htime->y)+(v->y<0.f)-htime->y)/(-v->y):VERY_BIG;
                     htime->z=(v->z>EPS||v->z<-EPS)?(floorf(htime->z)+(v->z<0.f)-htime->z)/(-v->z):VERY_BIG;
                     tmp1=fminf(fminf(htime->x,htime->y),htime->z);
		     tmp1*=JUST_ABOVE_ONE;
                     htime->x=p->x-(tmp0+tmp1)*v->x; /*htime is now the exact exit position*/
                     htime->y=p->y-(tmp0+tmp1)*v->y;
                     htime->z=p->z-(tmp0+tmp1)*v->z;
		     *backmove=tmp0+tmp1;
                 }
             }
           //}
      }else{
	  tmp0*=JUST_BELOW_ONE; // JUST_BELOW_ONE*JUST_ABOVE_ONE<1
	  htime->x=p0->x+tmp0*v->x;
	  htime->y=p0->y+tmp0*v->y;
	  htime->z=p0->z+tmp0*v->z;
	  *backmove=1.f-tmp0;
      }
}

__device__ inline void transmit(MCXdir *v, float n1, float n2,float flipdir){
      float tmp0=n1/n2;
      if(flipdir>=3.f) { //transmit through z plane
         v->x=tmp0*v->x;
         v->y=tmp0*v->y;
      }else if(flipdir>=2.f){ //transmit through y plane
         v->x=tmp0*v->x;
         v->z=tmp0*v->z;
      }else if(flipdir>=1.f){ //transmit through x plane
         v->y=tmp0*v->y;
         v->z=tmp0*v->z;
      }
      tmp0=rsqrtf(v->x*v->x+v->y*v->y+v->z*v->z);
      v->x=v->x*tmp0;
      v->y=v->y*tmp0;
      v->z=v->z*tmp0;
}

__device__ inline float reflectcoeff(MCXdir *v, float n1, float n2, float flipdir){
      float Icos=fabs(((float*)v)[__float2int_rn(flipdir)-1]);
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
__device__ inline int skipvoid(MCXpos *p,MCXdir *v,MCXtime *f,uchar media[]){
      int count=1,idx1d,idx1dold;
      while(1){
          if(p->x>=0.f && p->y>=0.f && p->z>=0.f && p->x < gcfg->maxidx.x
               && p->y < gcfg->maxidx.y && p->z < gcfg->maxidx.z){
	    idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	    if(media[idx1d]){ // if inside
	      float3 htime;
	      float flipdir,backmove=0.f;
	      MCXdir nv={-v->x,-v->y,-v->z,v->nscat};
	      MCXpos p0={p->x-v->x,p->y-v->y,p->z-v->z,p->w};
	      getentrypoint(p,&p0,&nv,&htime,&flipdir,idx1d,idx1dold,media,media[idx1d],&backmove);
	      if(gcfg->voidtime) f->t-=gcfg->minaccumtime*backmove;
	      *((float4*)(p))=float4(htime.x,htime.y,htime.z,p->w);
	      idx1d=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	      if(gproperty[media[idx1d]].w!=gproperty[0].w){
	          p->w*=1.f-reflectcoeff(v, gproperty[0].w,gproperty[media[idx1d]].w,flipdir);
	          transmit(v, gproperty[0].w,gproperty[media[idx1d]].w,flipdir);
	      }
	      return idx1d;
	    }
          }
	  if( (p->x<0.f) && (v->x<=0.f) || (p->x >= gcfg->maxidx.x) && (v->x>=0.f)
	   || (p->y<0.f) && (v->y<=0.f) || (p->y >= gcfg->maxidx.y) && (v->y>=0.f)
	   || (p->z<0.f) && (v->z<=0.f) || (p->z >= gcfg->maxidx.z) && (v->z>=0.f))
	      return -1;
	  idx1dold=(int(floorf(p->z))*gcfg->dimlen.y+int(floorf(p->y))*gcfg->dimlen.x+int(floorf(p->x)));
	  *((float4*)(p))=float4(p->x+v->x,p->y+v->y,p->z+v->z,p->w);
          if(gcfg->voidtime) f->t+=gcfg->minaccumtime;
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
   	  GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
      }else{
   	  *((float4*)v)=float4(stheta*cphi,stheta*sphi,(v->z>0.f)?ctheta:-ctheta,v->nscat);
   	  GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",v->x,v->y,v->z));
      }
}

__device__ inline int launchnewphoton(MCXpos *p,MCXdir *v,MCXtime *f,Medium *prop,uint *idx1d,
           uchar *mediaid,uchar isdet, float ppath[],float energyloss[],float energylaunched[],float n_det[],uint *dpnum,
	   RandType t[RAND_BUF_LEN],RandType tnew[RAND_BUF_LEN],RandType photonseed[RAND_BUF_LEN],
	   uchar media[],float srcpattern[],int threadid){
      int launchattempt=1;
      *energyloss+=p->w;  // sum all the remaining energy
#ifdef SAVE_DETECTORS
      // let's handle detectors here
      if(gcfg->savedet){
         if(*mediaid==0 && isdet)
	      savedetphoton(n_det,dpnum,v->nscat,ppath,p,photonseed);
	 clearpath(ppath,gcfg->maxmedia);
      }
#endif
      if(f->ndone>=(gcfg->threadphoton+(threadid<gcfg->oddphotons)))
          return 1; // all photos complete
      do{
	  *((float4*)p)=gcfg->ps;
	  *((float4*)v)=gcfg->c0;
	  *((float4*)f)=float4(0.f,0.f,gcfg->minaccumtime,f->ndone);
          *idx1d=gcfg->idx1dorig;
          *mediaid=gcfg->mediaidorig;      
	  //if(gcfg->srctype==MCX_SRC_PENCIL){ /*source can be outside*/
          for(int i=0;i<gcfg->issaveseed*RAND_BUF_LEN;i++)
	      photonseed[i]=t[i];
	  if(gcfg->srctype==MCX_SRC_PLANAR || gcfg->srctype==MCX_SRC_PATTERN|| gcfg->srctype==MCX_SRC_FOURIER){ /*a rectangular grid over a plane*/
	      rand_need_more(t,tnew);
	      RandType rx=rand_uniform01(t[0]);
	      rand_need_more(t,tnew);
	      RandType ry=rand_uniform01(t[0]);
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
	      rand_need_more(t,tnew);
	  }else if(gcfg->srctype==MCX_SRC_FOURIERX||gcfg->srctype==MCX_SRC_FOURIERX2D){ // [v1x][v1y][v1z][|v2|]; [kx][ky][phi0][M], unit(v0) x unit(v1)=unit(v2)
	      rand_need_more(t,tnew);
	      RandType rx=rand_uniform01(t[0]);
	      rand_need_more(t,tnew);
	      RandType ry=rand_uniform01(t[0]);
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
	      rand_need_more(t,tnew);
	  }else if(gcfg->srctype==MCX_SRC_DISK){ // uniform disk distribution
	      // Uniform disk point picking
	      // http://mathworld.wolfram.com/DiskPointPicking.html
	      float sphi, cphi;
	      rand_need_more(t,tnew);
	      RandType phi=TWO_PI*rand_uniform01(t[0]);
              sincosf(phi,&sphi,&cphi);
	      rand_need_more(t,tnew);
	      RandType r=sqrtf(rand_uniform01(t[0]))*gcfg->srcparam1.x;
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
	      rand_need_more(t,tnew);
              ang=TWO_PI*rand_uniform01(t[0]); //next arimuth angle
              sincosf(ang,&sphi,&cphi);
	      if(gcfg->srctype==MCX_SRC_CONE){  // a solid-angle section of a uniform sphere
        	  do{
		      rand_need_more(t,tnew);
		      ang=(gcfg->srcparam1.y>0) ? TWO_PI*rand_uniform01(t[0]) : acosf(2.f*rand_uniform01(t[0])-1.f); //sine distribution
		  }while(ang>gcfg->srcparam1.x);
	      }else{
		  rand_need_more(t,tnew);
	          if(gcfg->srctype==MCX_SRC_ISOTROPIC) // uniform sphere
		      ang=acosf(2.f*rand_uniform01(t[0])-1.f); //sine distribution
		  else
		      ang=ONE_PI*rand_uniform01(t[0]); //uniform distribution in zenith angle, arcsine
	      }
              sincosf(ang,&stheta,&ctheta);
              rotatevector(v,stheta,ctheta,sphi,cphi);
	  }else if(gcfg->srctype==MCX_SRC_GAUSSIAN){
              float ang,stheta,ctheta,sphi,cphi;
              rand_need_more(t,tnew);
	      ang=TWO_PI*rand_uniform01(t[0]); //next arimuth angle
	      sincosf(ang,&sphi,&cphi);
              rand_need_more(t,tnew);
              ang=sqrtf(-2.f*logf(rand_uniform01(t[0])))*(1.f-2.f*t[1])*gcfg->srcparam1.x;
	      sincosf(ang,&stheta,&ctheta);
	      rotatevector(v,stheta,ctheta,sphi,cphi);
	  }
	  if(*mediaid==0){
             int idx=skipvoid(p, v, f, media);
             if(idx>=0){
		 *idx1d=idx;
		 *mediaid=media[*idx1d];
	     }
	  }
	  if(launchattempt++>gcfg->maxvoidstep)
	     return -1;  // launch failed
      }while(*mediaid==0 || p->w<=gcfg->minenergy);
      f->ndone++; // launch successfully
      *((float4*)(prop))=gproperty[*mediaid]; //always use mediaid to read gproperty[]
      *energylaunched+=p->w;
      return 0;
}

kernel void mcx_test_rng(float field[],uint n_seed[]){
     int idx= blockDim.x * blockIdx.x + threadIdx.x;
     int i,j;
     int len=gcfg->maxidx.x*gcfg->maxidx.y*gcfg->maxidx.z*(int)((gcfg->twin1-gcfg->twin0)*gcfg->Rtstep+0.5f);
     RandType t[RAND_BUF_LEN],tnew[RAND_BUF_LEN];

     gpu_rng_init(t,tnew,n_seed,idx);

     for(i=0;i<len;i+=RAND_BUF_LEN){
       rand_need_more(t,tnew);
       for(j=0;j<min(RAND_BUF_LEN,len-i);j++)
	   field[i+j]=t[j];
     }
}

/**
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
   everything in the GPU kernels is in grid-unit. To convert back to length, use
   cfg->unitinmm (scattering/absorption coeff, T, speed etc)
*/
kernel void mcx_main_loop(uchar media[],float field[],float genergy[],uint n_seed[],
     float4 n_pos[],float4 n_dir[],float4 n_len[],float n_det[], uint detectedphoton[], 
     float srcpattern[]){

     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     MCXpos  p,p0;//{x,y,z}: coordinates in grid unit, w:packet weight
     MCXdir  v;   //{x,y,z}: unitary direction vector in grid unit, nscat:total scat event
     MCXtime f;   //pscat: remaining scattering probability,t: photon elapse time, 
                  //tnext: next accumulation time, ndone: completed photons
     float  energyloss=genergy[idx*3];
     float  energyabsorbed=genergy[idx*3+1];
     float  energylaunched=genergy[idx*3+2];

     uint idx1d, idx1dold;   //idx1dold is related to reflection
     uint moves=0;

#ifdef TEST_RACING
     int cc=0;
#endif
     uchar  mediaid,mediaidold;
     char   medid=-1;
     float  atten;         //can be taken out to minimize registers
     float  n1;   //reflection var

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN],tnew[RAND_BUF_LEN],photonseed[RAND_BUF_LEN];
     Medium prop;    //can become float2 if no reflection (mua/musp is in 1/grid unit)

     float len;

     float *ppath=sharedmem;
#ifdef  USE_CACHEBOX
  #ifdef  SAVE_DETECTORS
     float *cachebox=sharedmem+(gcfg->savedet ? blockDim.x*gcfg->maxmedia: 0);
  #else
     float *cachebox=sharedmem;
  #endif
     if(gcfg->skipradius2>EPS) clearcache(cachebox,(gcfg->cp1.x-gcfg->cp0.x+1)*(gcfg->cp1.y-gcfg->cp0.y+1)*(gcfg->cp1.z-gcfg->cp0.z+1));
#else
     float accumweight=0.f;
#endif

#ifdef  SAVE_DETECTORS
     ppath=sharedmem+threadIdx.x*gcfg->maxmedia;
     if(gcfg->savedet) clearpath(ppath,gcfg->maxmedia);
#endif
     *((float4*)(&p))=n_pos[idx];
     *((float4*)(&v))=n_dir[idx];
     *((float4*)(&f))=n_len[idx];

     gpu_rng_init(t,tnew,n_seed,idx);

     if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,0,ppath,&energyloss,&energylaunched,n_det,detectedphoton,t,tnew,photonseed,media,srcpattern,idx)){
         n_seed[idx]=NO_LAUNCH;
	 n_pos[idx]=*((float4*)(&p));
	 n_dir[idx]=*((float4*)(&v));
	 n_len[idx]=*((float4*)(&f));
         return;
     }

     /*
      using a while-loop to terminate a thread by np will cause MT RNG to be 3.5x slower
      LL5 RNG will only be slightly slower than for-loop with photon-move criterion

      we have switched to while-loop since v0.4.9, as LL5 was only minimally effected
      and we do not use MT as the default RNG.
     */

     while(f.ndone<(gcfg->threadphoton+(idx<gcfg->oddphotons))) {

          GPUDEBUG(("*i= (%d) L=%f w=%e a=%f\n",(int)f.ndone,f.pscat,p.w,f.t));

          // dealing with scattering

	  if(f.pscat<=0.f) {  // if this photon has finished his current jump, get next scat length & angles
               if(moves++>gcfg->reseedlimit){
                  moves=0;
                  gpu_rng_reseed(t,tnew,n_seed,idx,(p.x+p.y+p.z+p.w)+f.ndone*(v.x+v.y+v.z));
               }
   	       f.pscat=rand_next_scatlen(t,tnew); // random scattering probability, unit-less

               GPUDEBUG(("next scat len=%20.16e \n",f.pscat));
	       if(p.w<1.f){ // if this is not my first jump
                       //random arimuthal angle
	               float cphi,sphi,theta,stheta,ctheta;
                       float tmp0=TWO_PI*rand_next_aangle(t,tnew); //next arimuth angle
                       sincosf(tmp0,&sphi,&cphi);
                       GPUDEBUG(("next angle phi %20.16e\n",tmp0));

                       //Henyey-Greenstein Phase Function, "Handbook of Optical 
                       //Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002

                       if(prop.g>EPS){  //if prop.g is too small, the distribution of theta is bad
		           tmp0=(1.f-prop.g*prop.g)/(1.f-prop.g+2.f*prop.g*rand_next_zangle(t,tnew));
		           tmp0*=tmp0;
		           tmp0=(1.f+prop.g*prop.g-tmp0)/(2.f*prop.g);

                           // when ran=1, CUDA gives me 1.000002 for tmp0 which produces nan later
                           // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                           tmp0=fmax(-1.f, fmin(1.f, tmp0));

		           theta=acosf(tmp0);
		           stheta=sinf(theta);
		           ctheta=tmp0;
                       }else{
			   theta=acosf(2.f*rand_next_zangle(t,tnew)-1.f);
                           sincosf(theta,&stheta,&ctheta);
                       }
                       GPUDEBUG(("next scat angle theta %20.16e\n",theta));
                       rotatevector(&v,stheta,ctheta,sphi,cphi);
                       v.nscat++;
	       }
	  }

          n1=prop.n;
	  *((float4*)(&prop))=gproperty[mediaid];
	  len=gcfg->minstep*prop.mus; //unitless (minstep=grid, mus=1/grid)

          // dealing with absorption

          p0=p;
	  if(len>f.pscat){  //scattering ends in this voxel: mus*gcfg->minstep > s 
               float tmp0=f.pscat/prop.mus; // unit=grid
   	       *((float4*)(&p))=float4(p.x+v.x*tmp0,p.y+v.y*tmp0,p.z+v.z*tmp0,
                           p.w*expf(-prop.mua*tmp0)); //mua=1/grid, tmp0=grid
	       f.pscat=SAME_VOXEL;
	       f.t+=tmp0*prop.n*gcfg->oneoverc0;  //propagation time (unit=s)
#ifdef SAVE_DETECTORS
               if(gcfg->savedet) ppath[mediaid-1]+=tmp0; //(unit=grid)
#endif
               GPUDEBUG((">>ends in voxel %f<%f %f [%d]\n",f.pscat,len,prop.mus,idx1d));
	  }else{                      //otherwise, move gcfg->minstep
               if(mediaid!=medid)
                  atten=expf(-prop.mua*gcfg->minstep);

   	       *((float4*)(&p))=float4(p.x+v.x,p.y+v.y,p.z+v.z,p.w*atten);
               medid=mediaid;
	       f.pscat-=len;     //remaining probability: sum(s_i*mus_i), unit-less
	       f.t+=gcfg->minaccumtime*prop.n; //propagation time  (unit=s)
#ifdef SAVE_DETECTORS
               if(gcfg->savedet) ppath[mediaid-1]+=gcfg->minstep; //(unit=grid)
#endif
               GPUDEBUG((">>keep going %f<%f %f [%d] %e %e\n",f.pscat,len,prop.mus,idx1d,f.t,f.tnext));
	  }

          mediaidold=media[idx1d];
          idx1dold=idx1d;
          idx1d=(int(floorf(p.z))*gcfg->dimlen.y+int(floorf(p.y))*gcfg->dimlen.x+int(floorf(p.x)));
          GPUDEBUG(("old and new voxels: %d<->%d\n",idx1dold,idx1d));
          if(p.x<0||p.y<0||p.z<0||p.x>=gcfg->maxidx.x||p.y>=gcfg->maxidx.y||p.z>=gcfg->maxidx.z){
	      mediaid=0;
	  }else{
	      mediaid=(media[idx1d] & MED_MASK);
          }

          // dealing with boundaries

          //if it hits the boundary, exceeds the max time window or exits the domain, rebound or launch a new one
	  if(mediaid==0||f.t>gcfg->tmax||f.t>gcfg->twin1||(gcfg->dorefint && n1!=gproperty[mediaid].w) ){
	      float flipdir=0.f,tmp0,tmp1;
              float3 htime;            //reflection var

              if(gcfg->doreflect || (gcfg->savedet && (mediaidold & DET_MASK)) ) 
                  getentrypoint(&p0,&p,&v,&htime,&flipdir,idx1d,idx1dold,media,mediaidold,&tmp0);

              *((float4*)(&prop))=gproperty[mediaid]; // optical property across the interface

              GPUDEBUG(("->ID%d J%d C%d tlen %e flip %d %.1f!=%.1f dir=%f %f %f pos=%f %f %f\n",idx,(int)v.nscat,
                  (int)f.ndone,f.t, (int)flipdir, n1,prop.n,v.x,v.y,v.z,p.x,p.y,p.z));

              //recycled some old register variables to save memory
	      //if hit boundary within the time window and is n-mismatched, rebound

              if(gcfg->doreflect&&f.t<gcfg->tmax&&f.t<gcfg->twin1&& flipdir>0.f && n1!=prop.n &&p.w>gcfg->minenergy){
	          float Rtotal=1.f;
	          float cphi,sphi,stheta,ctheta;

                  tmp0=n1*n1;
                  tmp1=prop.n*prop.n;
                  if(flipdir>=3.f) { //flip in z axis
                     cphi=fabs(v.z);
                     sphi=v.x*v.x+v.y*v.y;
                  }else if(flipdir>=2.f){ //flip in y axis
                     cphi=fabs(v.y);
       	       	     sphi=v.x*v.x+v.z*v.z;
                  }else if(flipdir>=1.f){ //flip in x axis
                     cphi=fabs(v.x);                //cos(si)
                     sphi=v.y*v.y+v.z*v.z; //sin(si)^2
                  }
                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2 = cos(ti)^2
	          GPUDEBUG((" ref len=%f %f+%f=%f w=%f\n",len,cphi,sphi,cphi*cphi+sphi,p.w));

                  if(len>0.f) { // if not total internal reflection
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.n*cphi*sqrtf(len);
                     Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
	             GPUDEBUG(("  dir=%f %f %f htime=%f %f %f Rs=%f\n",v.x,v.y,v.z,htime.x,htime.y,htime.z,Rtotal));
	             GPUDEBUG(("  ID%d J%d C%d flip=%3f (%d %d) cphi=%f sphi=%f p=%f %f %f p0=%f %f %f\n",
                         idx,(int)v.nscat,(int)f.tnext,
	                 flipdir,idx1dold,idx1d,cphi,sphi,p.x,p.y,p.z,p0.x,p0.y,p0.z));
                  } // else, total internal reflection
	          if(Rtotal<1.f && rand_next_reflect(t,tnew)>Rtotal){ // do transmission
                        if(mediaid==0){ // transmission to external boundary
                            p.x=htime.x;p.y=htime.y;p.z=htime.z;p.w=p0.w;
		    	    if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,(mediaidold & DET_MASK),
			        ppath,&energyloss,&energylaunched,n_det,detectedphoton,t,tnew,photonseed,media,srcpattern,idx))
                                break;
			    continue;
			}
			transmit(&v,n1,prop.n,flipdir);
		  }else{ //do reflection
                        if(flipdir>=1.f)
                            ((float*)(&v))[__float2int_rn(flipdir)-1]*=-1.f;
                        p=p0;   //move to the reflection point
                	idx1d=idx1dold;
		 	mediaid=(media[idx1d] & MED_MASK);
        	  	*((float4*)(&prop))=gproperty[mediaid];
                  	n1=prop.n;
		  }
              }else{  // launch a new photon
                  p.x=htime.x;p.y=htime.y;p.z=htime.z;p.w=p0.w; // this is only used when savedet is true
		  if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,(mediaidold & DET_MASK),ppath,
		      &energyloss,&energylaunched,n_det,detectedphoton,t,tnew,photonseed,media,srcpattern,idx))
                       break;
		  continue;
              }
	  }

          // saving fluence to the memory

	  if(f.t>=f.tnext){
             GPUDEBUG(("field add to %d->%f(%d)  t(%e)>t0(%e)\n",idx1d,p.w,(int)f.ndone,f.t,f.tnext));
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(gcfg->save2pt && f.t>=gcfg->twin0 && f.t<gcfg->twin1){
                  energyabsorbed+=p.w*prop.mua;
#ifdef TEST_RACING
                  // enable TEST_RACING to determine how many missing accumulations due to race
                  if( (p.x-gcfg->ps.x)*(p.x-gcfg->ps.x)+(p.y-gcfg->ps.y)*(p.y-gcfg->ps.y)+(p.z-gcfg->ps.z)*(p.z-gcfg->ps.z)>gcfg->skipradius2) {
                      field[idx1d+(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=1.f;
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
			      +int(p.y-gcfg->cp0.y)*gcfg->cachebox.x+int(p.x-gcfg->cp0.x)),p.w);
  #else
                      if((p.x-gcfg->ps.x)*(p.x-gcfg->ps.x)+(p.y-gcfg->ps.y)*(p.y-gcfg->ps.y)+(p.z-gcfg->ps.z)*(p.z-gcfg->ps.z)<=gcfg->skipradius2){
                          accumweight+=p.w*prop.mua; // weight*absorption
  #endif
                      }else{
                          field[idx1d+(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=p.w;
                      }
                  }else{
                      field[idx1d+(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=p.w;
                  }
  #ifdef USE_ATOMIC
               }else{
                  // ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
		  atomicadd(& field[idx1d+(int)(floorf((f.t-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z], p.w);
               }
  #endif
#endif
	     }
             f.tnext+=gcfg->minaccumtime*prop.n; // fluence is a temporal-integration, unit=s
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

     genergy[idx*3]=energyloss;
     genergy[idx*3+1]=energyabsorbed;
     genergy[idx*3+2]=energylaunched;

#ifdef TEST_RACING
     n_seed[idx]=cc;
#endif
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
  obtain GPU core number per MP, this replaces 
  ConvertSMVer2Cores() in libcudautils to avoid 
  extra dependency.
*/

int mcx_corecount(int v1, int v2){
     int v=v1*10+v2;
     if(v<20)      return 8;
     else if(v<21) return 32;
     else          return 48;
}

/**
  query GPU info and set active GPU
*/
int mcx_set_gpu(Config *cfg){

#if __DEVICE_EMULATION__
    return 1;
#else
    int dev;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        MCX_FPRINTF(stderr,"No CUDA-capable GPU device found\n");
        return 0;
    }
    if (cfg->gpuid && cfg->gpuid > deviceCount){
        MCX_FPRINTF(stderr,"Specified GPU ID is out of range\n");
        return 0;
    }
    // scan from the first device
    for (dev = 0; dev<deviceCount; dev++) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, dev);
	if(cfg->autopilot && ((cfg->gpuid && dev==cfg->gpuid-1)
	 ||(cfg->gpuid==0 && dev==deviceCount-1) )){
                unsigned int needmem=cfg->dim.x*cfg->dim.y*cfg->dim.z; /*for mediam*/
		if(cfg->autopilot==1){
#ifdef USE_MT_RAND
                        cfg->nblocksize=1;
#else
			cfg->nblocksize=64;
#endif
			cfg->nthread=dp.multiProcessorCount*mcx_corecount(dp.major,dp.minor)*32;
			needmem+=cfg->nthread*sizeof(float4)*4+sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1+(cfg->issaveseed>0)*RAND_BUF_LEN)+10*1024*1024; /*keep 10M for other things*/
			cfg->maxgate=((unsigned int)dp.totalGlobalMem-needmem)/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
			cfg->maxgate=MIN((int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5),cfg->maxgate);
			MCX_FPRINTF(cfg->flog,"autopilot mode: setting thread number to %d, block size to %d and time gates to %d\n",cfg->nthread,cfg->nblocksize,cfg->maxgate);
		}else if(cfg->autopilot==2){
#ifdef USE_MT_RAND
                        cfg->nblocksize=1;
#else
                        cfg->nblocksize=64;
#endif
			cfg->nthread=dp.multiProcessorCount*mcx_corecount(dp.major,dp.minor)*16;
                        MCX_FPRINTF(cfg->flog,"autopilot mode: setting thread number to %d and block size to %d\n",cfg->nthread,cfg->nblocksize);
		}
	}
        if (strncmp(dp.name, "Device Emulation", 16)) {
	  if(cfg->isgpuinfo){
	    MCX_FPRINTF(stdout,"=============================   GPU Infomation  ================================\n");
	    MCX_FPRINTF(stdout,"Device %d of %d:\t\t%s\n",dev+1,deviceCount,dp.name);
	    MCX_FPRINTF(stdout,"Compute Capability:\t%u.%u\n",dp.major,dp.minor);
	    MCX_FPRINTF(stdout,"Global Memory:\t\t%u B\nConstant Memory:\t%u B\n\
Shared Memory:\t\t%u B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
               (unsigned int)dp.totalGlobalMem,(unsigned int)dp.totalConstMem,
               (unsigned int)dp.sharedMemPerBlock,(unsigned int)dp.regsPerBlock,dp.clockRate*1e-6f);
	  #if CUDART_VERSION >= 2000
	       MCX_FPRINTF(stdout,"Number of MPs:\t\t%u\nNumber of Cores:\t%u\n",
	          dp.multiProcessorCount,dp.multiProcessorCount*mcx_corecount(dp.major,dp.minor));
	  #endif
	  }
	}
    }
    if(cfg->isgpuinfo==2 && cfg->exportfield==NULL){ //list GPU info only
          exit(0);
    }
    if (cfg->gpuid==0)
        mcx_cu_assess(cudaSetDevice(deviceCount-1),__FILE__,__LINE__);
    else
        mcx_cu_assess(cudaSetDevice(cfg->gpuid-1),__FILE__,__LINE__);

#ifdef USE_MT_RAND
    if(cfg->nblocksize>N-M){
        mcx_error(-1,"block size can not be larger than 227 when using MT19937 RNG",__FILE__,__LINE__);
    }
#endif

    return 1;
#endif
}


/**
   host code for MCX kernels
*/
void mcx_run_simulation(Config *cfg){

     int i,iter;
     float  minstep=1.f; //MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     float4 c0=float4(cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f);
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     float energyloss=0.f,energyabsorbed=0.f,energylaunched=0.f;
     float *energy;
     int timegate=0, totalgates;

     unsigned int photoncount=0,printnum,exportedcount=0;
     unsigned int tic,tic0,tic1,toc=0,fieldlen;
     uint3 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint3 dimlen;
     float Vvox,scale,eabsorp;

     dim3 mcgrid, mcblock;
     dim3 clgrid, clblock;

     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     
     uchar  *media=(uchar *)(cfg->vol);
     float  *field;
     float4 *Ppos,*Pdir,*Plen,*Plen0;
     uint   *Pseed;
     float  *Pdet;
     uint    detected=0,sharedbuf=0;

     uchar *gmedia;
     float4 *gPpos,*gPdir,*gPlen;
     uint   *gPseed,*gdetected;
     float  *gPdet,*gsrcpattern,*gfield,*genergy;
     MCXParam param={cfg->steps,minstep,0,0,cfg->tend,R_C0*cfg->unitinmm,
                     cfg->issave2pt,cfg->isreflect,cfg->isrefint,cfg->issavedet,1.f/cfg->tstep,
		     p0,c0,maxidx,uint3(0,0,0),cp0,cp1,uint2(0,0),cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->srctype,
		     cfg->srcparam1,cfg->srcparam2,cfg->voidtime,cfg->maxdetphoton,
		     cfg->medianum-1,cfg->detnum,0,0,cfg->reseedlimit,ABS(cfg->sradius+2.f)<1e-5 /*isatomic*/,
		     cfg->maxvoidstep,cfg->issaveseed>0,cfg->maxdetphoton*(cfg->medianum+1),0,0};
     int detreclen=cfg->medianum+1+(cfg->issaveseed>0)*RAND_BUF_LEN;
     if(param.isatomic)
         param.skipradius2=0.f;

     if(cfg->respin>1){
         field=(float *)calloc(sizeof(float)*dimxyz,cfg->maxgate*2);
     }else{
         field=(float *)calloc(sizeof(float)*dimxyz,cfg->maxgate); //the second half will be used to accumulate
     }

     if(cfg->nthread%cfg->nblocksize)
     	cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;
     param.threadphoton=cfg->nphoton/cfg->nthread/cfg->respin;
     param.oddphotons=cfg->nphoton/cfg->respin-param.threadphoton*cfg->nthread;
     totalgates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(totalgates>cfg->maxgate && cfg->isnormalized){
         MCX_FPRINTF(stderr,"WARNING: GPU memory can not hold all time gates, disabling normalization to allow multiple runs\n");
         cfg->isnormalized=0;
     }
     fieldlen=dimxyz*cfg->maxgate;

     mcgrid.x=cfg->nthread/cfg->nblocksize;
     mcblock.x=cfg->nblocksize;

     clgrid.x=cfg->dim.x;
     clgrid.y=cfg->dim.y;
     clblock.x=cfg->dim.z;

     if(cfg->debuglevel & MCX_DEBUG_RNG){
           param.twin0=cfg->tstart;
           param.twin1=cfg->tend;
           Pseed=(uint*)malloc(sizeof(uint)*RAND_SEED_LEN);
           for (i=0; i<RAND_SEED_LEN; i++)
		Pseed[i]=rand();
           mcx_cu_assess(cudaMalloc((void **) &gPseed, sizeof(uint)*RAND_SEED_LEN),__FILE__,__LINE__);
	   mcx_cu_assess(cudaMemcpy(gPseed, Pseed, sizeof(uint)*RAND_SEED_LEN,  cudaMemcpyHostToDevice),__FILE__,__LINE__);
           mcx_cu_assess(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen),__FILE__,__LINE__);
           mcx_cu_assess(cudaMemset(gfield,0,sizeof(float)*fieldlen),__FILE__,__LINE__); // cost about 1 ms
           mcx_cu_assess(cudaMemcpyToSymbol(gcfg,   &param, sizeof(MCXParam), 0, cudaMemcpyHostToDevice),__FILE__,__LINE__);

           tic=StartTimer();
           MCX_FPRINTF(cfg->flog,"generating %d random numbers ... \t",fieldlen); fflush(cfg->flog);
           mcx_test_rng<<<1,1>>>(gfield,gPseed);
           tic1=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving random numbers ... \t",tic1-tic);
           mcx_cu_assess(cudaGetLastError(),__FILE__,__LINE__);

           cudaMemcpy(field, gfield,sizeof(float)*dimxyz*cfg->maxgate,cudaMemcpyDeviceToHost);
           MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);
	   if(cfg->exportfield)
	       memcpy(cfg->exportfield,field,fieldlen*sizeof(float));
	   else{
               MCX_FPRINTF(cfg->flog,"saving data to file ...\t");
	       mcx_savedata(field,fieldlen,timegate>0,"mc2",cfg);
               MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
               fflush(cfg->flog);
           }
	   cudaFree(gfield);
	   cudaFree(gPseed);
	   free(field);
	   free(Pseed);

	   cudaThreadExit();
	   return;
     }
     
     Ppos=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Pdir=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Plen=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Plen0=(float4*)malloc(sizeof(float4)*cfg->nthread);
     energy=(float*)calloc(cfg->nthread*3,sizeof(float));
     Pdet=(float*)calloc(cfg->maxdetphoton,sizeof(float)*(detreclen));
     Pseed=(uint*)malloc(sizeof(uint)*cfg->nthread*RAND_SEED_LEN);

     mcx_cu_assess(cudaMalloc((void **) &gmedia, sizeof(uchar)*(dimxyz)),__FILE__,__LINE__);
     //cudaBindTexture(0, texmedia, gmedia);
     mcx_cu_assess(cudaMalloc((void **) &gfield, sizeof(float)*fieldlen),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gPpos, sizeof(float4)*cfg->nthread),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gPdir, sizeof(float4)*cfg->nthread),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gPlen, sizeof(float4)*cfg->nthread),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gPdet, sizeof(float)*cfg->maxdetphoton*(detreclen)),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gdetected, sizeof(uint)),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &genergy, sizeof(float)*cfg->nthread*3),__FILE__,__LINE__);
     mcx_cu_assess(cudaMalloc((void **) &gPseed, sizeof(uint)*cfg->nthread*RAND_SEED_LEN),__FILE__,__LINE__);
     if(cfg->srctype==MCX_SRC_PATTERN)
         mcx_cu_assess(cudaMalloc((void **) &gsrcpattern, sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w)),__FILE__,__LINE__);

#ifndef SAVE_DETECTORS
     if(cfg->issavedet){
           MCX_FPRINTF(stderr,"WARNING: this MCX binary can not save partial path, please use mcx_det or mcx_det_cached\n");
           cfg->issavedet=0;
     }
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
     	srand(cfg->seed);
     else
        srand(time(0));
	
     for (i=0; i<cfg->nthread; i++) {
	   Ppos[i]=p0;  // initial position
           Pdir[i]=c0;
           Plen[i]=float4(0.f,0.f,param.minaccumtime,0.f);
     }
     mcx_printheader(cfg);

     tic=StartTimer();
#ifdef MCX_TARGET_NAME
     MCX_FPRINTF(cfg->flog,"- variant name: [%s] compiled for GPU Capability [%d] with CUDA [%d]\n",
             MCX_TARGET_NAME,MCX_CUDA_ARCH,CUDART_VERSION);
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
     MCX_FPRINTF(cfg->flog,"threadph=%d oddphotons=%d np=%d nthread=%d repetition=%d\n",param.threadphoton,param.oddphotons,
           cfg->nphoton,cfg->nthread,cfg->respin);
     MCX_FPRINTF(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);

     cudaMemcpy(gmedia, media, sizeof(uchar) *dimxyz, cudaMemcpyHostToDevice);
     cudaMemcpy(genergy,energy,sizeof(float) *cfg->nthread*3, cudaMemcpyHostToDevice);
     if(cfg->srcpattern)
         cudaMemcpy(gsrcpattern,cfg->srcpattern,sizeof(float)*(int)(cfg->srcparam1.w*cfg->srcparam2.w), cudaMemcpyHostToDevice);

     cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(gdetpos, cfg->detpos,  cfg->detnum*sizeof(float4), 0, cudaMemcpyHostToDevice);

     MCX_FPRINTF(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

     /*
         if one has to simulate a lot of time gates, using the GPU global memory
	 requires extra caution. If the total global memory is bigger than the total
	 memory to save all the snapshots, i.e. size(field)*(tend-tstart)/tstep, one
	 simply sets cfg->maxgate to the total gate number; this will run GPU kernel
	 once. If the required memory is bigger than the video memory, set cfg->maxgate
	 to a number which fits, and the snapshot will be saved with an increment of 
	 cfg->maxgate snapshots. In this case, the later simulations will restart from
	 photon launching and exhibit redundancies.
	 
	 The calculation of the energy conservation will only reflect the last simulation.
     */
#ifdef  USE_CACHEBOX
     if(cfg->sradius>EPS || ABS(cfg->sradius+1.f)<1e-5f)
        sharedbuf+=sizeof(float)*((cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1));
#endif
     if(cfg->issavedet)
        sharedbuf+=cfg->nblocksize*sizeof(float)*(cfg->medianum-1);
#ifdef USE_MT_RAND
     sharedbuf+=(N+2)*sizeof(uint); // MT RNG uses N+2 uint in the shared memory
#endif

     MCX_FPRINTF(cfg->flog,"requesting %d bytes of shared memory\n",sharedbuf);

     //simulate for all time-gates in maxgate groups per run
     for(timegate=0;timegate<totalgates;timegate+=cfg->maxgate){

       param.twin0=cfg->tstart+cfg->tstep*timegate;
       param.twin1=param.twin0+cfg->tstep*cfg->maxgate;
       cudaMemcpyToSymbol(gcfg,   &param,     sizeof(MCXParam), 0, cudaMemcpyHostToDevice);

       MCX_FPRINTF(cfg->flog,"lauching MCX simulation for time window [%.2ens %.2ens] ...\n"
           ,param.twin0*1e9,param.twin1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<cfg->respin;iter++){
           cudaMemset(gfield,0,sizeof(float)*fieldlen); // cost about 1 ms
           cudaMemset(gPdet,0,sizeof(float)*cfg->maxdetphoton*(detreclen));
           cudaMemset(gdetected,0,sizeof(float));

 	   cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
	   cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
	   cudaMemcpy(gPlen,  Plen,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
           for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++)
		Pseed[i]=rand();
	   cudaMemcpy(gPseed, Pseed, sizeof(uint)*cfg->nthread*RAND_SEED_LEN,  cudaMemcpyHostToDevice);

           tic0=GetTimeMillis();
           MCX_FPRINTF(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);
           mcx_main_loop<<<mcgrid,mcblock,sharedbuf>>>(gmedia,gfield,genergy,gPseed,gPpos,gPdir,gPlen,gPdet,gdetected,gsrcpattern);

           cudaThreadSynchronize();
	   cudaMemcpy(&detected, gdetected,sizeof(uint),cudaMemcpyDeviceToHost);
           tic1=GetTimeMillis();
	   toc+=tic1-tic0;
           MCX_FPRINTF(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",tic1-tic);
           mcx_cu_assess(cudaGetLastError(),__FILE__,__LINE__);

           cudaMemcpy(Plen0,  gPlen,  sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
           cfg->his.totalphoton=0;
           for(i=0;i<cfg->nthread;i++)
	      cfg->his.totalphoton+=int(Plen0[i].w+0.5f);
           photoncount+=cfg->his.totalphoton;

#ifdef SAVE_DETECTORS
           if(cfg->issavedet){
           	cudaMemcpy(Pdet, gPdet,sizeof(float)*cfg->maxdetphoton*(detreclen),cudaMemcpyDeviceToHost);
	        mcx_cu_assess(cudaGetLastError(),__FILE__,__LINE__);
		if(detected>cfg->maxdetphoton){
			MCX_FPRINTF(cfg->flog,"WARNING: the detected photon (%d) \
is more than what your have specified (%d), please use the -H option to specify a greater number\t"
                           ,detected,cfg->maxdetphoton);
		}else{
			MCX_FPRINTF(cfg->flog,"detected %d photons\t",detected);
		}
		cfg->his.unitinmm=cfg->unitinmm;
		cfg->his.detected=detected;
		cfg->his.savedphoton=MIN(detected,cfg->maxdetphoton);
		if(cfg->issaveseed)
		    cfg->his.seedbyte=sizeof(RandType)*RAND_BUF_LEN;
		if(cfg->exportdetected){
                        detected=exportedcount+cfg->his.savedphoton;
                        if(detected<cfg->maxdetphoton){
			    if(cfg->issaveseed>0)
			        memcpy(cfg->exportdetected+detected*(cfg->medianum+1),cfg->exportdetected+cfg->maxdetphoton*(cfg->medianum+1),detected*RAND_BUF_LEN*sizeof(float));
                            cfg->exportdetected=(float*)realloc(cfg->exportdetected,detected*detreclen*sizeof(float));
			}
	                memcpy(cfg->exportdetected+exportedcount*(detreclen),Pdet,cfg->his.savedphoton*(detreclen)*sizeof(float));
                        exportedcount+=cfg->his.savedphoton;
		}else{
			mcx_savedetphoton(Pdet,Pdet+cfg->maxdetphoton*(cfg->medianum+1),detected,timegate>0,cfg);
                }
	   }
#endif

	   //handling the 2pt distributions
           if(cfg->issave2pt){
               cudaMemcpy(field, gfield,sizeof(float) *dimxyz*cfg->maxgate,cudaMemcpyDeviceToHost);
               MCX_FPRINTF(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

               if(cfg->respin>1){
                   for(i=0;i<(int)fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
               if(iter+1==cfg->respin){
                   if(cfg->respin>1)  //copy the accumulated fields back
                       memcpy(field,field+fieldlen,sizeof(float)*fieldlen);

                   if(cfg->isnormalized){
                       MCX_FPRINTF(cfg->flog,"normalizing raw data ...\t");

                       cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*3,cudaMemcpyDeviceToHost);
                       eabsorp=0.f;
                       for(i=1;i<cfg->nthread;i++){
                           energy[0]+=energy[3*i];
       	       	       	   energy[1]+=energy[3*i+1];
       	       	       	   energy[2]+=energy[3*i+2];
                       }
		       for(i=0;i<cfg->nthread;i++)
                           eabsorp+=Plen0[i].z;  // the accumulative absorpted energy near the source
                       eabsorp+=energy[1];

                       scale=(energy[2]-energy[0])/(energy[2]*Vvox*cfg->tstep*eabsorp);
		       if(cfg->unitinmm!=1.f) 
		          scale*=cfg->unitinmm; /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */
                       MCX_FPRINTF(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   MCX_FPRINTF(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);

		   if(cfg->exportfield)
	                   memcpy(cfg->exportfield+fieldlen*(timegate/cfg->maxgate),field,fieldlen*sizeof(float));
		   else{
                           MCX_FPRINTF(cfg->flog,"saving data to file ...\t");
	                   mcx_savedata(field,fieldlen,timegate>0,"mc2",cfg);
                           MCX_FPRINTF(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
                           fflush(cfg->flog);
                   }
               }
           }
       }
       if(param.twin1<cfg->tend){
            cudaMemset(genergy,0,sizeof(float)*cfg->nthread*3);
       }
     }
     if(cfg->exportdetected)
         cfg->his.savedphoton=exportedcount;

     cudaMemcpy(Ppos,  gPpos, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pdir,  gPdir, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Plen,  gPlen, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pseed, gPseed,sizeof(uint)  *cfg->nthread*RAND_SEED_LEN,   cudaMemcpyDeviceToHost);
     cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*3,cudaMemcpyDeviceToHost);

     for (i=0; i<cfg->nthread; i++) {
          energyloss+=energy[3*i];
          energyabsorbed+=energy[3*i+1];
          energylaunched+=energy[3*i+2];
     }

#ifdef TEST_RACING
     {
       float totalcount=0.f,hitcount=0.f;
       for (i=0; i<fieldlen; i++)
          hitcount+=field[i];
       for (i=0; i<cfg->nthread; i++)
	  totalcount+=Pseed[i];
     
       MCX_FPRINTF(cfg->flog,"expected total recording number: %f, got %f, missed %f\n",
          totalcount,hitcount,(totalcount-hitcount)/totalcount);
     }
#endif

     printnum=cfg->nthread<cfg->printnum?cfg->nthread:cfg->printnum;
     for (i=0; i<(int)printnum; i++) {
            MCX_FPRINTF(cfg->flog,"% 4d[A% f % f % f]C%3d J%5d W% 8f(P%.13f %.13f %.13f)T% 5.3e L% 5.3f %.0f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i]);
     }
     // total energy here equals total simulated photons+unfinished photons for all threads
     MCX_FPRINTF(cfg->flog,"simulated %d photons (%d) with %d threads (repeat x%d)\nMCX simulation speed: %.2f photon/ms\n",
             photoncount,cfg->nphoton,cfg->nthread,cfg->respin,(double)photoncount/toc); fflush(cfg->flog);
     MCX_FPRINTF(cfg->flog,"exit energy:%16.8e + absorbed energy:%16.8e = total: %16.8e\n",
             energyloss,energylaunched-energyloss,energylaunched);fflush(cfg->flog);
     fflush(cfg->flog);

     cudaFree(gmedia);
     cudaFree(gfield);
     cudaFree(gPpos);
     cudaFree(gPdir);
     cudaFree(gPlen);
     cudaFree(gPseed);
     cudaFree(genergy);
     cudaFree(gPdet);
     cudaFree(gdetected);

     cudaThreadExit();

     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Plen0);
     free(Pseed);
     free(Pdet);
     free(energy);
     free(field);
}
