/////////////////////////////////////////////////////////////////////
//
//  MC Extreme  - GPU accelerated Monte-Carlo Simulation
//  
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//  History: 
//    2009/02/14 initial version written in BrookGPU
//    2009/02/15 translated to CUDA
//    2009/02/20 translated to Brook+
//    2009/02/21 added MT random number generator initial version
//    2009/02/24 MT rand now works fine, added FAST_MATH
//    2009/02/25 added CACHE_MEDIA read
//    2009/02/27 early support of boundary reflection
//
// License: unpublished version, use by author's permission only
//
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "br2cu.h"
#include "mt_rand_s.cu"
#include "tictoc.h"

// dimension of the target domain
#define DIMX 128
#define DIMY 128
#define DIMZ 128
/*
#define DIMX 256
#define DIMY 256
#define DIMZ 256
*/

#define DIMYZ (DIMY*DIMZ)
#define DIMXYZ (DIMX*DIMY*DIMZ)
#define INDXYZ(ii,jj,kk)  ((ii)*DIMYZ+(jj)*DIMZ+(kk))
#define MAX_MT_RAND 4294967296
#define R_MAX_MT_RAND 2.32830643653870e-10
#define TWO_PI 6.28318530717959f


#ifdef __DEVICE_EMULATION__
#define MAX_N      1
#define MAX_THREAD 1
#else
#define MAX_N      1024
#define MAX_THREAD 128
#endif
#define MAX_EVENT  1
#define MAX_PROP   256


#ifdef CACHE_MEDIA
#define MAX_MEDIA_CACHE   61440  /*52k for local media read*/
#define MAX_WRITE_CACHE   (MAX_MEDIA_CACHE>>4)
#define MEDIA_BITS  2            /*2^2=4 media types*/
#define MEDIA_PACK  ((8/MEDIA_BITS)>>1)            /*one byte packs 2^MEDIA_PACK voxel*/
#define MEDIA_MOD   ((1<<MEDIA_PACK)-1)    /*one byte packs 2^MEDIA_PACK voxel*/
#define MEDIA_MASK  ((1<<(MEDIA_BITS))-1)
#endif

#define MINUS_SAME_VOXEL -9999.f

#define GPUDIV(a,b)     __fdividef((a),(b))

#ifdef  FAST_MATH      /*define this to use the fast math functions*/
#define	GPULOG(x)       __logf(x)
#define GPUSIN(x)       __sinf(x)
#define	GPUSINCOS(x,a,b)       __sincosf(x,a,b)
#else
#define GPULOG(x)       logf(x)
#define GPUSIN(x)       sinf(x)
#define GPUSINCOS(x,a,b)      sincosf(x,a,b)
#endif


typedef unsigned char uchar;

/******************************
typedef struct PhotonData {
  float4 pos;  // x,y,z,weight
  float4 dir;  // ix,iy,iz,dummy
  float3 len; // resid,tot,count
  uint   seed; // random seed
} Photon;
******************************/

__constant__ float3 gproperty[MAX_PROP];

#ifdef CACHE_MEDIA
__constant__ uchar  gmediacache[MAX_MEDIA_CACHE];
#endif

// pass as many pre-computed values as possible to utilize the constant memory 

kernel void mcx_main_loop(int totalmove,uchar media[],float field[],float3 vsize,float minstep, 
     float lmax, float gg, float gg2,float ggx2, float one_add_gg2, float one_sub_gg2, float one_sub_gg,
     float4 p0,float4 c0,float3 maxidx,uint3 cp0,uint3 cp1,uint2 cachebox,uchar doreflect,
     uint n_seed[],float4 n_pos[],float4 n_dir[],float3 n_len[]){

     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     float4 npos=n_pos[idx];
     float4 ndir=n_dir[idx];
     float3 nlen=n_len[idx];
     float4 npos0;
     float3 htime;

     int i, idx1d, idx1dold,idxorig, mediaid;
     float flipdir,n1,Rtotal;
#ifdef CACHE_MEDIA
     int incache=0,incache0=0,cachebyte=-1,cachebyte0=-1;
#endif
     uint   ran;
     float3 prop;

     float len,cphi,sphi,theta,stheta,ctheta,tmp0,tmp1;


     mt19937si(n_seed[idx]);
     __syncthreads();

     // assuming the initial positions are within the domain
     idx1d=int(floorf(npos.x)*DIMYZ+floorf(npos.y)*DIMZ+floorf(npos.z));
     idxorig=idx1d;
     mediaid=media[idx1d];

#ifdef CACHE_MEDIA
     if(npos.x>=cp0.x && npos.x<=cp1.x && npos.y>=cp0.y && npos.y<=cp1.y && npos.z>=cp0.z && npos.z<=cp1.z){
	  incache=1;
          incache0=1;
          cachebyte=int(floorf(npos.x-cp0.x)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.z-cp0.z));
          cachebyte0=cachebyte;
          mediaid=(int)gmediacache[cachebyte>>MEDIA_PACK];
          mediaid=(mediaid >> (cachebyte & MEDIA_MOD)*MEDIA_BITS) & MEDIA_MASK;
     }
#endif

     if(mediaid==0) {
          return; /* the initial position is not within the medium*/
     }

     for(i=0;i<totalmove;i++){
	  if(nlen.x<=0.f) {  /* if this photon finished the current jump */

	       ran=mt19937s(); /*random number [0,MAX_MT_RAND)*/

   	       nlen.x=-GPULOG(ran*R_MAX_MT_RAND); /*probability of the next jump*/

	       if(npos.w<1.f){ /*weight*/
                       /*random arimuthal angle*/
                       ran=mt19937s();
		       tmp0=TWO_PI*ran*R_MAX_MT_RAND; /*will be reused to minimize register*/
                       GPUSINCOS(tmp0,&sphi,&cphi);

                       /*Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234*/
                       /*see Boas2003*/
		       ran=mt19937s();
                       if(gg>1e-10){
		           tmp0=GPUDIV(one_sub_gg2,(one_sub_gg+ggx2*ran*R_MAX_MT_RAND));
		           tmp0*=tmp0;
		           tmp0=GPUDIV((one_add_gg2-tmp0),ggx2);
		           theta=acosf(tmp0);
		           stheta=GPUSIN(theta);
		           ctheta=tmp0;
                       }else{
			   theta=TWO_PI*ran*R_MAX_MT_RAND;
                           GPUSINCOS(theta,&stheta,&ctheta);
                       }
		       if( ndir.z>-1.f && ndir.z<1.f ) {
		           tmp0=1.f-ndir.z*ndir.z;   /*reuse tmp to minimize registers*/
		           tmp1=rsqrtf(tmp0);
		           tmp1=stheta*tmp1;
			   if(stheta>1e-20) {  /*strange: if stheta=0, I will get nan :(  FQ */
			     ndir=float4(
				tmp1*(ndir.x*ndir.z*cphi - ndir.y*sphi) + ndir.x*ctheta,
				tmp1*(ndir.y*ndir.z*cphi + ndir.x*sphi) + ndir.y*ctheta,
				-tmp1*tmp0*cphi                         + ndir.z*ctheta,
				ndir.w
				);
                             }
		       }else{
			   ndir=float4(stheta*cphi,stheta*sphi,ctheta,ndir.w);
 		       }
                       ndir.w++;
	       }
	  }


          n1=prop.z;
	  prop=gproperty[mediaid];
	  len=minstep*prop.y;

          npos0=npos;
	  if(len>nlen.x){  /*scattering ends in this voxel*/
               tmp0=GPUDIV(nlen.x,prop.y);
   	       npos=float4(npos.x+ndir.x*tmp0,npos.y+ndir.y*tmp0,npos.z+ndir.z*tmp0,npos.w*expf(-prop.x * tmp0 ));
	       nlen.x=MINUS_SAME_VOXEL;
	       nlen.y+=tmp0;
	  }else{                      /*otherwise, move minstep*/
   	       npos=float4(npos.x+ndir.x,npos.y+ndir.y,npos.z+ndir.z,npos.w*expf(-prop.x * minstep ));
	       nlen.x-=len;     /*remaining probability*/
	       nlen.y+=minstep; /*total moved length along the current jump*/
               idx1dold=idx1d;
               idx1d=int(floorf(npos.x)*DIMYZ+floorf(npos.y)*DIMZ+floorf(npos.z));
#ifdef CACHE_MEDIA     
               if(npos.x>=cp0.x && npos.x<=cp1.x && npos.y>=cp0.y && npos.y<=cp1.y && npos.z>=cp0.z && npos.z<=cp1.z){
                    incache=1;
                    cachebyte=int(floorf(npos.x-cp0.x)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.z-cp0.z));
               }else{
		    incache=0;
               }
#endif
	  }

#ifdef CACHE_MEDIA
          if(incache){
		mediaid=(int)gmediacache[cachebyte>>MEDIA_PACK];
                mediaid=(mediaid >> (cachebyte & MEDIA_MOD)*MEDIA_BITS) & MEDIA_MASK;
          }else{
#endif
                mediaid=media[idx1d];
#ifdef CACHE_MEDIA
          }
#endif

	  if(mediaid==0||nlen.y>lmax||npos.x<0||npos.y<0||npos.z<0||npos.x>maxidx.x||npos.y>maxidx.y||npos.z>maxidx.z){
	      /*if hit the boundary or exit the domain, launch a new one*/

              /*time to hit the wall in each direction*/
              htime.x=(ndir.x>1e-10||ndir.x<-1e-10)?(floorf(npos.x)+(ndir.x>0.f)-npos.x)/ndir.x:1e10; /*this approximates*/
              htime.y=(ndir.y>1e-10||ndir.y<-1e-10)?(floorf(npos.y)+(ndir.y>0.f)-npos.y)/ndir.y:1e10;
              htime.z=(ndir.z>1e-10||ndir.z<-1e-10)?(floorf(npos.z)+(ndir.z>0.f)-npos.z)/ndir.z:1e10;
              tmp0=fminf(fminf(htime.x,htime.y),htime.z);
              flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));
              prop=gproperty[mediaid];

#ifdef __DEVICE_EMULATION__
              printf("--> ID%d J%d C%d len %f flip %f %f!=%f dir=%f %f %f \n",idx,(int)ndir.w,
                  (int)nlen.z,nlen.y, flipdir, n1,prop.z,ndir.x,ndir.y,ndir.z);
#endif

              /*I don't have the luxury to declare more vars in a kernel, so, I recycled some of old ones*/

              if(doreflect&&nlen.y<lmax && flipdir>0.f && n1!=prop.z){
                  tmp0=n1*n1;
                  tmp1=prop.z*prop.z;
                  if(flipdir>=3.f) { /*flip in z axis*/
                     cphi=fabs(ndir.z);
                     sphi=ndir.x*ndir.x+ndir.y*ndir.y;
                     ndir.z=-ndir.z;
                  }else if(flipdir>=2.f){ /*flip in y axis*/
                     cphi=fabs(ndir.y);
       	       	     sphi=ndir.x*ndir.x+ndir.z*ndir.z;
                     ndir.y=-ndir.y;
                  }else if(flipdir>=1.f){ /*flip in x axis*/
                     cphi=fabs(ndir.x);               /*cos(si)*/
                     sphi=ndir.y*ndir.y+ndir.z*ndir.z; /*sin(si)^2*/
                     ndir.x=-ndir.x;
                  }
                  npos=npos0;   /*move back*/
                  idx1d=idx1dold;
                  len=1.f-GPUDIV(tmp0,tmp1)*sphi;   /*1-[n1/n2*sin(si)]^2*/
                  if(len>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.z*cphi*sqrtf(len);
                     Rtotal=GPUDIV(ctheta-stheta,ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+GPUDIV(ctheta-stheta,ctheta+stheta))/2.f;
#ifdef __DEVICE_EMULATION__
printf("  dir=%f %f %f htime=%f %f %f Rs=%f\n",ndir.x,ndir.y,ndir.z,htime.x,htime.y,htime.z,Rtotal);
printf("  ID%d J%d C%d flip=%3f (%d %d) cphi=%f sphi=%f npos=%f %f %f npos0=%f %f %f\n",idx,(int)ndir.w,(int)nlen.z,
            flipdir,idx1dold,idx1d,cphi,sphi,npos.x,npos.y,npos.z,npos0.x,npos0.y,npos0.z);
#endif
                     npos.w*=Rtotal;
                  } /* else, total internal reflection, no loss*/
                  mediaid=media[idx1d];
                  prop=gproperty[mediaid];
                  n1=prop.z;
              }else{
	          npos=p0;
	          ndir=c0;
	          nlen=float3(0.f,0.f,nlen.z+1);
                  idx1d=idxorig;
#ifdef CACHE_MEDIA
	          cachebyte=cachebyte0;
	          incache=incache0;
#endif
              }
	  }else if(nlen.x>0){
              field[idx1d]+=npos.w;
	  }
     }
     n_seed[idx]=(ran&0xffffffffu);
     n_pos[idx]=npos;
     n_dir[idx]=ndir;
     n_len[idx]=nlen;
}

void savedata(float *dat,int len,char *name){
     FILE *fp;
     fp=fopen(name,"wb");
     fwrite(dat,sizeof(float),len,fp);
     fclose(fp);
}

int main (int argc, char *argv[]) {

     float3 vsize=float3(1.f,1.f,1.f);
     float  minstep=1.f;
     float  lmax=1000.f;
     float  gg=0.98f;
     float4 p0=float4(DIMX/2,DIMY/2,DIMZ/4,1.f);
     float4 c0=float4(0.f,0.f,1.f,0.f);
     float3 maxidx=float3(DIMX-1,DIMY-1,DIMZ-1);
     float3 property[MAX_PROP]={float3(0.f,0.f,1.0f),float3(0.009f,0.75f,1.37f),  // the 1st is air
                                float3(0.006f,0.75f,1.37f),float3(0.009f,0.95f,1.37f)};

     int i,j,k;
     int total=MAX_EVENT;
     int photoncount=0;
     int tic;
//     uint3 cp0=uint3(DIMX/2-30,DIMY/2-30,DIMZ/4),cp1=uint3(DIMX/2+30,DIMY/2+30,DIMZ/4+60);
     uint3 cp0=uint3(DIMX/2-10,DIMY/2-10,DIMZ/4),cp1=uint3(DIMX/2+10,DIMY/2+10,DIMZ/4+20);
     uint2 cachebox;

     dim3 GridDim(MAX_N/MAX_THREAD);
     dim3 BlockDim(MAX_THREAD);

     uchar  media[DIMXYZ];
     float  field[DIMXYZ];
#ifdef CACHE_MEDIA
     int count;
     uchar  mediacache[MAX_MEDIA_CACHE];
#endif

     float4 Ppos[MAX_N];
     float4 Pdir[MAX_N];
     float3 Plen[MAX_N];
     uint   Pseed[MAX_N];

     if(argc>1){
	   total=atoi(argv[1]);
     }

#ifdef CACHE_MEDIA
     printf("requested constant memory cache: %d (max allowed %d)\n",
         (cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1),(MAX_MEDIA_CACHE<<MEDIA_PACK));
     if((cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1)> (MAX_MEDIA_CACHE<<MEDIA_PACK)){
	printf("the requested cache size is too big\n");
	exit(1);
     }
#endif

     uchar *gmedia;
     cudaMalloc((void **) &gmedia, sizeof(uchar)*(DIMXYZ));
     float *gfield;
     cudaMalloc((void **) &gfield, sizeof(float)*(DIMXYZ));

     float4 *gPpos;
     cudaMalloc((void **) &gPpos, sizeof(float4)*(MAX_N));
     float4 *gPdir;
     cudaMalloc((void **) &gPdir, sizeof(float4)*(MAX_N));
     float3 *gPlen;
     cudaMalloc((void **) &gPlen, sizeof(float3)*(MAX_N));
     uint   *gPseed;
     cudaMalloc((void **) &gPseed, sizeof(uint)*(MAX_N));


     memset(field,0,sizeof(float)*DIMXYZ);
     memset(media,0,sizeof(uchar)*DIMXYZ);

     for (i=DIMX/4; i<3*DIMX/4; i++)
      for (j=DIMY/4; j<3*DIMY/4; j++)
       for (k=DIMZ/4; k<3*DIMZ/4; k++) {
           media[INDXYZ(i,j,k)]=1; 
       }

     cachebox.x=(cp1.z-cp0.z+1);
     cachebox.y=(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1);

#ifdef CACHE_MEDIA
     count=0;
     memset(mediacache,0,MAX_MEDIA_CACHE);
     for (i=cp0.x; i<=cp1.x; i++)
      for (j=cp0.y; j<=cp1.y; j++)
       for (k=cp0.z; k<=cp1.z; k++) {
//         printf("[%d %d %d]: %d %d %d %d (%d)\n",i,j,k,count,MEDIA_MASK,count>>MEDIA_PACK,(count & MEDIA_MOD)*MEDIA_BITS,
//                (media[INDXYZ(i,j,k)] & MEDIA_MASK )<<((count & MEDIA_MOD)*MEDIA_BITS) );
           mediacache[count>>MEDIA_PACK] |=  (media[INDXYZ(i,j,k)] & MEDIA_MASK )<<((count & MEDIA_MOD)*MEDIA_BITS );
           count++;
       }
#endif

     srand(time(0));
     for (i=0; i<MAX_N; i++) {
	   Ppos[i]=p0;  /* initial position */
           Pdir[i]=c0;
           Plen[i]=float3(0.f,0.f,0.f);
	   Pseed[i]=rand();
     }

     tic=GetTimeMillis();

     cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*MAX_N,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*MAX_N,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPlen,  Plen,  sizeof(float3)*MAX_N,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPseed, Pseed, sizeof(uint)*MAX_N,     cudaMemcpyHostToDevice);
     cudaMemcpy(gfield, field, sizeof(float)*DIMXYZ, cudaMemcpyHostToDevice);
     cudaMemcpy(gmedia, media, sizeof(uchar)*DIMXYZ,cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(gproperty, property, MAX_PROP*sizeof(float3), 0, cudaMemcpyHostToDevice);
#ifdef CACHE_MEDIA
     cudaMemcpyToSymbol(gmediacache, mediacache, MAX_MEDIA_CACHE, 0, cudaMemcpyHostToDevice);
#endif

     printf("complete cudaMemcpy : %d ms\n",GetTimeMillis()-tic);

     mcx_main_loop<<<GridDim,BlockDim>>>(total,gmedia,gfield,vsize,minstep,lmax,gg,gg*gg,2.f*gg,\
        	 1.f+gg*gg,1.f-gg*gg,1.f-gg,p0,c0,maxidx,cp0,cp1,cachebox,0,gPseed,gPpos,gPdir,gPlen);

     printf("complete launching kernels : %d ms\n",GetTimeMillis()-tic);

     cudaMemcpy(Ppos,  gPpos, sizeof(float4)*MAX_N, cudaMemcpyDeviceToHost);

     printf("complete retrieving pos : %d ms\n",GetTimeMillis()-tic);

     cudaMemcpy(Pdir,  gPdir, sizeof(float4)*MAX_N, cudaMemcpyDeviceToHost);
     cudaMemcpy(Plen,  gPlen, sizeof(float3)*MAX_N, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pseed, gPseed,sizeof(uint)*MAX_N,   cudaMemcpyDeviceToHost);
     cudaMemcpy(field, gfield,sizeof(float)*DIMXYZ,cudaMemcpyDeviceToHost);

     printf("complete retrieving all : %d ms\n",GetTimeMillis()-tic);

     for (i=0; i<MAX_N; i++) {
	  photoncount+=(int)Plen[i].z;
     }
     total=MAX_N<16?MAX_N:16;
     for (i=0; i<total; i++) {
           printf("% 4d[A% f % f % f]C%3d J%3d% 8f(P% 6.3f % 6.3f % 6.3f)T% 5.3f L% 5.3f %f %f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].z,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i], Pdir[i].x*Pdir[i].x+Pdir[i].y*Pdir[i].y+Pdir[i].z*Pdir[i].z);
     }
     printf("simulating total photon %d\n",photoncount);
     savedata(field,DIMX*DIMY*DIMZ,"field.dat");

     cudaFree(gmedia);
#ifdef CACHE_MEDIA
     cudaFree(gmediacache);
#endif
     cudaFree(gfield);
     cudaFree(gPpos);
     cudaFree(gPdir);
     cudaFree(gPlen);
     cudaFree(gPseed);

     return 0;
}
