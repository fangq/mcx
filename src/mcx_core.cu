#include "br2cu.h"
#include "mcx_utils.h"


#ifdef USE_MT_RAND
#include "mt_rand_s.cu"
#else
#include "logistic_rand.cu"
#endif

#ifdef CACHE_MEDIA
//#define MAX_MEDIA_CACHE   61440  /*52k for local media read*/
#define MAX_MEDIA_CACHE   40000  /*52k for local media read*/
#define MAX_WRITE_CACHE   (MAX_MEDIA_CACHE>>4)
#define MEDIA_BITS  8            /*theoretically one can use smaller bits to pack more media*/
#define MEDIA_PACK  ((8/MEDIA_BITS)>>1)      /*one byte packs 2^MEDIA_PACK voxel*/
#define MEDIA_MOD   ((1<<MEDIA_PACK)-1)      /*one byte packs 2^MEDIA_PACK voxel*/
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

#define MAX_MT_RAND 4294967296
#define R_MAX_MT_RAND 2.32830643653870e-10
#define TWO_PI 6.28318530717959f
#define EPS    (1e-10f)

#ifdef __DEVICE_EMULATION__
#define MAX_THREAD 1
#else
#define MAX_THREAD 128
//#define MAX_THREAD 256
#endif
#define MAX_EVENT  1
#define MAX_PROP   256
#define C0   299792458000.f  /*in mm/s*/
#define R_C0 3.335640951981520e-12f /*1/C0 in s/mm*/

#define MIN(a,b)  ((a)<(b)?(a):(b))


typedef unsigned char uchar;

/******************************
typedef struct PhotonData {
  float4 pos;  // x,y,z,weight
  float4 dir;  // ix,iy,iz,scat_event
  float4 len;  // resid,tot,next_int,photon_count
  uint   seed; // random seed
} Photon;
******************************/

__constant__ float4 gproperty[MAX_PROP];

#ifdef CACHE_MEDIA
__constant__ uchar  gmediacache[MAX_MEDIA_CACHE];
#endif

// pass as many pre-computed values as possible to utilize the constant memory 

kernel void mcx_main_loop(int totalmove,uchar media[],float field[],float genergy[],float3 vsize,float minstep, 
     float twin0,float twin1, float tmax, uint3 dimlen, uchar isrowmajor, uchar save2pt, float Rtstep,
     float4 p0,float4 c0,float3 maxidx,uint3 cp0,uint3 cp1,uint2 cachebox,uchar doreflect,
     uint n_seed[],float4 n_pos[],float4 n_dir[],float4 n_len[]){

     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     float4 npos=n_pos[idx];
     float4 ndir=n_dir[idx];
     float4 nlen=n_len[idx];
     float4 npos0;
     float3 htime;
     float  minaccumtime=minstep*R_C0;
     float  energyloss=genergy[idx<<1];
     float  energyabsorbed=genergy[(idx<<1)+1];

     int i, idx1d, idx1dold,idxorig, mediaid, mediaidorig;

     float flipdir,n1,Rtotal;
#ifdef CACHE_MEDIA
     int incache=0,incache0=0,cachebyte=-1,cachebyte0=-1;
#endif

#ifdef USE_MT_RAND
     uint   ran;
#else
     RandType ran, t[RAND_BUF_LEN],tnew[RAND_BUF_LEN];
#endif

     float4 prop;

     float len,cphi,sphi,theta,stheta,ctheta,tmp0,tmp1;

#ifdef USE_MT_RAND
     mt19937si(n_seed[idx]);
     __syncthreads();
#else
     logistic_init(t,tnew,n_seed,idx);
#endif

     // assuming the initial positions are within the domain
     idx1d=isrowmajor?int(floorf(npos.x)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.z)):\
                      int(floorf(npos.z)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.x));
     idxorig=idx1d;
     mediaid=media[idx1d];
     mediaidorig=mediaid;
	  
#ifdef CACHE_MEDIA
     if(npos.x>=cp0.x && npos.x<=cp1.x && npos.y>=cp0.y && npos.y<=cp1.y && npos.z>=cp0.z && npos.z<=cp1.z){
	  incache=1;
          incache0=1;
          cachebyte=isrowmajor?int(floorf(npos.x-cp0.x)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.z-cp0.z)):\
	                       int(floorf(npos.z-cp0.z)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.x-cp0.x));
          cachebyte0=cachebyte;
          mediaid=gmediacache[cachebyte];
          mediaidorig=mediaid;
     }
#endif

     if(mediaid==0) {
          return; /* the initial position is not within the medium*/
     }
     prop=gproperty[mediaid];

     // using "while(nlen.w<totalmove)" loop will make this 4 times slower with the same amount of photons

     for(i=0;i<totalmove;i++){

#ifdef __DEVICE_EMULATION__
          printf("*i=%d (%d) L=%f w=%e a=%f\n",i,(int)nlen.w,nlen.x,npos.w,nlen.y);
#endif
	  if(nlen.x<=0.f) {  /* if this photon has finished the current jump */

#ifdef USE_MT_RAND
	       ran=mt19937s(); /*random number [0,MAX_MT_RAND)*/
   	       nlen.x=-GPULOG(ran*R_MAX_MT_RAND); /*probability of the next jump*/
#else
               logistic_rand(t,tnew,RAND_BUF_LEN-1); /*create 3 random numbers*/
               ran=logistic_uniform(t[0]);           /*shuffled values*/
               nlen.x= ((ran==0.f)?(-GPULOG((2.f*R_PI)*sqrtf(t[0]))):(-GPULOG(ran)));
#endif

#ifdef __DEVICE_EMULATION__
               printf("next scat len=%20.16e \n",nlen.x);
#endif
	       if(npos.w<1.f){ /*weight*/
                       /*random arimuthal angle*/
#ifdef USE_MT_RAND
                       ran=mt19937s();
		       tmp0=TWO_PI*ran*R_MAX_MT_RAND; /*will be reused to minimize register*/
#else
                       ran=t[1]; /*random number [0,MAX_MT_RAND)*/
                       tmp0=TWO_PI*logistic_uniform(ran); /*will be reused to minimize register*/
#endif
                       GPUSINCOS(tmp0,&sphi,&cphi);
#ifdef __DEVICE_EMULATION__
                       printf("next angle phi %20.16e\n",tmp0);
#endif

                       /*Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234*/
                       /*see Boas2002*/

#ifdef USE_MT_RAND
		       ran=mt19937s();
#else
                       ran=t[2]; /*random number [0,MAX_MT_RAND)*/
#endif

                       if(prop.w>EPS){  /*if prop.w is too small, the distribution of theta is bad*/
#ifdef USE_MT_RAND
		           tmp0=GPUDIV(1.f-prop.w*prop.w,(1.f-prop.w+2.f*prop.w*ran*R_MAX_MT_RAND));
#else
                           tmp0=GPUDIV(1.f-prop.w*prop.w,(1.f-prop.w+2.f*prop.w*logistic_uniform(ran) ));
#endif
		           tmp0*=tmp0;
		           tmp0=GPUDIV((1+prop.w*prop.w-tmp0),2.f*prop.w);

                           /* when ran=1, CUDA will give me 1.000002 for tmp0 which produces nan later*/
                           if(tmp0>1.f) tmp0=1.f;

		           theta=acosf(tmp0);
		           stheta=GPUSIN(theta);
		           ctheta=tmp0;
                       }else{  /*Wang1995 has acos(2*ran-1), rather than 2*pi*ran, need to check*/
#ifdef USE_MT_RAND
			   theta=TWO_PI*ran*R_MAX_MT_RAND;
#else
                           theta=TWO_PI*logistic_uniform(ran);
#endif

                           GPUSINCOS(theta,&stheta,&ctheta);
                       }
#ifdef __DEVICE_EMULATION__
                       printf("next scat angle theta %20.16e\n",theta);
#endif

		       if( ndir.z>-1.f+EPS && ndir.z<1.f-EPS ) {
		           tmp0=1.f-ndir.z*ndir.z;   /*reuse tmp to minimize registers*/
		           tmp1=rsqrtf(tmp0);
		           tmp1=stheta*tmp1;
//			   if(stheta>1e-20) {  /*strange: if stheta=0, I will get nan :(  FQ */
			     ndir=float4(
				tmp1*(ndir.x*ndir.z*cphi - ndir.y*sphi) + ndir.x*ctheta,
				tmp1*(ndir.y*ndir.z*cphi + ndir.x*sphi) + ndir.y*ctheta,
				-tmp1*tmp0*cphi                         + ndir.z*ctheta,
				ndir.w
				);
//                             }
#ifdef __DEVICE_EMULATION__
                               printf("new dir: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z);
#endif
		       }else{
			   ndir=float4(stheta*cphi,stheta*sphi,ctheta*ndir.z,ndir.w);
#ifdef __DEVICE_EMULATION__
                           printf("new dir-z: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z);
#endif
 		       }
                       ndir.w++;
	       }
	  }

          n1=prop.z;
	  prop=gproperty[mediaid];
	  len=minstep*prop.y; /*Wang1995: minstep*(prop.x+prop.y)*/

          npos0=npos;
	  if(len>nlen.x){  /*scattering ends in this voxel: mus*minstep > s */
               tmp0=GPUDIV(nlen.x,prop.y);
	       energyabsorbed+=npos.w;
   	       npos=float4(npos.x+ndir.x*tmp0,npos.y+ndir.y*tmp0,npos.z+ndir.z*tmp0,npos.w*expf(-prop.x * tmp0 ));
	       energyabsorbed-=npos.w;
	       nlen.x=MINUS_SAME_VOXEL;
	       nlen.y+=tmp0*prop.z*R_C0;  // accumulative time
#ifdef __DEVICE_EMULATION__
               printf(">>ends in voxel %f<%f %f [%d]\n",nlen.x,len,prop.y,idx1d);
#endif
	  }else{                      /*otherwise, move minstep*/
	       energyabsorbed+=npos.w;
   	       npos=float4(npos.x+ndir.x,npos.y+ndir.y,npos.z+ndir.z,npos.w*expf(-prop.x * minstep ));
	       energyabsorbed-=npos.w;
	       nlen.x-=len;     /*remaining probability: sum(s_i*mus_i)*/
	       nlen.y+=minaccumtime*prop.z; /*total time*/
#ifdef __DEVICE_EMULATION__
               printf(">>keep going %f<%f %f [%d] %e %e\n",nlen.x,len,prop.y,idx1d,nlen.y,nlen.z);
#endif
	  }

          idx1dold=idx1d;
          idx1d=isrowmajor?int(floorf(npos.x)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.z)):\
                           int(floorf(npos.z)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.x));
#ifdef __DEVICE_EMULATION__
                           printf("old and new voxel: %d<->%d\n",idx1dold,idx1d);
#endif
#ifdef CACHE_MEDIA  
          if(npos.x>=cp0.x && npos.x<=cp1.x && npos.y>=cp0.y && npos.y<=cp1.y && npos.z>=cp0.z && npos.z<=cp1.z){
               incache=1;
               cachebyte=isrowmajor?int(floorf(npos.x-cp0.x)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.z-cp0.z)):\
                                    int(floorf(npos.z-cp0.z)*cachebox.y+floorf(npos.y-cp0.y)*cachebox.x+floorf(npos.x-cp0.x));
          }else{
	       incache=0;
          }
#endif
          if(npos.x<0||npos.y<0||npos.z<0||npos.x>=maxidx.x||npos.y>=maxidx.y||npos.z>=maxidx.z){
	      mediaid=0;
	  }else{
#ifdef CACHE_MEDIA
              mediaid=incache?gmediacache[cachebyte]:media[idx1d];
#else
              mediaid=media[idx1d];
#endif
          }
	  if(mediaid==0||nlen.y>tmax||nlen.y>twin1){
	      /*if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one*/

              /*time to hit the wall in each direction*/
              htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floorf(npos0.x)+(ndir.x>0.f)-npos0.x)/ndir.x:1e10;
              htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floorf(npos0.y)+(ndir.y>0.f)-npos0.y)/ndir.y:1e10f;
              htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floorf(npos0.z)+(ndir.z>0.f)-npos0.z)/ndir.z:1e10f;
              tmp0=fminf(fminf(htime.x,htime.y),htime.z);
              flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));

              htime.x=floorf(npos0.x+tmp0*1.0001*ndir.x); /*move to the 1st intersection pt*/
       	      htime.y=floorf(npos0.y+tmp0*1.0001*ndir.y);
       	      htime.z=floorf(npos0.z+tmp0*1.0001*ndir.z);

              if(htime.x>=0&&htime.y>=0&&htime.z>=0&&htime.x<maxidx.x&&htime.y<maxidx.y&&htime.z<maxidx.z){
                  if( media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)]){ /*hit again*/

#ifdef __DEVICE_EMULATION__
                     printf(" first try failed: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime.x,htime.y,htime.z,
                           media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)], maxidx.x, maxidx.y,maxidx.z);
#endif
                     htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floorf(npos.x)+(ndir.x<0.f)-npos.x)/(-ndir.x):1e10;
                     htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floorf(npos.y)+(ndir.y<0.f)-npos.y)/(-ndir.y):1e10f;
                     htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floorf(npos.z)+(ndir.z<0.f)-npos.z)/(-ndir.z):1e10f;
                     tmp0=fminf(fminf(htime.x,htime.y),htime.z);
                     flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));
                }
              }
              prop=gproperty[mediaid];

#ifdef __DEVICE_EMULATION__
              printf("->ID%d J%d C%d tlen %e flip %d %.1f!=%.1f dir=%f %f %f pos=%f %f %f\n",idx,(int)ndir.w,
                  (int)nlen.w,nlen.y, (int)flipdir, n1,prop.z,ndir.x,ndir.y,ndir.z,npos.x,npos.y,npos.z);
#endif

              /*recycled some old register variables to save memory*/

	      /*if hit boundary within the time window and is n-mismatched, rebound*/

              if(doreflect&&nlen.y<tmax&&nlen.y<twin1&& flipdir>0.f && n1!=prop.z){
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
                     cphi=fabs(ndir.x);                /*cos(si)*/
                     sphi=ndir.y*ndir.y+ndir.z*ndir.z; /*sin(si)^2*/
                     ndir.x=-ndir.x;
                  }
		  energyabsorbed+=npos.w-npos0.w;
                  npos=npos0;   /*move back*/
                  idx1d=idx1dold;
                  len=1.f-GPUDIV(tmp0,tmp1)*sphi;   /*1-[n1/n2*sin(si)]^2*/
#ifdef __DEVICE_EMULATION__
	          printf(" ref len=%f %f+%f=%f w=%f\n",len,cphi,sphi,cphi*cphi+sphi,npos.w);
#endif
                  if(len>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.z*cphi*sqrtf(len);
                     Rtotal=GPUDIV(ctheta-stheta,ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+GPUDIV(ctheta-stheta,ctheta+stheta))/2.f;
#ifdef __DEVICE_EMULATION__
	          printf("  dir=%f %f %f htime=%f %f %f Rs=%f\n",ndir.x,ndir.y,ndir.z,htime.x,htime.y,htime.z,Rtotal);
	          printf("  ID%d J%d C%d flip=%3f (%d %d) cphi=%f sphi=%f npos=%f %f %f npos0=%f %f %f\n",
                         idx,(int)ndir.w,(int)nlen.w,
	                 flipdir,idx1dold,idx1d,cphi,sphi,npos.x,npos.y,npos.z,npos0.x,npos0.y,npos0.z);
#endif
		     energyloss+=(1.f-Rtotal)*npos.w; /*energy loss due to reflection*/
                     npos.w*=Rtotal;
                  } /* else, total internal reflection, no loss*/
                  mediaid=media[idx1d];
                  prop=gproperty[mediaid];
                  n1=prop.z;
                  ndir.w++;
              }else{
                  energyloss+=npos.w;  // sum all the remaining energy
	          npos=p0;
	          ndir=c0;
	          nlen=float4(0.f,0.f,minaccumtime,nlen.w+1);
                  idx1d=idxorig;
		  mediaid=mediaidorig;
#ifdef CACHE_MEDIA
	          cachebyte=cachebyte0;
	          incache=incache0;
#endif
              }
	  }else if(nlen.y>=nlen.z){
#ifdef __DEVICE_EMULATION__
    printf("field add to %d->%f(%d)  t(%e)>t0(%e)\n",idx1d,npos.w,(int)nlen.w,nlen.y,nlen.z);
#endif
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(save2pt&&nlen.y>=twin0 & nlen.y<twin1){
                  field[idx1d+(int)(floorf((nlen.y-twin0)*Rtstep))*dimlen.z]+=npos.w;
	     }
             nlen.z+=minaccumtime; // fluence is a temporal-integration
	  }
     }
     energyloss+=(npos.w<1.f)?npos.w:0.f;  /*if the last photon has not been terminated, sum energy*/

     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energyabsorbed;
#ifdef USE_MT_RAND
     n_seed[idx]=(ran&0xffffffffu);
#else
     n_seed[idx]=ran*0xffffffffu;
#endif
     n_pos[idx]=npos;
     n_dir[idx]=ndir;
     n_len[idx]=nlen;
}

kernel void mcx_sum_trueabsorption(float energy[],uchar media[], float field[], int maxgate,uint3 dimlen){
     int i;
     float phi=0.f;
     int idx= blockIdx.x*dimlen.y+blockIdx.y*dimlen.x+ threadIdx.x;

     for(i=0;i<maxgate;i++){
        phi+=field[i*dimlen.z+idx];
     }
     energy[2]+=phi*gproperty[media[idx]].x;
}

void mcx_run_simulation(Config *cfg){

     int i,j,iter;
     float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     float4 c0=float4(cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f);
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     float t,twindow0,twindow1;
     float energyloss=0.f,energyabsorbed=0.f;
     float *energy;

     int photoncount=0,printnum;
     int tic,fieldlen;
     uint3 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint3 dimlen;
     //uint3 threaddim;
     float Vvox,scale,absorp,eabsorp;

     dim3 mcgrid, mcblock;
     dim3 clgrid, clblock;
     
     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     
     uchar  *media=(uchar *)(cfg->vol);
     float  *field;
     if(cfg->respin>1){
         field=(float *)malloc(sizeof(float)*dimxyz*cfg->maxgate*2);
         memset(field,0,sizeof(float)*dimxyz*cfg->maxgate*2);
     }
     else{
         field=(float *)malloc(sizeof(float)*dimxyz*cfg->maxgate); /*the second half will be used to accumulate*/
         memset(field,0,sizeof(float)*dimxyz*cfg->maxgate);
     }
	 
#ifdef CACHE_MEDIA
     int count,j,k;
     uchar  mediacache[MAX_MEDIA_CACHE];
#endif

     float4 *Ppos;
     float4 *Pdir;
     float4 *Plen;
     uint   *Pseed;

     if(cfg->nthread%MAX_THREAD)
     	cfg->nthread=(cfg->nthread/MAX_THREAD)*MAX_THREAD;
     mcgrid.x=cfg->nthread/MAX_THREAD;
     mcblock.x=MAX_THREAD;
     
     clgrid.x=cfg->dim.x;
     clgrid.y=cfg->dim.y;
     clblock.x=cfg->dim.z;
	
     Ppos=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Pdir=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Plen=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Pseed=(uint*)malloc(sizeof(uint)*cfg->nthread*RAND_BUF_LEN);
     energy=(float*)malloc(sizeof(float)*cfg->nthread*2);

#ifdef CACHE_MEDIA
     printf("requested constant memory cache: %d (max allowed %d)\n",
         (cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1),(MAX_MEDIA_CACHE<<MEDIA_PACK));

     if((cp1.x-cp0.x+1)*(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1)> (MAX_MEDIA_CACHE<<MEDIA_PACK)){
	mcx_error(-9,"the requested cache size is too big\n");
     }
#endif

     uchar *gmedia;
     cudaMalloc((void **) &gmedia, sizeof(uchar)*(dimxyz));
     float *gfield;
     cudaMalloc((void **) &gfield, sizeof(float)*(dimxyz)*cfg->maxgate);

     float4 *gPpos;
     cudaMalloc((void **) &gPpos, sizeof(float4)*cfg->nthread);
     float4 *gPdir;
     cudaMalloc((void **) &gPdir, sizeof(float4)*cfg->nthread);
     float4 *gPlen;
     cudaMalloc((void **) &gPlen, sizeof(float4)*cfg->nthread);
     uint   *gPseed;
     cudaMalloc((void **) &gPseed, sizeof(uint)*cfg->nthread*RAND_BUF_LEN);

     float *genergy;
     cudaMalloc((void **) &genergy, sizeof(float)*cfg->nthread*2);
     
     if(cfg->isrowmajor){ // if the volume is stored in C array order
	     cachebox.x=(cp1.z-cp0.z+1);
	     cachebox.y=(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1);
	     dimlen.x=cfg->dim.z;
	     dimlen.y=cfg->dim.y*cfg->dim.z;
     }else{               // if the volume is stored in matlab/fortran array order
	     cachebox.x=(cp1.x-cp0.x+1);
	     cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
	     dimlen.x=cfg->dim.x;
	     dimlen.y=cfg->dim.y*cfg->dim.x;
     }
     dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     /*
      threaddim.x=cfg->dim.z;
      threaddim.y=cfg->dim.y*cfg->dim.z;
      threaddim.z=dimlen.z;
     */
     Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z;

#ifdef CACHE_MEDIA
     count=0;
     memset(mediacache,0,MAX_MEDIA_CACHE);

     /*only use 1-byte to store media info, unpacking bits on-the-fly turned out to be expensive in gpu*/

     if(cfg->isrowmajor) {
       for (i=cp0.x; i<=cp1.x; i++)
	for (j=cp0.y; j<=cp1.y; j++)
	 for (k=cp0.z; k<=cp1.z; k++) {
  //         printf("[%d %d %d]: %d %d %d %d (%d)\n",i,j,k,count,MEDIA_MASK,count>>MEDIA_PACK,(count & MEDIA_MOD)*MEDIA_BITS,
  //                (media[INDXYZ(i,j,k)] & MEDIA_MASK )<<((count & MEDIA_MOD)*MEDIA_BITS) );
             mediacache[count>>MEDIA_PACK] |=  (media[i*dimlen.y+j*dimlen.x+k] & MEDIA_MASK )<<((count & MEDIA_MOD)*MEDIA_BITS );
             count++;
	 }
     }else{
       for (i=cp0.x; i<=cp1.x; i++)
	for (j=cp0.y; j<=cp1.y; j++)
	 for (k=cp0.z; k<=cp1.z; k++) {
             mediacache[count>>MEDIA_PACK] |=  (media[k*dimlen.y+j*dimlen.x+i] & MEDIA_MASK )<<((count & MEDIA_MOD)*MEDIA_BITS );
             count++;
	 }
     }
#endif
     if(cfg->seed>0)
     	srand(cfg->seed);
     else
        srand(time(0));
	
     for (i=0; i<cfg->nthread; i++) {
	   Ppos[i]=p0;  /* initial position */
           Pdir[i]=c0;
           Plen[i]=float4(0.f,0.f,minstep*R_C0,0.f);
     }
     for (i=0; i<cfg->nthread*RAND_BUF_LEN; i++) {
	   Pseed[i]=rand();
     }
     tic=GetTimeMillis();

     printf("initializing streams ...\t");
     fieldlen=dimxyz*cfg->maxgate;

     cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPlen,  Plen,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPseed, Pseed, sizeof(uint)  *cfg->nthread*RAND_BUF_LEN,  cudaMemcpyHostToDevice);
     cudaMemcpy(gfield, field, sizeof(float) *fieldlen, cudaMemcpyHostToDevice);
     cudaMemcpy(gmedia, media, sizeof(uchar) *dimxyz, cudaMemcpyHostToDevice);
     cudaMemcpy(genergy,energy,sizeof(float) *cfg->nthread*2, cudaMemcpyHostToDevice);

     cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice);
#ifdef CACHE_MEDIA
     cudaMemcpyToSymbol(gmediacache, mediacache, MAX_MEDIA_CACHE, 0, cudaMemcpyHostToDevice);
#endif

     printf("init complete : %d ms\n",GetTimeMillis()-tic);

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
     
     /*simulate for all time-gates in maxgate groups per run*/
     for(t=cfg->tstart;t<cfg->tend;t+=cfg->tstep*cfg->maxgate){
       twindow0=t;
       twindow1=t+cfg->tstep*cfg->maxgate;

       printf("lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n",twindow0*1e9,twindow1*1e9);

       /*total number of repetition for the simulations, results will be accumulated to field*/
       for(iter=0;iter<cfg->respin;iter++){

           printf("simulation run #%d ... \t",iter+1);
           mcx_main_loop<<<mcgrid,mcblock>>>(cfg->totalmove,gmedia,gfield,genergy,cfg->steps,minstep,\
	        	 twindow0,twindow1,cfg->tend,dimlen,cfg->isrowmajor,cfg->issave2pt,\
                	 1.f/cfg->tstep,p0,c0,maxidx,cp0,cp1,cachebox,cfg->isreflect,gPseed,gPpos,gPdir,gPlen);

	   /*handling the 2pt distributions*/
           if(cfg->issave2pt){
               cudaMemcpy(field, gfield,sizeof(float),cudaMemcpyDeviceToHost);
               printf("kernel complete: %d ms\nretrieving fields ... \t",GetTimeMillis()-tic);
               cudaMemcpy(field, gfield,sizeof(float) *dimxyz*cfg->maxgate,cudaMemcpyDeviceToHost);
               printf("transfer complete: %d ms\n",GetTimeMillis()-tic);

               if(cfg->respin>1){
                   for(i=0;i<fieldlen;i++)  /*accumulate field, can be done in the GPU*/
                      field[fieldlen+i]+=field[i];
               }
               if(iter+1==cfg->respin){ 
                   if(cfg->respin>1)  /*copy the accumulated fields back*/
                       memcpy(field,field+fieldlen,sizeof(float)*fieldlen);

                   if(cfg->isnormalized){
                       /*normalize field if it is the last iteration, temporarily do it in CPU*/
                       //mcx_sum_trueabsorption<<<clgrid,clblock>>>(genergy,gmedia,gfield,
                       //  	cfg->maxgate,threaddim);

                       printf("normizing raw data ...\t");

                       cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*2,cudaMemcpyDeviceToHost);
                       for(i=1;i<cfg->nthread;i++){
                           energy[0]+=energy[i<<1];
       	       	       	   energy[1]+=energy[(i<<1)+1];
                       }
                       eabsorp=0.f;
       	       	       for(i=0;i<dimxyz;i++){
                           absorp=0.f;
                           for(j=0;j<cfg->maxgate;j++)
                              absorp+=field[j*dimxyz+i];
                           eabsorp+=absorp*cfg->prop[media[i]].mua;
       	       	       }
                       scale=energy[1]/(energy[0]+energy[1])/Vvox/cfg->tstep/eabsorp;
                       printf("normalization factor alpha=%f\n",scale);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   printf("data normalization complete : %d ms\n",GetTimeMillis()-tic);

                   printf("saving data to file ...\t");
                   mcx_savedata(field,fieldlen,cfg);
                   printf("saving data complete : %d ms\n",GetTimeMillis()-tic);
               }
           }
	   /*initialize the next simulation*/
	   if(twindow1<cfg->tend && iter+1<cfg->respin){
                  cudaMemset(gfield,0,sizeof(float)*fieldlen); /* cost about 1 ms */

 		  cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice); /*following 3 cost about 50 ms*/
		  cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
		  cudaMemcpy(gPlen,  Plen,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
	   }
	   if(cfg->respin>1 && RAND_BUF_LEN>1){
               for (i=0; i<cfg->nthread*RAND_BUF_LEN; i++)
		   Pseed[i]=rand();
	       cudaMemcpy(gPseed, Pseed, sizeof(uint)*cfg->nthread*RAND_BUF_LEN,  cudaMemcpyHostToDevice);
	   }
       }
       if(twindow1<cfg->tend){
            cudaMemset(genergy,0,sizeof(float)*cfg->nthread*2);
       }
     }

     cudaMemcpy(Ppos,  gPpos, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pdir,  gPdir, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Plen,  gPlen, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pseed, gPseed,sizeof(uint)  *cfg->nthread*RAND_BUF_LEN,   cudaMemcpyDeviceToHost);
     cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*2,cudaMemcpyDeviceToHost);

     for (i=0; i<cfg->nthread; i++) {
	  photoncount+=(int)Plen[i].w;
          energyloss+=energy[i<<1];
          energyabsorbed+=energy[(i<<1)+1];
     }

     printnum=cfg->nthread<16?cfg->nthread:16;
//     printnum=cfg->nthread;
     for (i=0; i<printnum; i++) {
           printf("% 4d[A% f % f % f]C%3d J%5d% 8f(P%6.3f %6.3f %6.3f)T% 5.3f L% 5.3f %f %f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i], 
            Pdir[i].x*Pdir[i].x+Pdir[i].y*Pdir[i].y+Pdir[i].z*Pdir[i].z);
     }
     // total energy here equals total simulated photons+unfinished photons for all threads
     printf("simulated %d photons, exit energy:%16.8e + absorbed energy:%16.8e = total: %16.8e\n",
            photoncount,energyloss,energyabsorbed,energyloss+energyabsorbed);

     cudaFree(gmedia);
#ifdef CACHE_MEDIA
     cudaFree(gmediacache);
#endif
     cudaFree(gfield);
     cudaFree(gPpos);
     cudaFree(gPdir);
     cudaFree(gPlen);
     cudaFree(gPseed);
     cudaFree(genergy);

     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Pseed);
}
