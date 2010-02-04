////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
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

#define MIN(a,b)           ((a)<(b)?(a):(b))

#ifdef __DEVICE_EMULATION__
#define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
#else
#define GPUDEBUG(x)
#endif

typedef unsigned char uchar;

/*
the following is the definition for a photon state:

typedef struct PhotonData {
  float4 pos;  //{x,y,z}: x,y,z coordinates,{w}:packet weight
  float4 dir;  //{x,y,z}: ix,iy,iz unitary direction vector,{w}:total scat event
  float4 len;  //{x}:remaining probability
               //{y}:accumulative time (on-photon timer)
               //{z}:next accum. time (accum. happens at a fixed time interval)
               //{w}:total completed photons
  uint   seed; // random seed, length may different for different RNGs
} Photon;

*/

struct  __align__(16) KernelParams {
  float3 vsize;
  float  minstep;
  float  twin0,twin1,tmax;
  uchar  isrowmajor,save2pt,doreflect,doreflect3;
  float  Rstep;
  float4 p0,c0;
  float3 maxidx;
  uint3  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minenergy;
  float  minaccumtime;
};

// optical properties saved in the constant memory
// {x}:mua,{y}:mus,{z}:anisotropy (g),{w}:refraction index (n)
__constant__ float4 gproperty[MAX_PROP];

// kernel parameters
__constant__ KernelParams gparam;

// tested with texture memory for media, only improved 1% speed
// to keep code portable, use global memory for now
// also need also change all media[idx1d] to tex1Dfetch() below
//texture<uchar, 1, cudaReadModeElementType> texmedia;

#ifdef USE_ATOMIC
/*
 float-atomic-add from:
 http://forums.nvidia.com/index.php?showtopic=67691&st=0&p=380935&#entry380935
 this makes the program non-scalable and about 5 times slower compared 
 with non-atomic write, see Fig. 4 and 7 in Fang2009
*/
__device__ inline void atomicFloatAdd(float *address, float val){
      int i_val = __float_as_int(val);
      int tmp0 = 0,tmp1;
      while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0){
              tmp0 = tmp1;
              i_val = __float_as_int(val + __int_as_float(tmp1));
      }
}
#endif

//need to move these arguments to the constant memory, as they use shared memory

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
kernel void mcx_main_loop(int nphoton,int ophoton,uchar media[],float field[],float genergy[],float3 vsize,float minstep, 
     float twin0,float twin1, float tmax, uint3 dimlen, uchar isrowmajor, uchar save2pt, float Rtstep,
     float4 p0,float4 c0,float3 maxidx,uint3 cp0,uint3 cp1,uint2 cachebox,uchar doreflect,uchar doreflect3, 
     float minenergy, float sradius2, uint n_seed[],float4 n_pos[],float4 n_dir[],float4 n_len[]){

     int idx= blockDim.x * blockIdx.x + threadIdx.x;

     float4 npos=n_pos[idx];  //{x,y,z}: x,y,z coordinates,{w}:packet weight
     float4 ndir=n_dir[idx];  //{x,y,z}: ix,iy,iz unitary direction vector, {w}:total scat event
                              //ndir.w can be dropped to save register
     float4 nlen=n_len[idx];  //nlen.w can be dropped to save register
     float4 npos0;            //reflection var, to save pre-reflection npos state
     float3 htime;            //reflection var
     float  minaccumtime=minstep*R_C0;   //can be moved to constant memory
     float  energyloss=genergy[idx<<1];
     float  energyabsorbed=genergy[(idx<<1)+1];

     int i,idx1d, idx1dold,idxorig;   //idx1dold is related to reflection
     //int np=nphoton+((idx==blockDim.x*gridDim.x-1) ? ophoton: 0);

#ifdef TEST_RACING
     int cc=0;
#endif
     uchar  mediaid, mediaidorig;
     char   medid=-1;
     float  atten;         //can be taken out to minimize registers
     float  flipdir,n1,Rtotal;   //reflection var

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN],tnew[RAND_BUF_LEN];
     float4 prop;    //can become float2 if no reflection

     float len,cphi,sphi,theta,stheta,ctheta,tmp0,tmp1;
     float accumweight=0.f;

     gpu_rng_init(t,tnew,n_seed,idx);

     // assuming the initial position is within the domain (mcx_config is supposed to ensure)
     idx1d=isrowmajor?int(floorf(npos.x)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.z)):\
                      int(floorf(npos.z)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.x));
     idxorig=idx1d;
     mediaid=media[idx1d];
     mediaidorig=mediaid;
	  
     if(mediaid==0) {
          return; // the initial position is not within the medium
     }
     prop=gproperty[mediaid];

     /*
      using a while-loop to terminate a thread by np will cause MT RNG to be 3.5x slower
      LL5 RNG will only be slightly slower than for-loop with photon-move criterion
     */
     //while(nlen.w<np) {

     for(i=0;i<nphoton;i++){ // here nphoton actually means photon moves

          GPUDEBUG(("*i= (%d) L=%f w=%e a=%f\n",(int)nlen.w,nlen.x,npos.w,nlen.y));
	  if(nlen.x<=0.f) {  // if this photon has finished the current jump
               rand_need_more(t,tnew);
   	       nlen.x=rand_next_scatlen(t);

               GPUDEBUG(("next scat len=%20.16e \n",nlen.x));
	       if(npos.w<1.f){ //weight
                       //random arimuthal angle
                       tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                       sincosf(tmp0,&sphi,&cphi);
                       GPUDEBUG(("next angle phi %20.16e\n",tmp0));

                       //Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234
                       //see Boas2002

                       if(prop.w>EPS){  //if prop.w is too small, the distribution of theta is bad
		           tmp0=(1.f-prop.w*prop.w)/(1.f-prop.w+2.f*prop.w*rand_next_zangle(t));
		           tmp0*=tmp0;
		           tmp0=(1.f+prop.w*prop.w-tmp0)/(2.f*prop.w);

                           // when ran=1, CUDA will give me 1.000002 for tmp0 which produces nan later
                           if(tmp0>1.f) tmp0=1.f;
                           // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                           if(tmp0<-1.f) tmp0=-1.f;

		           theta=acosf(tmp0);
		           stheta=sinf(theta);
		           ctheta=tmp0;
                       }else{  //Wang1995 has acos(2*ran-1), rather than 2*pi*ran, need to check
			   theta=ONE_PI*rand_next_zangle(t);
                           sincosf(theta,&stheta,&ctheta);
                       }
                       GPUDEBUG(("next scat angle theta %20.16e\n",theta));

		       if( ndir.z>-1.f+EPS && ndir.z<1.f-EPS ) {
		           tmp0=1.f-ndir.z*ndir.z;   //reuse tmp to minimize registers
		           tmp1=rsqrtf(tmp0);
		           tmp1=stheta*tmp1;
		           ndir=float4(
				tmp1*(ndir.x*ndir.z*cphi - ndir.y*sphi) + ndir.x*ctheta,
				tmp1*(ndir.y*ndir.z*cphi + ndir.x*sphi) + ndir.y*ctheta,
				-tmp1*tmp0*cphi                         + ndir.z*ctheta,
				ndir.w
			   );
                           GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z));
		       }else{
			   ndir=float4(stheta*cphi,stheta*sphi,(ndir.z>0.f)?ctheta:-ctheta,ndir.w);
                           GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z));
 		       }
                       ndir.w++;
	       }
	  }

          n1=prop.z;
	  prop=gproperty[mediaid];
	  len=minstep*prop.y; //Wang1995: minstep*(prop.x+prop.y)

          npos0=npos;
	  if(len>nlen.x){  //scattering ends in this voxel: mus*minstep > s 
               tmp0=nlen.x/prop.y;
	       energyabsorbed+=npos.w;
   	       npos=float4(npos.x+ndir.x*tmp0,npos.y+ndir.y*tmp0,npos.z+ndir.z*tmp0,
                           npos.w*expf(-prop.x*tmp0));
	       energyabsorbed-=npos.w;
	       nlen.x=SAME_VOXEL;
	       nlen.y+=tmp0*prop.z*R_C0;  // accumulative time
               GPUDEBUG((">>ends in voxel %f<%f %f [%d]\n",nlen.x,len,prop.y,idx1d));
	  }else{                      //otherwise, move minstep
	       energyabsorbed+=npos.w;
               if(mediaid!=medid){
                  atten=expf(-prop.x*minstep);
               }
   	       npos=float4(npos.x+ndir.x,npos.y+ndir.y,npos.z+ndir.z,npos.w*atten);
               medid=mediaid;
	       energyabsorbed-=npos.w;
	       nlen.x-=len;     //remaining probability: sum(s_i*mus_i)
	       nlen.y+=minaccumtime*prop.z; //total time
               GPUDEBUG((">>keep going %f<%f %f [%d] %e %e\n",nlen.x,len,prop.y,idx1d,nlen.y,nlen.z));
	  }

          idx1dold=idx1d;
          idx1d=isrowmajor?int(floorf(npos.x)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.z)):\
                           int(floorf(npos.z)*dimlen.y+floorf(npos.y)*dimlen.x+floorf(npos.x));
          GPUDEBUG(("old and new voxel: %d<->%d\n",idx1dold,idx1d));
          if(npos.x<0||npos.y<0||npos.z<0||npos.x>=maxidx.x||npos.y>=maxidx.y||npos.z>=maxidx.z){
	      mediaid=0;
	  }else{
              mediaid=media[idx1d];
          }

          //if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one
	  if(mediaid==0||nlen.y>tmax||nlen.y>twin1){
              flipdir=0.f;
              if(doreflect) {
                //time-of-flight to hit the wall in each direction
                htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floorf(npos0.x)+(ndir.x>0.f)-npos0.x)/ndir.x:VERY_BIG;
                htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floorf(npos0.y)+(ndir.y>0.f)-npos0.y)/ndir.y:VERY_BIG;
                htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floorf(npos0.z)+(ndir.z>0.f)-npos0.z)/ndir.z:VERY_BIG;
                //get the direction with the smallest time-of-flight
                tmp0=fminf(fminf(htime.x,htime.y),htime.z);
                flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));

                //move to the 1st intersection pt
                tmp0*=JUST_ABOVE_ONE;
                htime.x=floorf(npos0.x+tmp0*ndir.x);
       	        htime.y=floorf(npos0.y+tmp0*ndir.y);
       	        htime.z=floorf(npos0.z+tmp0*ndir.z);

                if(htime.x>=0&&htime.y>=0&&htime.z>=0&&htime.x<maxidx.x&&htime.y<maxidx.y&&htime.z<maxidx.z){
                    if( media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)]){ //hit again

                     GPUDEBUG((" first try failed: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime.x,htime.y,htime.z,
                           media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)], maxidx.x, maxidx.y,maxidx.z));

                     htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floorf(npos.x)+(ndir.x<0.f)-npos.x)/(-ndir.x):VERY_BIG;
                     htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floorf(npos.y)+(ndir.y<0.f)-npos.y)/(-ndir.y):VERY_BIG;
                     htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floorf(npos.z)+(ndir.z<0.f)-npos.z)/(-ndir.z):VERY_BIG;
                     tmp0=fminf(fminf(htime.x,htime.y),htime.z);
                     tmp1=flipdir;   //save the previous ref. interface id
                     flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));

                     if(doreflect3){
                       tmp0*=JUST_ABOVE_ONE;
                       htime.x=floorf(npos.x-tmp0*ndir.x); //move to the last intersection pt
                       htime.y=floorf(npos.y-tmp0*ndir.y);
                       htime.z=floorf(npos.z-tmp0*ndir.z);

                       if(tmp1!=flipdir&&htime.x>=0&&htime.y>=0&&htime.z>=0&&htime.x<maxidx.x&&htime.y<maxidx.y&&htime.z<maxidx.z){
                           if(! media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                                  int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)]){ //this is an air voxel

                               GPUDEBUG((" second try failed: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime.x,htime.y,htime.z,
                                   media[isrowmajor?int(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                                   int(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)], maxidx.x, maxidx.y,maxidx.z));

                               /*to compute the remaining interface, we used the following fact to accelerate: 
                                 if there exist 3 intersections, photon must pass x/y/z interface exactly once,
                                 we solve the coeff of the following equation to find the last interface:
                                    a*1+b*2+c=3
       	       	       	       	    a*1+b*3+c=2 -> [a b c]=[-1 -1 6], this will give the remaining interface id
       	       	       	       	    a*2+b*3+c=1
                               */
                               flipdir=-tmp1-flipdir+6.f;
                           }
                       }
                     }
                  }
                }
              }

              prop=gproperty[mediaid];

              GPUDEBUG(("->ID%d J%d C%d tlen %e flip %d %.1f!=%.1f dir=%f %f %f pos=%f %f %f\n",idx,(int)ndir.w,
                  (int)nlen.w,nlen.y, (int)flipdir, n1,prop.z,ndir.x,ndir.y,ndir.z,npos.x,npos.y,npos.z));

              //recycled some old register variables to save memory
	      //if hit boundary within the time window and is n-mismatched, rebound

              if(doreflect&&nlen.y<tmax&&nlen.y<twin1&& flipdir>0.f && n1!=prop.z&&npos.w>minenergy){
                  tmp0=n1*n1;
                  tmp1=prop.z*prop.z;
                  if(flipdir>=3.f) { //flip in z axis
                     cphi=fabs(ndir.z);
                     sphi=ndir.x*ndir.x+ndir.y*ndir.y;
                     ndir.z=-ndir.z;
                  }else if(flipdir>=2.f){ //flip in y axis
                     cphi=fabs(ndir.y);
       	       	     sphi=ndir.x*ndir.x+ndir.z*ndir.z;
                     ndir.y=-ndir.y;
                  }else if(flipdir>=1.f){ //flip in x axis
                     cphi=fabs(ndir.x);                //cos(si)
                     sphi=ndir.y*ndir.y+ndir.z*ndir.z; //sin(si)^2
                     ndir.x=-ndir.x;
                  }
		  energyabsorbed+=npos.w-npos0.w;
                  npos=npos0;   //move back
                  idx1d=idx1dold;
                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2
	          GPUDEBUG((" ref len=%f %f+%f=%f w=%f\n",len,cphi,sphi,cphi*cphi+sphi,npos.w));

                  if(len>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.z*cphi*sqrtf(len);
                     Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
	             GPUDEBUG(("  dir=%f %f %f htime=%f %f %f Rs=%f\n",ndir.x,ndir.y,ndir.z,htime.x,htime.y,htime.z,Rtotal));
	             GPUDEBUG(("  ID%d J%d C%d flip=%3f (%d %d) cphi=%f sphi=%f npos=%f %f %f npos0=%f %f %f\n",
                         idx,(int)ndir.w,(int)nlen.w,
	                 flipdir,idx1dold,idx1d,cphi,sphi,npos.x,npos.y,npos.z,npos0.x,npos0.y,npos0.z));
		     energyloss+=(1.f-Rtotal)*npos.w; //energy loss due to reflection
                     npos.w*=Rtotal;
                  } // else, total internal reflection, no loss
                  mediaid=media[idx1d];
                  prop=gproperty[mediaid];
                  n1=prop.z;
                  //ndir.w++;
              }else{  // launch a new photon
                  energyloss+=npos.w;  // sum all the remaining energy
	          npos=p0;
	          ndir=c0;
	          nlen=float4(0.f,0.f,minaccumtime,nlen.w+1);
                  idx1d=idxorig;
		  mediaid=mediaidorig;
              }
	  }else if(nlen.y>=nlen.z){
             GPUDEBUG(("field add to %d->%f(%d)  t(%e)>t0(%e)\n",idx1d,npos.w,(int)nlen.w,nlen.y,nlen.z));
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(save2pt&&nlen.y>=twin0 & nlen.y<twin1){
#ifdef TEST_RACING
                  // enable TEST_RACING to determine how many missing accumulations due to race
                  if( (npos.x-p0.x)*(npos.x-p0.x)+(npos.y-p0.y)*(npos.y-p0.y)+(npos.z-p0.z)*(npos.z-p0.z)>sradius2) {
                      field[idx1d+(int)(floorf((nlen.y-twin0)*Rtstep))*dimlen.z]+=1.f;
		      cc++;
                  }
#else
  #ifndef USE_ATOMIC
                  // set sradius2 to only start depositing energy when dist^2>sradius2 
                  if(sradius2>EPS){
                      if((npos.x-p0.x)*(npos.x-p0.x)+(npos.y-p0.y)*(npos.y-p0.y)+(npos.z-p0.z)*(npos.z-p0.z)>sradius2){
                          field[idx1d+(int)(floorf((nlen.y-twin0)*Rtstep))*dimlen.z]+=npos.w;
                      }else{
                          accumweight+=npos.w*prop.x; // weight*absorption
                      }
                  }else{
                      field[idx1d+(int)(floorf((nlen.y-twin0)*Rtstep))*dimlen.z]+=npos.w;
                  }
  #else
                  // ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
		  atomicFloatAdd(& field[idx1d+(int)(floorf((nlen.y-twin0)*Rtstep))*dimlen.z], npos.w);
  #endif
#endif
	     }
             nlen.z+=minaccumtime; // fluence is a temporal-integration
	  }
     }
     // accumweight saves the total absorbed energy in the sphere r<sradius.
     // in non-atomic mode, accumweight is more accurate than saving to the grid
     // as it is not influenced by race conditions.
     // now I borrow nlen.z to pass this value back

     nlen.z=accumweight;

     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energyabsorbed;

#ifdef TEST_RACING
     n_seed[idx]=cc;
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

/*
  query GPU info and set active GPU
*/
int mcx_set_gpu(int printinfo){

#if __DEVICE_EMULATION__
    return 1;
#else
    int dev;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        printf("No CUDA-capable GPU device found\n");
        return 0;
    }
    // scan from the last device, hopefully it is more dedicated
    for (dev = deviceCount-1; dev>=0; dev--) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, dev);
        if (strncmp(dp.name, "Device Emulation", 16)) {
	  if(printinfo){
	    printf("=============================   GPU Infomation  ================================\n");
	    printf("Device %d of %d:\t\t%s\n",dev+1,deviceCount,dp.name);
	    printf("Global Memory:\t\t%u B\nConstant Memory:\t%u B\n\
Shared Memory:\t\t%u B\nRegisters:\t\t%u\nClock Speed:\t\t%.2f GHz\n",
               dp.totalGlobalMem,dp.totalConstMem,
               dp.sharedMemPerBlock,dp.regsPerBlock,dp.clockRate*1e-6f);
	  #if CUDART_VERSION >= 2000
	       printf("Number of MPs:\t\t%u\nNumber of Cores:\t%u\n",
	          dp.multiProcessorCount,dp.multiProcessorCount<<3);
	  #endif
	  }
          if(printinfo!=2) break;
	}
    }
    if(printinfo==2){ //list GPU info only
          exit(0);
    }
    if (dev == deviceCount)
        return 0;
    else {
        cudaSetDevice(dev);
        return 1;
    }
#endif
}

/*
   assert cuda memory allocation result
*/
void mcx_cu_assess(cudaError_t cuerr){
     if(cuerr!=cudaSuccess){
         mcx_error(-(int)cuerr,(char *)cudaGetErrorString(cuerr));
     }
}

/*
   master driver code to run MC simulations
*/
void mcx_run_simulation(Config *cfg){

     int i,j,iter;
     float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     float4 p0=float4(cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f);
     float4 c0=float4(cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f);
     float3 maxidx=float3(cfg->dim.x,cfg->dim.y,cfg->dim.z);
     float t,twindow0,twindow1;
     float energyloss=0.f,energyabsorbed=0.f;
     float *energy;
     int threadphoton, oddphotons;

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
         field=(float *)calloc(sizeof(float)*dimxyz,cfg->maxgate*2);
     }else{
         field=(float *)calloc(sizeof(float)*dimxyz,cfg->maxgate); //the second half will be used to accumulate
     }
     threadphoton=cfg->nphoton/cfg->nthread/cfg->respin;
     oddphotons=cfg->nphoton-threadphoton*cfg->nthread*cfg->respin;

     float4 *Ppos;
     float4 *Pdir;
     float4 *Plen;
     uint   *Pseed;

     if(cfg->nthread%cfg->nblocksize)
     	cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;
     mcgrid.x=cfg->nthread/cfg->nblocksize;
     mcblock.x=cfg->nblocksize;

     clgrid.x=cfg->dim.x;
     clgrid.y=cfg->dim.y;
     clblock.x=cfg->dim.z;
	
     Ppos=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Pdir=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Plen=(float4*)malloc(sizeof(float4)*cfg->nthread);
     Pseed=(uint*)malloc(sizeof(uint)*cfg->nthread*RAND_SEED_LEN);
     energy=(float*)calloc(sizeof(float),cfg->nthread*2);

     uchar *gmedia;
     mcx_cu_assess(cudaMalloc((void **) &gmedia, sizeof(uchar)*(dimxyz)));
     float *gfield;
     mcx_cu_assess(cudaMalloc((void **) &gfield, sizeof(float)*(dimxyz)*cfg->maxgate));

     //cudaBindTexture(0, texmedia, gmedia);

     float4 *gPpos;
     mcx_cu_assess(cudaMalloc((void **) &gPpos, sizeof(float4)*cfg->nthread));
     float4 *gPdir;
     mcx_cu_assess(cudaMalloc((void **) &gPdir, sizeof(float4)*cfg->nthread));
     float4 *gPlen;
     mcx_cu_assess(cudaMalloc((void **) &gPlen, sizeof(float4)*cfg->nthread));
     uint   *gPseed;
     mcx_cu_assess(cudaMalloc((void **) &gPseed, sizeof(uint)*cfg->nthread*RAND_SEED_LEN));

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

     if(cfg->seed>0)
     	srand(cfg->seed);
     else
        srand(time(0));
	
     for (i=0; i<cfg->nthread; i++) {
	   Ppos[i]=p0;  // initial position
           Pdir[i]=c0;
           Plen[i]=float4(0.f,0.f,minstep*R_C0,0.f);
     }
     for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++) {
	   Pseed[i]=rand();
     }
     fprintf(cfg->flog,"\
###############################################################################\n\
#                  Monte Carlo Extreme (MCX) -- CUDA                          #\n\
###############################################################################\n\
$MCX $Rev::     $ Last Commit:$Date::                     $ by $Author:: fangq$\n\
###############################################################################\n");

     tic=StartTimer();
     fprintf(cfg->flog,"compiled with: [RNG] %s [Seed Length] %d\n",MCX_RNG_NAME,RAND_SEED_LEN);
     fprintf(cfg->flog,"threadph=%d oddphotons=%d np=%d nthread=%d repetition=%d\n",threadphoton,oddphotons,
           cfg->nphoton,cfg->nthread,cfg->respin);
     fprintf(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);
     fieldlen=dimxyz*cfg->maxgate;

     cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPlen,  Plen,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
     cudaMemcpy(gPseed, Pseed, sizeof(uint)  *cfg->nthread*RAND_SEED_LEN,  cudaMemcpyHostToDevice);
     cudaMemcpy(gfield, field, sizeof(float) *fieldlen, cudaMemcpyHostToDevice);
     cudaMemcpy(gmedia, media, sizeof(uchar) *dimxyz, cudaMemcpyHostToDevice);
     cudaMemcpy(genergy,energy,sizeof(float) *cfg->nthread*2, cudaMemcpyHostToDevice);

     cudaMemcpyToSymbol(gproperty, cfg->prop,  cfg->medianum*sizeof(Medium), 0, cudaMemcpyHostToDevice);
     fprintf(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

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
     
     //simulate for all time-gates in maxgate groups per run
     for(t=cfg->tstart;t<cfg->tend;t+=cfg->tstep*cfg->maxgate){
       twindow0=t;
       twindow1=t+cfg->tstep*cfg->maxgate;

       fprintf(cfg->flog,"lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n"
           ,twindow0*1e9,twindow1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<cfg->respin;iter++){

           fprintf(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);
           mcx_main_loop<<<mcgrid,mcblock>>>(cfg->nphoton,0,gmedia,gfield,genergy,cfg->steps,minstep,\
	        	 twindow0,twindow1,cfg->tend,dimlen,cfg->isrowmajor,cfg->issave2pt,\
                	 1.f/cfg->tstep,p0,c0,maxidx,cp0,cp1,cachebox,cfg->isreflect,cfg->isref3,cfg->minenergy,\
                         cfg->sradius*cfg->sradius,gPseed,gPpos,gPdir,gPlen);

	   //handling the 2pt distributions
           if(cfg->issave2pt){
               cudaThreadSynchronize();
               cudaMemcpy(field, gfield,sizeof(float),cudaMemcpyDeviceToHost);
               fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",GetTimeMillis()-tic);
               cudaMemcpy(field, gfield,sizeof(float) *dimxyz*cfg->maxgate,cudaMemcpyDeviceToHost);
               fprintf(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

               mcx_cu_assess(cudaGetLastError());

               if(cfg->respin>1){
                   for(i=0;i<fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
               if(iter+1==cfg->respin){ 
                   if(cfg->respin>1)  //copy the accumulated fields back
                       memcpy(field,field+fieldlen,sizeof(float)*fieldlen);

                   if(cfg->isnormalized){
                       //normalize field if it is the last iteration, temporarily do it in CPU
                       //mcx_sum_trueabsorption<<<clgrid,clblock>>>(genergy,gmedia,gfield,
                       //  	cfg->maxgate,threaddim);

                       fprintf(cfg->flog,"normizing raw data ...\t");

                       cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*2,cudaMemcpyDeviceToHost);
		       cudaMemcpy(Plen,  gPlen,  sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
                       eabsorp=0.f;
                       for(i=1;i<cfg->nthread;i++){
                           energy[0]+=energy[i<<1];
       	       	       	   energy[1]+=energy[(i<<1)+1];
                           eabsorp+=Plen[i].z;  // the accumulative absorpted energy near the source
                       }
       	       	       for(i=0;i<dimxyz;i++){
                           absorp=0.f;
                           for(j=0;j<cfg->maxgate;j++)
                              absorp+=field[j*dimxyz+i];
                           eabsorp+=absorp*cfg->prop[media[i]].mua;
       	       	       }
                       scale=energy[1]/(energy[0]+energy[1])/Vvox/cfg->tstep/eabsorp;
                       fprintf(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   fprintf(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);

                   fprintf(cfg->flog,"saving data to file ...\t");
                   mcx_savedata(field,fieldlen,cfg);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
                   fflush(cfg->flog);
               }
           }
	   //initialize the next simulation
	   if(twindow1<cfg->tend && iter+1<cfg->respin){
                  cudaMemset(gfield,0,sizeof(float)*fieldlen); // cost about 1 ms

 		  cudaMemcpy(gPpos,  Ppos,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice); //following 3 cost about 50 ms
		  cudaMemcpy(gPdir,  Pdir,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
		  cudaMemcpy(gPlen,  Plen,  sizeof(float4)*cfg->nthread,  cudaMemcpyHostToDevice);
	   }
	   if(cfg->respin>1 && RAND_SEED_LEN>1){
               for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++)
		   Pseed[i]=rand();
	       cudaMemcpy(gPseed, Pseed, sizeof(uint)*cfg->nthread*RAND_SEED_LEN,  cudaMemcpyHostToDevice);
	   }
       }
       if(twindow1<cfg->tend){
            cudaMemset(genergy,0,sizeof(float)*cfg->nthread*2);
       }
     }

     cudaMemcpy(Ppos,  gPpos, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pdir,  gPdir, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Plen,  gPlen, sizeof(float4)*cfg->nthread, cudaMemcpyDeviceToHost);
     cudaMemcpy(Pseed, gPseed,sizeof(uint)  *cfg->nthread*RAND_SEED_LEN,   cudaMemcpyDeviceToHost);
     cudaMemcpy(energy,genergy,sizeof(float)*cfg->nthread*2,cudaMemcpyDeviceToHost);

     for (i=0; i<cfg->nthread; i++) {
	  photoncount+=(int)Plen[i].w;
          energyloss+=energy[i<<1];
          energyabsorbed+=energy[(i<<1)+1];
     }

#ifdef TEST_RACING
     {
       float totalcount=0.f,hitcount=0.f;
       for (i=0; i<fieldlen; i++)
          hitcount+=field[i];
       for (i=0; i<cfg->nthread; i++)
	  totalcount+=Pseed[i];
     
       fprintf(cfg->flog,"expected total recording number: %f, got %f, missed %f\n",
          totalcount,hitcount,(totalcount-hitcount)/totalcount);
     }
#endif

     printnum=cfg->nthread<cfg->printnum?cfg->nthread:cfg->printnum;
     for (i=0; i<printnum; i++) {
           fprintf(cfg->flog,"% 4d[A% f % f % f]C%3d J%5d W% 8f(P%6.3f %6.3f %6.3f)T% 5.3e L% 5.3f %.0f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i]);
     }
     // total energy here equals total simulated photons+unfinished photons for all threads
     fprintf(cfg->flog,"simulated %d photons (%d) with %d threads (repeat x%d)\n",
             photoncount,cfg->nphoton,cfg->nthread,cfg->respin); fflush(cfg->flog);
     fprintf(cfg->flog,"exit energy:%16.8e + absorbed energy:%16.8e = total: %16.8e\n",
             energyloss,energyabsorbed,energyloss+energyabsorbed);fflush(cfg->flog);
     fflush(cfg->flog);

     cudaFree(gmedia);
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
     free(energy);
     free(field);
}
