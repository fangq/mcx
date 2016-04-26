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
//  mcxlab.cpp: MCX for Matlab and Octave
//
//  License: GNU General Public License v3, see LICENSE.txt for details
//
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <exception>
#include <time.h>
#include <math.h>

#include "mex.h"
#include "mcx_const.h"
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_shapes.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

#define RAND_BUF_LEN 5

#define GET_1ST_FIELD(x,y)  if(strcmp(name,#y)==0) {double *val=mxGetPr(item);x->y=val[0];printf("mcx.%s=%g;\n",#y,(float)(x->y));}
#define GET_ONE_FIELD(x,y)  else GET_1ST_FIELD(x,y)
#define GET_VEC3_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];\
                                 printf("mcx.%s=[%g %g %g];\n",#v,(float)(u->v.x),(float)(u->v.y),(float)(u->v.z));}
#define GET_VEC4_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];u->v.w=val[3];\
                                 printf("mcx.%s=[%g %g %g %g];\n",#v,(float)(u->v.x),(float)(u->v.y),(float)(u->v.z),(float)(u->v.w));}

#define SET_GPU_INFO(output,id,v)  mxSetField(output,id,#v,mxCreateDoubleScalar(gpuinfo[i].v));

#ifdef USE_MT_RAND
    #define RAND_BUF_LEN 0
#else
    #define RAND_BUF_LEN 5
#endif


void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg);
void mcx_validate_config(Config *cfg);
void mcxlab_usage();

float *detps=NULL;
int    dimdetps[2]={0,0};
int    seedbyte=0;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  Config cfg;
  GPUInfo *gpuinfo=NULL;
  mxArray    *tmp;
  int        ifield, jstruct;
  int        ncfg, nfields;
  int        fielddim[4];
  int        activedev=0;
  const char       *outputtag[]={"data"};
  const char       *datastruct[]={"data","stat"};
  const char       *statstruct[]={"runtime","nphoton","energytot","energyabs","normalizer","workload"};
  const char       *gpuinfotag[]={"name","id","devcount","major","minor","globalmem",
                                  "constmem","sharedmem","regcount","clock","sm","core",
                                  "autoblock","autothread","maxgate"};

  if (nrhs==0){
     mcxlab_usage();
     return;
  }
  if(nrhs==1 && mxIsChar(prhs[0])){
        char shortcmd[MAX_SESSION_LENGTH];
        mxGetString(prhs[0], shortcmd, MAX_SESSION_LENGTH);
        shortcmd[MAX_SESSION_LENGTH-1]='\0';
        if(strcmp(shortcmd,"gpuinfo")==0){
            mcx_initcfg(&cfg);
            cfg.isgpuinfo=3;
            if(!(activedev=mcx_list_gpu(&cfg,&gpuinfo))){
                mexWarnMsgTxt("no active GPU device found");
            }
            plhs[0] = mxCreateStructMatrix(gpuinfo[0].devcount,1,15,gpuinfotag);
            for(int i=0;i<gpuinfo[0].devcount;i++){
		mxSetField(plhs[0],i,"name",mxCreateString(gpuinfo[i].name));
		SET_GPU_INFO(plhs[0],i,id);
		SET_GPU_INFO(plhs[0],i,devcount);
		SET_GPU_INFO(plhs[0],i,major);
		SET_GPU_INFO(plhs[0],i,minor);
		SET_GPU_INFO(plhs[0],i,globalmem);
		SET_GPU_INFO(plhs[0],i,constmem);
		SET_GPU_INFO(plhs[0],i,sharedmem);
		SET_GPU_INFO(plhs[0],i,regcount);
		SET_GPU_INFO(plhs[0],i,clock);
		SET_GPU_INFO(plhs[0],i,sm);
		SET_GPU_INFO(plhs[0],i,core);
		SET_GPU_INFO(plhs[0],i,autoblock);
		SET_GPU_INFO(plhs[0],i,autothread);
		SET_GPU_INFO(plhs[0],i,maxgate);
            }
            mcx_cleargpuinfo(&gpuinfo);
            mcx_clearcfg(&cfg);
	}
	return;
  }
  printf("Launching MCXLAB - Monte Carlo eXtreme for MATLAB & GNU Octave ...\n");
  if (!mxIsStruct(prhs[0]))
     mexErrMsgTxt("Input must be a structure.");

  nfields = mxGetNumberOfFields(prhs[0]);
  ncfg = mxGetNumberOfElements(prhs[0]);

  if(nlhs>=1)
      plhs[0] = mxCreateStructMatrix(ncfg,1,2,datastruct);
  if(nlhs>=2)
      plhs[1] = mxCreateStructMatrix(ncfg,1,1,outputtag);
  if(nlhs>=3)
      plhs[2] = mxCreateStructMatrix(ncfg,1,1,outputtag);
  if(nlhs>=4)
      plhs[3] = mxCreateStructMatrix(ncfg,1,1,outputtag);

  for (jstruct = 0; jstruct < ncfg; jstruct++) {  /* how many configs */
    try{
	printf("Running simulations for configuration #%d ...\n", jstruct+1);

	mcx_initcfg(&cfg);

	for (ifield = 0; ifield < nfields; ifield++) { /* how many input struct fields */
            tmp = mxGetFieldByNumber(prhs[0], jstruct, ifield);
	    if (tmp == NULL) {
		    continue;
	    }
	    mcx_set_field(prhs[0],tmp,ifield,&cfg);
	}
#ifndef MATLAB_MEX_FILE
        mexEvalString("fflush(stdout);");
#else
	mexEvalString("drawnow;");
#endif
	cfg.issave2pt=(nlhs>=1);
	cfg.issavedet=(nlhs>=2);
	cfg.issaveseed=(nlhs>=4);
#if defined(USE_MT_RAND)
        cfg.issaveseed=0;
#endif
	if(cfg.vol==NULL || cfg.medianum==0){
	    mexErrMsgTxt("You must define 'vol' and 'prop' field.");
	}
	if(!(activedev=mcx_list_gpu(&cfg,&gpuinfo))){
            mexErrMsgTxt("No active GPU device found");
	}
	if(nlhs>=1){
            int fieldlen=cfg.dim.x*cfg.dim.y*cfg.dim.z*(int)((cfg.tend-cfg.tstart)/cfg.tstep+0.5);
	    cfg.exportfield = (float*)calloc(fieldlen,sizeof(float));
	}
	if(nlhs>=2){
	    cfg.exportdetected=(float*)malloc((cfg.medianum+1)*cfg.maxdetphoton*sizeof(float));
        }
        if(nlhs>=4){
	    cfg.seeddata=malloc(cfg.maxdetphoton*sizeof(float)*RAND_BUF_LEN);
	}
        mcx_validate_config(&cfg);

#ifdef _OPENMP
        omp_set_num_threads(activedev);
#pragma omp parallel
{
#endif

        mcx_run_simulation(&cfg,gpuinfo);

#ifdef _OPENMP
}
#endif

        if(nlhs>=4){
            fielddim[0]=(cfg.issaveseed>0)*RAND_BUF_LEN*sizeof(float); fielddim[1]=cfg.detectedcount; // his.savedphoton is for one repetition, should correct
    	    fielddim[2]=0; fielddim[3]=0;
		    mxSetFieldByNumber(plhs[3],jstruct,0, mxCreateNumericArray(2,fielddim,mxUINT8_CLASS,mxREAL));
		    memcpy((unsigned char*)mxGetPr(mxGetFieldByNumber(plhs[3],jstruct,0)),cfg.seeddata,fielddim[0]*fielddim[1]);
	    free(cfg.seeddata);
            cfg.seeddata=NULL;
	}
	if(nlhs>=3){
            fielddim[0]=cfg.dim.x; fielddim[1]=cfg.dim.y;
            fielddim[2]=cfg.dim.z; fielddim[3]=0;
            if(cfg.vol){
                    mxSetFieldByNumber(plhs[2],jstruct,0, mxCreateNumericArray(3,fielddim,mxUINT8_CLASS,mxREAL));
                    memcpy((unsigned char*)mxGetPr(mxGetFieldByNumber(plhs[2],jstruct,0)),cfg.vol,
                	 fielddim[0]*fielddim[1]*fielddim[2]*sizeof(unsigned char));
            }
	}
	if(nlhs>=2){
            fielddim[0]=(cfg.medianum+1); fielddim[1]=cfg.detectedcount; 
            fielddim[2]=0; fielddim[3]=0;
            if(cfg.detectedcount>0){
                    mxSetFieldByNumber(plhs[1],jstruct,0, mxCreateNumericArray(2,fielddim,mxSINGLE_CLASS,mxREAL));
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[1],jstruct,0)),cfg.exportdetected,
                         fielddim[0]*fielddim[1]*sizeof(float));
            }
            free(cfg.exportdetected);
            cfg.exportdetected=NULL;
	}
        if(nlhs>=1){
            fielddim[0]=cfg.dim.x; fielddim[1]=cfg.dim.y; 
	    fielddim[2]=cfg.dim.z; fielddim[3]=(int)((cfg.tend-cfg.tstart)/cfg.tstep+0.5);
	    mxSetFieldByNumber(plhs[0],jstruct,0, mxCreateNumericArray(4,fielddim,mxSINGLE_CLASS,mxREAL));
            memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[0],jstruct,0)),cfg.exportfield,
                         fielddim[0]*fielddim[1]*fielddim[2]*fielddim[3]*sizeof(float));
            free(cfg.exportfield);
            cfg.exportfield=NULL;

            mxArray *stat=mxCreateStructMatrix(1,1,6,statstruct);
            mxArray *val = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(val) = cfg.runtime;
            mxSetFieldByNumber(stat,0,0, val);

            val = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(val) = cfg.nphoton;
            mxSetFieldByNumber(stat,0,1, val);

            val = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(val) = cfg.energytot;
            mxSetFieldByNumber(stat,0,2, val);

            val = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(val) = cfg.energyabs;
            mxSetFieldByNumber(stat,0,3, val);

            val = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(val) = cfg.normalizer;
            mxSetFieldByNumber(stat,0,4, val);

            val = mxCreateDoubleMatrix(1,activedev,mxREAL);
	    for(int i=0;i<activedev;i++)
                *(mxGetPr(val)+i) = cfg.workload[i];
            mxSetFieldByNumber(stat,0,5, val);

	    mxSetFieldByNumber(plhs[0],jstruct,1, stat);
        }
    }catch(const char *err){
      mexPrintf("Error: %s\n",err);
    }catch(const std::exception &err){
      mexPrintf("C++ Error: %s\n",err.what());
    }catch(...){
      mexPrintf("Unknown Exception");
    }
    if(detps)
       free(detps);
    mcx_cleargpuinfo(&gpuinfo);
    mcx_clearcfg(&cfg);
  }
  return;
}


void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg){
    const char *name=mxGetFieldNameByNumber(root,idx);
    const int *arraydim;
    char *jsonshapes=NULL;
    int i,j;

    if(strcmp(name,"nphoton")==0 && cfg->replay.seed!=NULL)
	return;

    cfg->flog=stderr;
    GET_1ST_FIELD(cfg,nphoton)
    GET_ONE_FIELD(cfg,nblocksize)
    GET_ONE_FIELD(cfg,nthread)
    GET_ONE_FIELD(cfg,tstart)
    GET_ONE_FIELD(cfg,tstep)
    GET_ONE_FIELD(cfg,tend)
    GET_ONE_FIELD(cfg,maxdetphoton)
    GET_ONE_FIELD(cfg,sradius)
    GET_ONE_FIELD(cfg,maxgate)
    GET_ONE_FIELD(cfg,respin)
    GET_ONE_FIELD(cfg,isreflect)
    GET_ONE_FIELD(cfg,isref3)
    GET_ONE_FIELD(cfg,isrefint)
    GET_ONE_FIELD(cfg,isnormalized)
    GET_ONE_FIELD(cfg,isgpuinfo)
    GET_ONE_FIELD(cfg,issrcfrom0)
    GET_ONE_FIELD(cfg,autopilot)
    GET_ONE_FIELD(cfg,minenergy)
    GET_ONE_FIELD(cfg,unitinmm)
    GET_ONE_FIELD(cfg,reseedlimit)
    GET_ONE_FIELD(cfg,printnum)
    GET_ONE_FIELD(cfg,voidtime)
    GET_ONE_FIELD(cfg,issaveseed)
    GET_ONE_FIELD(cfg,replaydet)
    GET_ONE_FIELD(cfg,faststep)
    GET_VEC3_FIELD(cfg,srcpos)
    GET_VEC3_FIELD(cfg,srcdir)
    GET_VEC3_FIELD(cfg,steps)
    GET_VEC3_FIELD(cfg,crop0)
    GET_VEC3_FIELD(cfg,crop1)
    GET_VEC4_FIELD(cfg,srcparam1)
    GET_VEC4_FIELD(cfg,srcparam2)
    else if(strcmp(name,"vol")==0){
        if(!mxIsUint8(item) || mxGetNumberOfDimensions(item)!=3 )
             mexErrMsgTxt("the 'vol' field must be a 3D uint8 array");
        arraydim=mxGetDimensions(item);
	for(i=0;i<3;i++) ((unsigned int *)(&cfg->dim))[i]=arraydim[i];
	if(cfg->vol) free(cfg->vol);
	cfg->vol=(unsigned char *)malloc(cfg->dim.x*cfg->dim.y*cfg->dim.z);
	memcpy(cfg->vol,mxGetData(item),cfg->dim.x*cfg->dim.y*cfg->dim.z);
        printf("mcx.dim=[%d %d %d];\n",cfg->dim.x,cfg->dim.y,cfg->dim.z);
    }else if(strcmp(name,"detpos")==0){
        arraydim=mxGetDimensions(item);
	if(arraydim[0]>0 && arraydim[1]!=4)
            mexErrMsgTxt("the 'detpos' field must have 4 columns (x,y,z,radius)");
        double *val=mxGetPr(item);
        cfg->detnum=arraydim[0];
	if(cfg->detpos) free(cfg->detpos);
        cfg->detpos=(float4 *)malloc(cfg->detnum*sizeof(float4));
        for(j=0;j<4;j++)
          for(i=0;i<cfg->detnum;i++)
             ((float *)(&cfg->detpos[i]))[j]=val[j*cfg->detnum+i];
        printf("mcx.detnum=%d;\n",cfg->detnum);
    }else if(strcmp(name,"prop")==0){
        arraydim=mxGetDimensions(item);
        if(arraydim[0]>0 && arraydim[1]!=4)
            mexErrMsgTxt("the 'prop' field must have 4 columns (mua,mus,g,n)");
        double *val=mxGetPr(item);
        cfg->medianum=arraydim[0];
        if(cfg->prop) free(cfg->prop);
        cfg->prop=(Medium *)malloc(cfg->medianum*sizeof(Medium));
        for(j=0;j<4;j++)
          for(i=0;i<cfg->medianum;i++)
             ((float *)(&cfg->prop[i]))[j]=val[j*cfg->medianum+i];
        printf("mcx.medianum=%d;\n",cfg->medianum);
    }else if(strcmp(name,"session")==0){
        int len=mxGetNumberOfElements(item);
        if(!mxIsChar(item) || len==0)
             mexErrMsgTxt("the 'session' field must be a non-empty string");
	if(len>MAX_SESSION_LENGTH)
	     mexErrMsgTxt("the 'session' field is too long");
        int status = mxGetString(item, cfg->session, MAX_SESSION_LENGTH);
        if (status != 0)
             mexWarnMsgTxt("not enough space. string is truncated.");

	printf("mcx.session='%s';\n",cfg->session);
    }else if(strcmp(name,"srctype")==0){
        int len=mxGetNumberOfElements(item);
        const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar","pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian","line","slit",""};
        char strtypestr[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
             mexErrMsgTxt("the 'srctype' field must be a non-empty string");
	if(len>MAX_SESSION_LENGTH)
	     mexErrMsgTxt("the 'srctype' field is too long");
        int status = mxGetString(item, strtypestr, MAX_SESSION_LENGTH);
        if (status != 0)
             mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->srctype=mcx_keylookup(strtypestr,srctypeid);
        if(cfg->srctype==-1)
             mexErrMsgTxt("the specified source type is not supported");
	printf("mcx.srctype='%s';\n",strtypestr);
    }else if(strcmp(name,"outputtype")==0){
        int len=mxGetNumberOfElements(item);
        const char *outputtype[]={"flux","fluence","energy","jacobian",""};
        char outputstr[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
             mexErrMsgTxt("the 'outputtype' field must be a non-empty string");
	if(len>MAX_SESSION_LENGTH)
	     mexErrMsgTxt("the 'outputtype' field is too long");
        int status = mxGetString(item, outputstr, MAX_SESSION_LENGTH);
        if (status != 0)
             mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->outputtype=mcx_keylookup(outputstr,outputtype);
        if(cfg->outputtype==-1)
             mexErrMsgTxt("the specified output type is not supported");
	printf("mcx.outputtype='%s';\n",outputstr);
    }else if(strcmp(name,"debuglevel")==0){
        int len=mxGetNumberOfElements(item);
        const char debugflag[]={'R','\0'};
        char debuglevel[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
             mexErrMsgTxt("the 'debuglevel' field must be a non-empty string");
	if(len>MAX_SESSION_LENGTH)
	     mexErrMsgTxt("the 'debuglevel' field is too long");
        int status = mxGetString(item, debuglevel, MAX_SESSION_LENGTH);
        if (status != 0)
             mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->debuglevel=mcx_parsedebugopt(debuglevel,debugflag);
        if(cfg->debuglevel==0)
             mexWarnMsgTxt("the specified debuglevel is not supported");
	printf("mcx.debuglevel='%d';\n",cfg->debuglevel);
    }else if(strcmp(name,"srcpattern")==0){
        arraydim=mxGetDimensions(item);
        double *val=mxGetPr(item);
	if(cfg->srcpattern) free(cfg->srcpattern);
        cfg->srcpattern=(float*)malloc(arraydim[0]*arraydim[1]*sizeof(float));
        for(i=0;i<arraydim[0]*arraydim[1];i++)
             cfg->srcpattern[i]=val[i];
        printf("mcx.srcpattern=[%d %d];\n",arraydim[0],arraydim[1]);
    }else if(strcmp(name,"shapes")==0){
        int len=mxGetNumberOfElements(item);
        if(!mxIsChar(item) || len==0)
             mexErrMsgTxt("the 'shapes' field must be a non-empty string");

        jsonshapes=new char[len+1];
        mxGetString(item, jsonshapes, len+1);
        jsonshapes[len]='\0';
    }else if(strcmp(name,"detphotons")==0){
        arraydim=mxGetDimensions(item);
	dimdetps[0]=arraydim[0];
	dimdetps[1]=arraydim[1];
        detps=(float *)malloc(arraydim[0]*arraydim[1]*sizeof(float));
        memcpy(detps,mxGetData(item),arraydim[0]*arraydim[1]*sizeof(float));
        printf("mcx.detphotons=[%d %d];\n",arraydim[0],arraydim[1]);
    }else if(strcmp(name,"seed")==0){
        arraydim=mxGetDimensions(item);
        if(MAX(arraydim[0],arraydim[1])==0)
            mexErrMsgTxt("the 'seed' field can not be empty");
        if(!mxIsUint8(item)){
            double *val=mxGetPr(item);
            cfg->seed=val[0];
            printf("mcx.seed=%d;\n",cfg->seed);
        }else{
	    seedbyte=arraydim[0];
            cfg->replay.seed=malloc(arraydim[0]*arraydim[1]);
            if(arraydim[0]!=sizeof(float)*RAND_BUF_LEN)
                mexErrMsgTxt("the row number of cfg.seed does not match RNG seed byte-length");
            memcpy(cfg->replay.seed,mxGetData(item),arraydim[0]*arraydim[1]);
            cfg->seed=SEED_FROM_FILE;
            cfg->nphoton=arraydim[1];
            printf("mcx.nphoton=%d;\n",cfg->nphoton);
        }
    }else if(strcmp(name,"gpuid")==0){
        int len=mxGetNumberOfElements(item);

        if(mxIsChar(item)){
	   if(len==0)
             mexErrMsgTxt("the 'gpuid' field must be an integer or non-empty string");
	   if(len>MAX_DEVICE)
		mexErrMsgTxt("the 'gpuid' field is too long");
           int status = mxGetString(item, cfg->deviceid, MAX_DEVICE);
           if (status != 0)
        	mexWarnMsgTxt("not enough space. string is truncated.");

           printf("mcx.gpuid='%s';\n",cfg->deviceid);
	}else{
           double *val=mxGetPr(item);
	   cfg->gpuid=val[0];
           memset(cfg->deviceid,0,MAX_DEVICE);
           if(cfg->gpuid<MAX_DEVICE){
                memset(cfg->deviceid,'0',cfg->gpuid-1);
           	cfg->deviceid[cfg->gpuid-1]='1';
           }else
           	mexErrMsgTxt("GPU id can not be more than 256");
           printf("mcx.gpuid=%d;\n",cfg->gpuid);
	}
        for(int i=0;i<MAX_DEVICE;i++)
           if(cfg->deviceid[i]=='0')
              cfg->deviceid[i]='\0';
    }else if(strcmp(name,"workload")==0){
        double *val=mxGetPr(item);
	arraydim=mxGetDimensions(item);
	if(arraydim[0]*arraydim[1]>MAX_DEVICE)
	     mexErrMsgTxt("the workload list can not be longer than 256");
	for(i=0;i<arraydim[0]*arraydim[1];i++)
	     cfg->workload[i]=val[i];
        printf("mcx.workload=<<%d>>;\n",arraydim[0]*arraydim[1]);
    }else{
        printf("WARNING: redundant field '%s'\n",name);
    }
    if(jsonshapes){
        Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},0};
        if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
        int status=mcx_parse_shapestring(&grid,jsonshapes);
        delete [] jsonshapes;
        if(status){
              mexErrMsgTxt(mcx_last_shapeerror());
        }
    }
}

void mcx_replay_prep(Config *cfg){
    int i,j;
    if(detps==NULL || cfg->seed!=SEED_FROM_FILE)
        return;
    if(cfg->nphoton!=dimdetps[1])
        mexErrMsgTxt("the column numbers of detphotons and seed do not match");
    if(seedbyte==0)
        mexErrMsgTxt("the seed input is empty");

    cfg->replay.weight=(float*)malloc(cfg->nphoton*sizeof(float));
    cfg->replay.tof=(float*)calloc(cfg->nphoton,sizeof(float));

    cfg->nphoton=0;
    for(i=0;i<dimdetps[1];i++)
        if(cfg->replaydet==0 || cfg->replaydet==(int)(detps[i*dimdetps[0]])){
            if(i!=cfg->nphoton)
                memcpy((char *)(cfg->replay.seed)+cfg->nphoton*seedbyte, (char *)(cfg->replay.seed)+i*seedbyte, seedbyte);
            cfg->replay.weight[cfg->nphoton]=1.f;
            for(j=2;j<cfg->medianum+1;j++){
                cfg->replay.weight[cfg->nphoton]*=expf(-cfg->prop[j-1].mua*detps[i*dimdetps[0]+j]*cfg->unitinmm);
                cfg->replay.tof[cfg->nphoton]+=detps[i*dimdetps[0]+j]*cfg->unitinmm*R_C0*cfg->prop[j-1].n;
            }
            if(cfg->replay.tof[cfg->nphoton]<cfg->tstart || cfg->replay.tof[cfg->nphoton]>cfg->tend) /*need to consider -g*/
                continue;
            cfg->nphoton++;
        }
}

void mcx_validate_config(Config *cfg){
     int i,gates,idx1d;

     if(!cfg->issrcfrom0){
        cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
     }
     if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
         mexErrMsgTxt("incorrect time gate settings");
     }
     if(ABS(cfg->srcdir.x*cfg->srcdir.x+cfg->srcdir.y*cfg->srcdir.y+cfg->srcdir.z*cfg->srcdir.z - 1.f)>1e-5)
         mexErrMsgTxt("field 'srcdir' must be a unitary vector");
     if(cfg->steps.x==0.f || cfg->steps.y==0.f || cfg->steps.z==0.f)
         mexErrMsgTxt("field 'steps' can not have zero elements");
     if(cfg->tend<=cfg->tstart)
         mexErrMsgTxt("field 'tend' must be greater than field 'tstart'");
     gates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(cfg->maxgate>gates)
	 cfg->maxgate=gates;
     if(cfg->sradius>0.f){
     	cfg->crop0.x=MAX((int)(cfg->srcpos.x-cfg->sradius),0);
     	cfg->crop0.y=MAX((int)(cfg->srcpos.y-cfg->sradius),0);
     	cfg->crop0.z=MAX((int)(cfg->srcpos.z-cfg->sradius),0);
     	cfg->crop1.x=MIN((int)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	cfg->crop1.y=MIN((int)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	cfg->crop1.z=MIN((int)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
     }else if(cfg->sradius==0.f){
     	memset(&(cfg->crop0),0,sizeof(uint3));
     	memset(&(cfg->crop1),0,sizeof(uint3));
     }else{
        /*
            if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
        */
        if(!cfg->issrcfrom0){
            cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
            cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
        }
     }
     if(cfg->medianum==0)
        mexErrMsgTxt("you must define the 'prop' field in the input structure");
     if(cfg->dim.x==0||cfg->dim.y==0||cfg->dim.z==0)
        mexErrMsgTxt("the 'vol' field in the input structure can not be empty");
     if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcpattern==NULL)
        mexErrMsgTxt("the 'srcpattern' field can not be empty when your 'srctype' is 'pattern'");
     if(cfg->steps.x!=1.f && cfg->unitinmm==1.f)
        cfg->unitinmm=cfg->steps.x;

     if(cfg->unitinmm!=1.f){
        cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
        for(i=1;i<cfg->medianum;i++){
		cfg->prop[i].mus*=cfg->unitinmm;
		cfg->prop[i].mua*=cfg->unitinmm;
        }
     }
     if(cfg->issavedet && cfg->detnum==0) 
      	cfg->issavedet=0;
     if(cfg->seed<0 && cfg->seed!=SEED_FROM_FILE) cfg->seed=time(NULL);
     if(cfg->outputtype==otJacobian && cfg->seed!=SEED_FROM_FILE)
         mexErrMsgTxt("Jacobian output is only valid in the reply mode. Please define cfg.seed");     
     for(i=0;i<cfg->detnum;i++){
        if(!cfg->issrcfrom0){
		cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	}
     }
     if(1){
        cfg->isrowmajor=0; /*matlab is always col-major*/
	if(cfg->isrowmajor){
		/*from here on, the array is always col-major*/
		mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
		cfg->isrowmajor=0;
	}
	if(cfg->issavedet)
		mcx_maskdet(cfg);
        if(cfg->seed==SEED_FROM_FILE){
            if(cfg->respin>1){
	       cfg->respin=1;
	       fprintf(stderr,"Warning: respin is disabled in the replay mode\n");
	    }
        }
     }
     cfg->his.maxmedia=cfg->medianum-1; /*skip medium 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+1; /*column count=maxmedia+2*/
     mcx_replay_prep(cfg);
}

extern "C" int mcx_throw_exception(const int id, const char *msg, const char *filename, const int linenum){
     printf("MCXLAB ERROR %d in unit %s:%d\n",id,filename,linenum);
     throw msg;
     return id;
}

void mcxlab_usage(){
     printf("Usage:\n    [flux,detphoton,vol,seeds]=mcxlab(cfg);\n\nPlease run 'help mcxlab' for more details.\n");
}
