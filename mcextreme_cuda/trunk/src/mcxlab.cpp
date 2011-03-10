////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon transport
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
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

#include "mex.h"
#include "mcx_utils.h"
#include "mcx_core.h"

#define GET_1ST_FIELD(x,y)  if(strcmp(name,#y)==0) {double *val=mxGetPr(item);x->y=val[0];printf("mcx.%s=%g;\n",#y,(float)(x->y));}
#define GET_ONE_FIELD(x,y)  else GET_1ST_FIELD(x,y)
#define GET_VEC3_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];}

void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg);
void mcx_validate_config(Config *cfg);
void mcxlab_usage();

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  Config cfg;
  mxArray    *tmp, *fout;
  int        ifield, jstruct;
  int        ncfg, nfields, ndim;
  mwSize     fielddim[4];
  const char       *outputtag[]={"data"};

  if (nrhs==0){
     mcxlab_usage();
     return;
  }
  if (!mxIsStruct(prhs[0]))
     mexErrMsgTxt("Input must be a structure.");

  nfields = mxGetNumberOfFields(prhs[0]);
  ncfg = mxGetNumberOfElements(prhs[0]);

  if(nlhs>=1)
      plhs[0] = mxCreateStructMatrix(ncfg,1,1,outputtag);
  if(nlhs>=2)
      plhs[1] = mxCreateStructMatrix(ncfg,1,1,outputtag);

  for (jstruct = 0; jstruct < ncfg; jstruct++) {
    mcx_initcfg(&cfg);

    for (ifield = 0; ifield < nfields; ifield++) { /* how many configs */
        tmp = mxGetFieldByNumber(prhs[0], jstruct, ifield);
	if (tmp == NULL) {
		continue;
	}
	mcx_set_field(prhs[0],tmp,ifield,&cfg);
    }
    if(cfg.vol==NULL || cfg.medianum==0){
	mexErrMsgTxt("You must define 'vol' and 'prop' field.");
    }
    if(!mcx_set_gpu(&cfg)){
        mexErrMsgTxt("No GPU device found");
    }
    if(nlhs>=1){
        fielddim[0]=cfg.dim.x; fielddim[1]=cfg.dim.y; 
	fielddim[2]=cfg.dim.z; fielddim[3]=cfg.maxgate; 
	mxSetFieldByNumber(plhs[0],jstruct,0, mxCreateNumericArray(4,fielddim,mxSINGLE_CLASS,mxREAL));
	cfg.exportfield = (float*)mxGetPr(mxGetFieldByNumber(plhs[0],jstruct,0));
    }
    if(nlhs>=2){
        fielddim[0]=(cfg.medianum+1); fielddim[1]=cfg.his.savedphoton; 
	fielddim[2]=0; fielddim[3]=0;
	if(cfg.his.savedphoton>0){
		mxSetFieldByNumber(plhs[1],jstruct,0, mxCreateNumericArray(2,fielddim,mxSINGLE_CLASS,mxREAL));
		cfg.exportdetected = (float*)mxGetPr(mxGetFieldByNumber(plhs[1],jstruct,0));
	}
    }
    mcx_validate_config(&cfg);
    //mcx_writeconfig((const char*)"test.txt",&cfg);
    mcx_run_simulation(&cfg);
    mcx_clearcfg(&cfg);
  }
  return;
}


void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg){
    const char *name=mxGetFieldNameByNumber(root,idx);
    const int *arraydim;
    int i,j;

    GET_1ST_FIELD(cfg,nphoton)
    GET_ONE_FIELD(cfg,nblocksize)
    GET_ONE_FIELD(cfg,nthread)
    GET_ONE_FIELD(cfg,seed)
    GET_ONE_FIELD(cfg,tstart)
    GET_ONE_FIELD(cfg,tstep)
    GET_ONE_FIELD(cfg,tend)
    GET_ONE_FIELD(cfg,maxdetphoton)
    GET_ONE_FIELD(cfg,detradius)
    GET_ONE_FIELD(cfg,sradius)
    GET_ONE_FIELD(cfg,maxgate)
    GET_ONE_FIELD(cfg,respin)
    GET_ONE_FIELD(cfg,gpuid)
    GET_ONE_FIELD(cfg,isreflect)
    GET_ONE_FIELD(cfg,isref3)
    GET_ONE_FIELD(cfg,isreflectin)
    GET_ONE_FIELD(cfg,isnormalized)
    GET_ONE_FIELD(cfg,issavedet)
    GET_ONE_FIELD(cfg,issave2pt)
    GET_ONE_FIELD(cfg,isgpuinfo)
    GET_ONE_FIELD(cfg,autopilot)
    GET_ONE_FIELD(cfg,minenergy)
    GET_ONE_FIELD(cfg,unitinmm)
    GET_VEC3_FIELD(cfg,srcpos)
    GET_VEC3_FIELD(cfg,srcdir)
    GET_VEC3_FIELD(cfg,steps)
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
            mexErrMsgTxt("the 'prop' field must have 4 columns (mua,mus,n,g)");
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
     }else{
     	memset(&(cfg->crop0),0,sizeof(uint3));
     	memset(&(cfg->crop1),0,sizeof(uint3));
     }
     if(cfg->medianum==0)
        mexErrMsgTxt("you must define the 'prop' field in the input structure");
     if(cfg->dim.x==0||cfg->dim.y==0||cfg->dim.z==0)
        mexErrMsgTxt("the 'vol' field in the input structure can not be empty");

     for(i=1;i<cfg->medianum;i++){
	if(cfg->unitinmm!=1.f){
		cfg->prop[i].mus*=cfg->unitinmm;
		cfg->prop[i].mua*=cfg->unitinmm;
		cfg->tstart*=cfg->unitinmm;
		cfg->tend*=cfg->unitinmm;
		cfg->tstep*=cfg->unitinmm;
	}
     }
     if(cfg->issavedet && cfg->detnum==0) 
      	cfg->issavedet=0;
     for(i=0;i<cfg->detnum;i++){
	cfg->detpos[i].w=cfg->detpos[i].w*cfg->detpos[i].w;
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
	if(cfg->srcpos.x<0.f || cfg->srcpos.y<0.f || cfg->srcpos.z<0.f || 
	   cfg->srcpos.x>=cfg->dim.x || cfg->srcpos.y>=cfg->dim.y || cfg->srcpos.z>=cfg->dim.z)
		mexErrMsgTxt("source position is outside of the volume");
	idx1d=(int)(int(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+int(cfg->srcpos.y)*cfg->dim.x+int(cfg->srcpos.x));

        /* if the specified source position is outside the domain, move the source
	   along the initial vector until it hit the domain */
	if(cfg->vol && cfg->vol[idx1d]==0){
                printf("source (%f %f %f) is located outside the domain, vol[%d]=%d\n",
		      cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,idx1d,cfg->vol[idx1d]);
		while(cfg->vol[idx1d]==0){
			cfg->srcpos.x+=cfg->srcdir.x;
			cfg->srcpos.y+=cfg->srcdir.y;
			cfg->srcpos.z+=cfg->srcdir.z;
			printf("fixing source position to (%f %f %f)\n",cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
			idx1d=(int)(int(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+int(cfg->srcpos.y)*cfg->dim.x+int(cfg->srcpos.x));
		}
	}
     }
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+1; /*column count=maxmedia+2*/
}

void mcxlab_usage(){
     printf("====================================================================\n\
      MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave\n\
--------------------------------------------------------------------\n\
Copyright (c) 2010,2011 Qianqian Fang <fangq at nmr.mgh.harvard.edu>\n\
                      URL: http://mcx.sf.net\n\
====================================================================\n\
\n\
Format:\n\
    [flux,detphoton]=mcxlab(cfg);\n\
\n\
Input:\n\
    cfg: a struct, or struct array. Each element of cfg defines \n\
         the parameters associated with a simulation. \n\
\n\
    It may contain the following fields:\n\
     *cfg.nphoton:    the total number of photons to be simulated (integer)\n\
     *cfg.vol:        a 3D array specifying the media index in the domain\n\
     *cfg.prop:       an N by 4 array, each row specifies [mua, mus, n, g] in order.\n\
                      the first row corresponds to medium type 0 which is \n\
                      typically [0 0 1 1]. The second row is type 1, and so on.\n\
     *cfg.tstart:     starting time of the simulation (in seconds)\n\
     *cfg.tstep:      time-gate width of the simulation (in seconds)\n\
     *cfg.tend:       ending time of the simulation (in second)\n\
     *cfg.srcpos:     a 1 by 3 vector, specifying the position of the source\n\
     *cfg.srcdir:     a 1 by 3 vector, specifying the incident vector\n\
      cfg.nblocksize: how many CUDA thread blocks to be used [64]\n\
      cfg.nthread:    the total CUDA thread number [2048]\n\
      cfg.session:    a string for output file names (used when no return variables)\n\
      cfg.seed:       seed for the random number generator (integer) [0]\n\
      cfg.maxdetphoton:   maximum number of photons saved by the detectors [1000000]\n\
      cfg.detpos:     an N by 4 array, each row specifying a detector: [x,y,z,radius]\n\
      cfg.detradius:  radius of the detector (in mm) [1.0]\n\
      cfg.sradius:    radius within which we use atomic operations (in mm) [0.0]\n\
      cfg.respin:     repeat simulation for the given time (integer) [1]\n\
      cfg.gpuid:      which GPU to use (run 'mcx -L' to list all GPUs) [1]\n\
      cfg.isreflect:  [1]-consider refractive index mismatch, 0-matched index\n\
      cfg.isref3:     [1]-consider maximum 3 reflection interface; 0-only 2\n\
      cfg.isnormalized:[1]-normalize the output flux to unitary source, 0-no reflection\n\
      cfg.issavedet:  1-to save detected photon partial path length, [0]-do not save\n\
      cfg.issave2pt:  [1]-to save flux distribution, 0-do not save\n\
      cfg.isgpuinfo:  1-print GPU info, [0]-do not print\n\
      cfg.autopilot:  1-automatically set threads and blocks, [0]-use nthread/nblocksize\n\
      cfg.minenergy:  terminate photon when weight less than this level (float) [0.0]\n\
      cfg.unitinmm:   defines the length unit for a grid edge length [1.0]\n\
\n\
      fields with * are required; options in [] are the default values\n\
\n\
Output:\n\
      flux: a struct array, with a length equals to that of cfg.\n\
            For each element of flux, flux(i).data is a 4D array with\n\
            dimensions specified by [size(vol) total-time-gates]. \n\
            The content of the array is the normalized flux at \n\
            each voxel of each time-gate.\n\
      detphoton: a struct array, with a length equals to that of cfg.\n\
            For each element of detphoton, detphoton(i).data is a 2D array with\n\
            dimensions [size(cfg.prop,1)+1 saved-photon-num]. The first row\n\
            is the ID(>0) of the detector that captures the photon; the second\n\
	    row is the weight of the photon when it is detected; the rest rows\n\
	    are the partial path lengths (in mm) traveling in medium 1 up to the last.\n\
\n\
      if detphoton is ignored, the detected photon will be saved in an .mch file \n\
      if cfg.issavedeet=1; if no output is given, the flux will be saved to an \n\
      .mc2 file if cfg.issave2pt=1 (which is true by default).\n\
\n\
Example:\n\
      cfg.nphoton=1e7;\n\
      cfg.vol=uint8(ones(60,60,60));\n\
      cfg.srcpos=[30 30 1];\n\
      cfg.srcdir=[0 0 1];\n\
      cfg.gpuid=1;\n\
      cfg.autopilot=1;\n\
      cfg.prop=[0 0 1 1;0.005 1 1.37 0];\n\
      cfg.tstart=0;\n\
      cfg.tend=5e-9;\n\
      cfg.tstep=5e-10;\n\
      flux=mcxlab(cfg);\n\
      \n\
      cfgs(1)=cfg;\n\
      cfgs(2)=cfg;\n\
      cfgs(1).isreflect=0;\n\
      cfgs(2).isreflect=1;\n\
      cfgs(2).issavedet=1;\n\
      cfgs(2).detpos=[30 20 1 1;30 40 1 1;20 30 1 1;40 30 1 1];\n\
      [fluxs,detps]=mcxlab(cfgs);\n\
      \n\
      imagesc(squeeze(log(fluxs(1).data(:,30,:,1)))-squeeze(log(fluxs(2).data(:,30,:,1))));\n\
");
}
