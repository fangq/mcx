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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  Config cfg;
  mxArray    *tmp, *fout;
  int        ifield, jstruct;
  int        ncfg, nfields, ndim;

  if (nrhs!=1)
     mexErrMsgTxt("One input is required.");
  if (!mxIsStruct(prhs[0]))
     mexErrMsgTxt("Input must be a structure.");

  nfields = mxGetNumberOfFields(prhs[0]);
  ncfg = mxGetNumberOfElements(prhs[0]);
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
    mcx_validate_config(&cfg);
    //mcx_writeconfig((const char*)"test.txt",&cfg);
    mcx_run_simulation(&cfg);
    mcx_clearcfg(&cfg);
  }
  return;
}


void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg){
    const char *name=mxGetFieldNameByNumber(root,idx);
    const mwSize *arraydim;
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
	for(i=0;i<3;i++) ((uint *)(&cfg->dim))[i]=arraydim[i];
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
         mexErrMsgTxt("field 'steps' can not be have zero elements");
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
	cfg->detpos[i].w=cfg->detradius*cfg->detradius;
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
