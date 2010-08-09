#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "mcx_utils.h"
#include "mcx_core.h"

#define GET_1ST_FIELD(x,y) if(strcmp(name,#y)==0) {val=mxGetPr(item);x->y =val[0];printf("%s is set to %d\n",#y, x->y);}
#define GET_ONE_FIELD(x,y) else GET_1ST_FIELD(x,y)


void mcx_set_field(const mxArray *root,const mxArray *item,int idx, Config *cfg){
    double *val;
    const char *name=mxGetFieldNameByNumber(root,idx);
    GET_1ST_FIELD(cfg,nphoton)
    GET_ONE_FIELD(cfg,nblocksize)
    GET_ONE_FIELD(cfg,nthread)
    GET_ONE_FIELD(cfg,seed)
    GET_ONE_FIELD(cfg,tstart)
    GET_ONE_FIELD(cfg,tstep)
    GET_ONE_FIELD(cfg,tend)
    GET_ONE_FIELD(cfg,medianum)
    GET_ONE_FIELD(cfg,detnum)
    GET_ONE_FIELD(cfg,maxdetphoton)
    GET_ONE_FIELD(cfg,detradius)
    GET_ONE_FIELD(cfg,sradius)
    GET_ONE_FIELD(cfg,maxgate)
    GET_ONE_FIELD(cfg,respin)
    GET_ONE_FIELD(cfg,gpuid)
    GET_ONE_FIELD(cfg,isrowmajor)
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
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

  Config cfg;
  mxArray    *tmp, *fout;
  int        ifield, jstruct, *classIDflags;
  int        NStructElems, nfields, ndim;

  if (nrhs != 1)
     mexErrMsgTxt("One input is required.");
  if (!mxIsStruct(prhs[0]))
     mexErrMsgTxt("Input must be a structure.");

  nfields = mxGetNumberOfFields(prhs[0]);
  NStructElems = mxGetNumberOfElements(prhs[0]);

  for (jstruct = 0; jstruct < NStructElems; jstruct++) {
    mcx_initcfg(&cfg);

    for (ifield = 0; ifield < nfields; ifield++) { /* how many configs */
        tmp = mxGetFieldByNumber(prhs[0], jstruct, ifield);
	if (tmp == NULL) {
		continue;
	}
	mcx_set_field(prhs[0],tmp,ifield,&cfg);
    }
    mcx_run_simulation(&cfg);
    mcx_clearcfg(&cfg);
  }
  return;
}

