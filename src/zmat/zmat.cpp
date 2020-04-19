/***************************************************************************//**
**  \mainpage ZMAT: a data compression function for MATLAB/octave
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2019
**
**  Functions: base64_encode()/base64_decode()
**  \copyright 2005-2011, Jouni Malinen <j@w1.fi>
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/


/***************************************************************************//**
\file    zmat.cpp

@brief   mex function for ZMAT
*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <exception>
#include <ctype.h>
#include <assert.h>

#include "mex.h"
#include "zmatlib.h"
#include "zlib.h"

void zmat_usage();

const char  *metadata[]={"type","size","byte","method","status"};

/** @brief Mex function for the zmat - an interface to compress/decompress binary data
 *  This is the master function to interface for zipping and unzipping a char/int8 buffer
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  TZipMethod zipid=zmZlib;
  int iscompress=1;
#if defined(NO_LZ4) && defined(NO_LZMA)
  const char *zipmethods[]={"zlib","gzip","base64",""};
#elif !defined(NO_LZMA) && defined(NO_LZ4)
  const char *zipmethods[]={"zlib","gzip","base64","lzip","lzma",""};
#elif defined(NO_LZMA) && !defined(NO_LZ4)
  const char *zipmethods[]={"zlib","gzip","base64","lz4","lz4hc",""};
#else
  const char *zipmethods[]={"zlib","gzip","base64","lzip","lzma","lz4","lz4hc",""};
#endif

  /**
   * If no input is given for this function, it prints help information and return.
   */
  if (nrhs==0){
     zmat_usage();
     return;
  }

  if(nrhs>=2){
      double *val=mxGetPr(prhs[1]);
      iscompress=val[0];
  }
  if(nrhs>=3){
      int len=mxGetNumberOfElements(prhs[2]);
      if(!mxIsChar(prhs[2]) || len==0)
             mexErrMsgTxt("the 'method' field must be a non-empty string");
      if((zipid=(TZipMethod)zmat_keylookup((char *)mxArrayToString(prhs[2]), zipmethods))<0)
             mexErrMsgTxt("the specified compression method is not supported");
  }

  try{
	  if(mxIsChar(prhs[0]) || (mxIsNumeric(prhs[0]) && ~mxIsComplex(prhs[0])) || mxIsLogical(prhs[0])){
	       int ret=-1;
	       mwSize inputsize=mxGetNumberOfElements(prhs[0])*mxGetElementSize(prhs[0]);
	       mwSize buflen[2]={0};
	       unsigned char *outputbuf=NULL;
	       size_t outputsize=0;
	       unsigned char * inputstr=(mxIsChar(prhs[0])? (unsigned char *)mxArrayToString(prhs[0]) : (unsigned char *)mxGetData(prhs[0]));
    	       int errcode=zmat_run(inputsize, inputstr, &outputsize, &outputbuf, zipid, &ret, iscompress);
               if(errcode<0){
	           if(outputbuf)
		       free(outputbuf);
		   outputbuf=NULL;
	           outputsize=0;
	       }
	       buflen[0]=1;
	       buflen[1]=outputsize;
	       plhs[0] = mxCreateNumericArray(2,buflen,mxUINT8_CLASS,mxREAL);
	       if(outputbuf){
		    memcpy((unsigned char*)mxGetPr(plhs[0]),outputbuf,buflen[1]);
		    free(outputbuf);
	       }
	       if(nlhs>1){
	            mwSize inputdim[2]={1,0}, *dims=(mwSize *)mxGetDimensions(prhs[0]);
		    unsigned int *inputsize=NULL;
	            plhs[1]=mxCreateStructMatrix(1,1,5,metadata);
		    mxArray *val = mxCreateString(mxGetClassName(prhs[0]));
                    mxSetFieldByNumber(plhs[1],0,0, val);

		    inputdim[1]=mxGetNumberOfDimensions(prhs[0]);
		    inputsize=(unsigned int *)malloc(inputdim[1]*sizeof(unsigned int));
		    val = mxCreateNumericArray(2, inputdim, mxUINT32_CLASS, mxREAL);
		    for(int i=0;i<inputdim[1];i++)
		        inputsize[i]=dims[i];
		    memcpy(mxGetPr(val),inputsize,inputdim[1]*sizeof(unsigned int));
                    mxSetFieldByNumber(plhs[1],0,1, val);

                    val = mxCreateDoubleMatrix(1,1,mxREAL);
                    *mxGetPr(val) = mxGetElementSize(prhs[0]);
                    mxSetFieldByNumber(plhs[1],0,2, val);

		    val = mxCreateString(zipmethods[zipid]);
                    mxSetFieldByNumber(plhs[1],0,3, val);

                    val = mxCreateDoubleMatrix(1,1,mxREAL);
                    *mxGetPr(val) = ret;
                    mxSetFieldByNumber(plhs[1],0,4, val);
	       }
	       if(errcode<0){
	           mexErrMsgTxt(zmat_error(-errcode));
	       }
	  }else{
	      mexErrMsgTxt("the input must be a char, non-complex numerical or logical array");
	  }
  }catch(const char *err){
      mexPrintf("Error: %s\n",err);
  }catch(const std::exception &err){
      mexPrintf("C++ Error: %s\n",err.what());
  }catch(...){
      mexPrintf("Unknown Exception");
  }
  return;
}

/**
 * @brief Print a brief help information if nothing is provided
 */

void zmat_usage(){
     printf("Usage:\n\t[output,info]=zmat(input,iscompress,method);\n\nPlease run 'help zmat' for more details.\n");
}
