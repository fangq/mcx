/***************************************************************************//**
**  \mainpage ZMat - A portable C-library and MATLAB/Octave toolbox for inline data compression
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2019,2020,2022
**
**  ZMat provides an easy-to-use interface for stream compression and decompression.
**
**  It can be compiled as a MATLAB/Octave mex function (zipmat.mex/zmat.m) and compresses
**  arrays and strings in MATLAB/Octave. It can also be compiled as a lightweight
**  C-library (libzmat.a/libzmat.so) that can be called in C/C++/FORTRAN etc to
**  provide stream-level compression and decompression.
**
**  Currently, zmat/libzmat supports 6 different compression algorthms, including
**     - zlib and gzip : the most widely used algorithm algorithms for .zip and .gz files
**     - lzma and lzip : high compression ratio LZMA based algorithms for .lzma and .lzip files
**     - lz4 and lz4hc : real-time compression based on LZ4 and LZ4HC algorithms
**     - base64        : base64 encoding and decoding
**
**  ZMat is part of the NeuroJSON project (https://neurojson.org)
**  More information can be found at https://github.com/NeuroJSON/zmat
**
**  Depencency: ZLib library: https://www.zlib.net/
**  author: (C) 1995-2017 Jean-loup Gailly and Mark Adler
**
**  Depencency: LZ4 library: https://lz4.github.io/lz4/
**  author: (C) 2011-2019, Yann Collet,
**
**  Depencency: Original LZMA library
**  author: Igor Pavlov
**
**  Depencency: Eazylzma: https://github.com/lloyd/easylzma
**  author: Lloyd Hilaiel (lloyd)
**
**  Depencency: base64_encode()/base64_decode()
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

const char*  metadata[] = {"type", "size", "byte", "method", "status", "level"};

/** @brief Mex function for the zmat - an interface to compress/decompress binary data
 *  This is the master function to interface for zipping and unzipping a char/int8 buffer
 */

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    TZipMethod zipid = zmZlib;
    const char* zipmethods[] = {
        "zlib",
        "gzip",
        "base64",
#if !defined(NO_LZMA)
        "lzip",
        "lzma",
#endif
#if !defined(NO_LZ4)
        "lz4",
        "lz4hc",
#endif
#if !defined(NO_ZSTD)
        "zstd",
#endif
#if !defined(NO_BLOSC2)
        "blosc2blosclz",
        "blosc2lz4",
        "blosc2lz4hc",
        "blosc2zlib",
        "blosc2zstd",
#endif
        ""
    };

    const TZipMethod  zipmethodid[] = {
        zmZlib,
        zmGzip,
        zmBase64,
#if !defined(NO_LZMA)
        zmLzip,
        zmLzma,
#endif
#if !defined(NO_LZ4)
        zmLz4,
        zmLz4hc,
#endif
#if !defined(NO_ZSTD)
        zmZstd,
#endif
#if !defined(NO_BLOSC2)
        zmBlosc2Blosclz,
        zmBlosc2Lz4,
        zmBlosc2Lz4hc,
        zmBlosc2Zlib,
        zmBlosc2Zstd,
#endif
        zmUnknown
    };

    int use4bytedim = 0;

    union cflag {
        int iscompress;
        struct settings {
            char clevel;
            char nthread;
            char shuffle;
            char typesize;
        } param;
    } flags = {0};

    /**
     * If no input is given for this function, it prints help information and return.
     */
    if (nrhs == 0) {
        zmat_usage();
        return;
    }

    /**
     * Octave and MATLAB had changed mwSize from 4 byte to size_t (8 byte on 64bit machines)
     * in Octave 5/R2016. When running a mex/oct file compiled on newer libraries over an older
     * MATLAB/Octave versions, this can cause empyt output. The flag use4bytedim defined
     * in the below test is 1 when this situation happens, and adapt output accordingly.
     */
    if (sizeof(mwSize) == 8) {
        mwSize dims[2] = {0, 0};
        unsigned int* intdims = (unsigned int*)dims;
        intdims[0] = 2;
        intdims[1] = 3;
        mxArray* tmp = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL);
        use4bytedim = (mxGetNumberOfElements(tmp) == 6);
        mxDestroyArray(tmp);
    }

    flags.param.nthread = 1;
    flags.param.shuffle = 1;
    flags.param.typesize = 4;

    if (nrhs >= 2) {
        double* val = mxGetPr(prhs[1]);
        flags.param.clevel = (int)val[0];
    }

    if (nrhs >= 3) {
        int len = mxGetNumberOfElements(prhs[2]);

        if (!mxIsChar(prhs[2]) || len == 0) {
            mexErrMsgTxt("the 'method' field must be a non-empty string");
        }

        if ((zipid = (TZipMethod)zmat_keylookup((char*)mxArrayToString(prhs[2]), zipmethods)) < 0) {
            mexErrMsgTxt("the specified compression method is not supported");
        }

        zipid = zipmethodid[(int)zipid];
    }

    if (nrhs >= 4) {
        double* val = mxGetPr(prhs[3]);
        flags.param.nthread = val[0];
    }

    if (nrhs >= 5) {
        double* val = mxGetPr(prhs[4]);
        flags.param.shuffle = val[0];
    }

    if (nrhs >= 6) {
        double* val = mxGetPr(prhs[5]);
        flags.param.typesize = val[0];
    }

    try {
        if (mxIsChar(prhs[0]) || (mxIsNumeric(prhs[0]) && !mxIsComplex(prhs[0])) || mxIsLogical(prhs[0])) {
            int ret = -1;
            mwSize inputsize = mxGetNumberOfElements(prhs[0]) * mxGetElementSize(prhs[0]);
            mwSize buflen[2] = {0};
            unsigned char* outputbuf = NULL;
            size_t outputsize = 0;
            unsigned char* inputstr = (mxIsChar(prhs[0]) ? (unsigned char*)mxArrayToString(prhs[0]) : (unsigned char*)mxGetData(prhs[0]));
            int errcode = 0;

            if (inputsize > 0) {
                errcode = zmat_run(inputsize, inputstr, &outputsize, &outputbuf, zipid, &ret, flags.iscompress);
            }

            if (errcode < 0) {
                if (outputbuf) {
                    free(outputbuf);
                }

                outputbuf = NULL;
                outputsize = 0;
            }

            buflen[0] = 1;
            buflen[1] = outputsize;

            if (use4bytedim) {
                unsigned int intdims[4] = {0};
                intdims[0] = 1;
                intdims[1] = (unsigned int)outputsize;
                plhs[0] = mxCreateNumericArray(2, (mwSize*)intdims, mxUINT8_CLASS, mxREAL);
            } else {
                plhs[0] = mxCreateNumericArray(2, buflen, mxUINT8_CLASS, mxREAL);
            }

            if (outputbuf) {
                memcpy((unsigned char*)mxGetPr(plhs[0]), outputbuf, buflen[1]);
                free(outputbuf);
            }

            if (nlhs > 1) {
                mwSize inputdim[2] = {1, 0}, *dims = (mwSize*)mxGetDimensions(prhs[0]);
                unsigned int* inputsize = NULL;
                plhs[1] = mxCreateStructMatrix(1, 1, 6, metadata);
                mxArray* val = mxCreateString(mxGetClassName(prhs[0]));
                mxSetFieldByNumber(plhs[1], 0, 0, val);

                inputdim[1] = mxGetNumberOfDimensions(prhs[0]);
                inputsize = (unsigned int*)malloc(inputdim[1] * sizeof(unsigned int));

                if (use4bytedim) {
                    unsigned int intinputdim[4] = {0}, *intdims = (unsigned int*)(mxGetDimensions(prhs[0]));
                    intinputdim[0] = 1;
                    intinputdim[1] = (unsigned int)inputdim[1];
                    val = mxCreateNumericArray(2, (mwSize*)intinputdim, mxUINT32_CLASS, mxREAL);

                    for (int i = 0; i < intinputdim[1]; i++) {
                        inputsize[i] = intdims[i];
                    }

                    memcpy(mxGetPr(val), inputsize, intinputdim[1]*sizeof(unsigned int));
                } else {
                    val = mxCreateNumericArray(2, inputdim, mxUINT32_CLASS, mxREAL);

                    for (int i = 0; i < inputdim[1]; i++) {
                        inputsize[i] = dims[i];
                    }

                    memcpy(mxGetPr(val), inputsize, inputdim[1]*sizeof(mwSize));
                }

                mxSetFieldByNumber(plhs[1], 0, 1, val);

                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = mxGetElementSize(prhs[0]);
                mxSetFieldByNumber(plhs[1], 0, 2, val);

                val = mxCreateString(zipmethods[zipid]);
                mxSetFieldByNumber(plhs[1], 0, 3, val);

                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = ret;
                mxSetFieldByNumber(plhs[1], 0, 4, val);

                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = flags.param.clevel;
                mxSetFieldByNumber(plhs[1], 0, 5, val);
            }

            if (errcode < 0) {
                mexWarnMsgTxt(zmat_error(-errcode));
            }
        } else {
            mexErrMsgTxt("the input must be a char, non-complex numerical or logical array");
        }
    } catch (const char* err) {
        mexPrintf("Error: %s\n", err);
    } catch (const std::exception& err) {
        mexPrintf("C++ Error: %s\n", err.what());
    } catch (...) {
        mexPrintf("Unknown Exception\n");
    }

    return;
}

/**
 * @brief Print a brief help information if nothing is provided
 */

void zmat_usage() {
    mexPrintf("ZMat (v0.9.9)\nUsage:\n\t[output,info]=zmat(input,iscompress,method);\n\nPlease run 'help zmat' for more details.\n");
}
