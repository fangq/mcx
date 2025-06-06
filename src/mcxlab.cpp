/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section sformat Formatting
**          Please always run "make pretty" inside the \c src folder before each commit.
**          The above command requires \c astyle to perform automatic formatting.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcxlab.cpp

@brief   mex function for MCXLAB
*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <exception>
#include <math.h>

#include "mex.h"
#include "mcx_const.h"
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_shapes.h"
#include "mcx_lang.h"
#include "mcx_lang.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#if defined(USE_XOROSHIRO128P_RAND)
    #define RAND_WORD_LEN 4
#elif defined(USE_POSIX_RAND)
    #define RAND_WORD_LEN 4
#else
    #define RAND_WORD_LEN 4       /**< number of Words per RNG state */
#endif

/**  Macro to read the 1st scalar cfg member */
#define GET_1ST_FIELD(x,y)  if(strcmp(name,#y)==0) {double *val=mxGetPr(item);x->y=val[0];printf("mcx.%s=%g;\n",#y,(float)(x->y));}

/**  Macro to read one scalar cfg member */
#define GET_ONE_FIELD(x,y)  else GET_1ST_FIELD(x,y)

/**  Macro to read one 3-element vector member of cfg */
#define GET_VEC3_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];\
        printf("mcx.%s=[%g %g %g];\n",#v,(float)(u->v.x),(float)(u->v.y),(float)(u->v.z));}

/**  Macro to read one 3- or 4-element vector member of cfg */
#define GET_VEC34_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];if(mxGetNumberOfElements(item)==4) u->v.w=val[3];\
        printf("mcx.%s=[%g %g %g %g];\n",#v,(float)(u->v.x),(float)(u->v.y),(float)(u->v.z),(float)(u->v.w));}

/**  Macro to read one 4-element vector member of cfg */
#define GET_VEC4_FIELD(u,v) else if(strcmp(name,#v)==0) {double *val=mxGetPr(item);u->v.x=val[0];u->v.y=val[1];u->v.z=val[2];u->v.w=val[3];\
        printf("mcx.%s=[%g %g %g %g];\n",#v,(float)(u->v.x),(float)(u->v.y),(float)(u->v.z),(float)(u->v.w));}

/**  Macro to output GPU parameters as field */
#define SET_GPU_INFO(output,id,v)  mxSetField(output,id,#v,mxCreateDoubleScalar(gpuinfo[i].v));

typedef mwSize dimtype;

extern cJSON* mcx_lang;

void mcx_set_field(const mxArray* root, const mxArray* item, int idx, Config* cfg);
void mcxlab_usage();

float* detps = NULL;       //! buffer to receive data from cfg.detphotons field
int    dimdetps[2] = {0, 0}; //! dimensions of the cfg.detphotons array
int    seedbyte = 0;

/** @brief Mex function for the MCX host function for MATLAB/Octave
 *  This is the master function to interface all MCX features inside MATLAB.
 *  In MCXLAB, all inputs are read from the cfg structure, which contains all
 *  simuation parameters and data.
 */

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    Config cfg;
    GPUInfo* gpuinfo = NULL;
    mxArray*    tmp;
    int        ifield, jstruct;
    int        ncfg, nfields;
    dimtype    fielddim[6];
    int        activedev = 0;
    int        errorflag = 0;
    int        threadid = 0;
    const char*       outputtag[] = {"data"};
    const char*       datastruct[] = {"data", "stat", "dref", "prop"};
    const char*       statstruct[] = {"runtime", "nphoton", "energytot", "energyabs", "normalizer", "unitinmm", "workload"};
    const char*       gpuinfotag[] = {"name", "id", "devcount", "major", "minor", "globalmem",
                                      "constmem", "sharedmem", "regcount", "clock", "sm", "core",
                                      "autoblock", "autothread", "maxgate"
                                     };

    /**
     * If no input is given for this function, it prints help information and return.
     */
    if (nrhs == 0) {
        mcxlab_usage();
        return;
    }

    /**
     * If a single string is passed, and if this string is 'gpuinfo', this function
     * returns the list of GPUs on this host and return.
     */
    if (nrhs == 1 && mxIsChar(prhs[0])) {
        char shortcmd[MAX_SESSION_LENGTH];
        mxGetString(prhs[0], shortcmd, MAX_SESSION_LENGTH);
        shortcmd[MAX_SESSION_LENGTH - 1] = '\0';

        if (strcmp(shortcmd, "gpuinfo") == 0) {
            mcx_initcfg(&cfg);
            cfg.isgpuinfo = 3;

            if (!(activedev = mcx_list_gpu(&cfg, &gpuinfo))) {
                mexErrMsgTxt("no active GPU device found");
            }

            plhs[0] = mxCreateStructMatrix(gpuinfo[0].devcount, 1, 15, gpuinfotag);

            for (int i = 0; i < gpuinfo[0].devcount; i++) {
                mxSetField(plhs[0], i, "name", mxCreateString(gpuinfo[i].name));
                SET_GPU_INFO(plhs[0], i, id);
                SET_GPU_INFO(plhs[0], i, devcount);
                SET_GPU_INFO(plhs[0], i, major);
                SET_GPU_INFO(plhs[0], i, minor);
                SET_GPU_INFO(plhs[0], i, globalmem);
                SET_GPU_INFO(plhs[0], i, constmem);
                SET_GPU_INFO(plhs[0], i, sharedmem);
                SET_GPU_INFO(plhs[0], i, regcount);
                SET_GPU_INFO(plhs[0], i, clock);
                SET_GPU_INFO(plhs[0], i, sm);
                SET_GPU_INFO(plhs[0], i, core);
                SET_GPU_INFO(plhs[0], i, autoblock);
                SET_GPU_INFO(plhs[0], i, autothread);
                SET_GPU_INFO(plhs[0], i, maxgate);
            }

            mcx_cleargpuinfo(&gpuinfo);
            mcx_clearcfg(&cfg);
        } else if (strcmp(shortcmd, "version") == 0) {
            mcx_initcfg(&cfg);
            mcx_printheader(&cfg);
            mcx_clearcfg(&cfg);
            plhs[0] = mxCreateString(MCX_VERSION);
        }

        return;
    }

    /**
     * If a structure is passed to this function, a simulation will be launched.
     */
    printf("Launching MCXLAB - Monte Carlo eXtreme for MATLAB & GNU Octave ...\n");

    if (!mxIsStruct(prhs[0])) {
        mexErrMsgTxt("Input must be a structure.");
    }

    /**
     * Find out information about input and output.
     */
    nfields = mxGetNumberOfFields(prhs[0]); /** how many subfield in the input cfg data structure */
    ncfg = mxGetNumberOfElements(prhs[0]);  /** if input is a struct array, each element of the struct is a simulation */

    /**
     * The function can return 1-5 outputs (i.e. the LHS)
     */
    if (nlhs >= 1 || (cfg.debuglevel & MCX_DEBUG_MOVE_ONLY)) {
        plhs[0] = mxCreateStructMatrix(ncfg, 1, 4, datastruct);
    }

    if (nlhs >= 2) {
        plhs[1] = mxCreateStructMatrix(ncfg, 1, 1, outputtag);
    }

    if (nlhs >= 3) {
        plhs[2] = mxCreateStructMatrix(ncfg, 1, 1, outputtag);
    }

    if (nlhs >= 4) {
        plhs[3] = mxCreateStructMatrix(ncfg, 1, 1, outputtag);
    }

    if (nlhs >= 5) {
        plhs[4] = mxCreateStructMatrix(ncfg, 1, 1, outputtag);
    }

    /**
     * Loop over each element of the struct if it is an array, each element is a simulation
     */
    for (jstruct = 0; jstruct < ncfg; jstruct++) {  /* how many configs */
        try {
            unsigned int partialdata, hostdetreclen, debuglen;
            printf("Running simulations for configuration #%d ...\n", jstruct + 1);

            /** Initialize cfg with default values first */
            mcx_initcfg(&cfg);
            detps = NULL;

            /** Read each struct element from input and set value to the cfg configuration */
            for (ifield = 0; ifield < nfields; ifield++) { /* how many input struct fields */
                tmp = mxGetFieldByNumber(prhs[0], jstruct, ifield);

                if (tmp == NULL) {
                    continue;
                }

                mcx_set_field(prhs[0], tmp, ifield, &cfg);
            }

            mcx_flush(&cfg);

            /** Overwite the output flags using the number of output present */
            if (nlhs < 1) {
                cfg.issave2pt = 0;    /** issave2pt defualt is 1, but allow users to manually disable, auto disable only if there is no output */
            }

            cfg.issavedet = (nlhs >= 2 && cfg.issavedet == 0) ? 1 : ((nlhs < 2) ? 0 : cfg.issavedet); /** save detected photon data to the 2nd output if present */
            cfg.issaveseed = (nlhs >= 4); /** save detected photon seeds to the 4th output if present */

            if (nlhs >= 5 || (cfg.debuglevel & MCX_DEBUG_MOVE_ONLY)) {
                cfg.debuglevel |= MCX_DEBUG_MOVE;
            }

            /** Validate all input fields, and warn incompatible inputs */
            mcx_validatecfg(&cfg, detps, dimdetps, seedbyte);

            debuglen = MCX_DEBUG_REC_LEN + (cfg.istrajstokes << 2);

            partialdata = (cfg.medianum - 1) * (SAVE_NSCAT(cfg.savedetflag) + SAVE_PPATH(cfg.savedetflag) + SAVE_MOM(cfg.savedetflag));
            hostdetreclen = partialdata + SAVE_DETID(cfg.savedetflag) + 3 * (SAVE_PEXIT(cfg.savedetflag) + SAVE_VEXIT(cfg.savedetflag)) + SAVE_W0(cfg.savedetflag) + 4 * SAVE_IQUV(cfg.savedetflag);

            /** One must define the domain and properties */
            if (cfg.vol == NULL || cfg.medianum == 0) {
                mexErrMsgTxt("You must define 'vol' and 'prop' field.");
            }

            /** One must also choose one of the GPUs */
            if (!(activedev = mcx_list_gpu(&cfg, &gpuinfo))) {
                mexErrMsgTxt("No active GPU device found");
            }

            /** Initialize all buffers necessary to store the output variables */
            if (nlhs >= 1 && cfg.issave2pt) {
                size_t fieldlen = cfg.dim.x * cfg.dim.y * cfg.dim.z * (int)((cfg.tend - cfg.tstart) / cfg.tstep + 0.5) * cfg.srcnum;

                if (cfg.replay.seed != NULL && cfg.replaydet == -1) {
                    fieldlen *= cfg.detnum;
                }

                if (cfg.replay.seed != NULL && (cfg.outputtype == otRF || cfg.outputtype == otRFmus)) {
                    fieldlen *= 2;
                }

                if (cfg.extrasrclen && cfg.srcid == -1) {
                    fieldlen *= (cfg.extrasrclen + 1);
                }

                cfg.exportfield = (float*)calloc(fieldlen, sizeof(float));
            }

            if (nlhs >= 2) {
                cfg.exportdetected = (float*)malloc(hostdetreclen * cfg.maxdetphoton * sizeof(float));
            }

            if (nlhs >= 4) {
                cfg.seeddata = malloc(cfg.maxdetphoton * sizeof(float) * RAND_WORD_LEN);
            }

            if (nlhs >= 5 || (cfg.debuglevel & MCX_DEBUG_MOVE_ONLY)) {
                cfg.exportdebugdata = (float*)malloc(cfg.maxjumpdebug * sizeof(float) * debuglen);
                cfg.debuglevel |= MCX_DEBUG_MOVE;
            }

            /** Start multiple threads, one thread to run portion of the simulation on one CUDA GPU, all in parallel */
#ifdef _OPENMP
            omp_set_num_threads(activedev);
            #pragma omp parallel shared(errorflag)
            {
                threadid = omp_get_thread_num();
#endif

                /** Enclose all simulation calls inside a try/catch construct for exception handling */
                try {
                    /** Call the main simulation host function to start the simulation */
                    mcx_run_simulation(&cfg, gpuinfo);

                } catch (const char* err) {
                    mexPrintf("Error from thread (%d): %s\n", threadid, err);
                    errorflag++;
                } catch (const std::exception& err) {
                    mexPrintf("C++ Error from thread (%d): %s\n", threadid, err.what());
                    errorflag++;
                } catch (...) {
                    mexPrintf("Unknown Exception from thread (%d)", threadid);
                    errorflag++;
                }

#ifdef _OPENMP
            }
#endif

            /** If error is detected, gracefully terminate the mex and return back to MATLAB */
            if (errorflag) {
                mexErrMsgTxt("MCXLAB Terminated due to an exception!");
            }

            fielddim[4] = 1;
            fielddim[5] = 1;

            /** if 5th output presents, output the photon trajectory data */
            if (nlhs >= 5 || (cfg.debuglevel & MCX_DEBUG_MOVE_ONLY)) {
                int outputidx = (cfg.debuglevel & MCX_DEBUG_MOVE_ONLY) ? 0 : 4;
                fielddim[0] = debuglen;
                fielddim[1] = cfg.debugdatalen; // his.savedphoton is for one repetition, should correct
                fielddim[2] = 0;
                fielddim[3] = 0;
                mxSetFieldByNumber(plhs[outputidx], jstruct, 0, mxCreateNumericArray(2, fielddim, mxSINGLE_CLASS, mxREAL));

                if (cfg.debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) {
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[outputidx], jstruct, 0)), cfg.exportdebugdata, fielddim[0]*fielddim[1]*sizeof(float));
                }

                if (cfg.exportdebugdata) {
                    free(cfg.exportdebugdata);
                }

                cfg.exportdebugdata = NULL;
            }

            /** if the 4th output presents, output the detected photon seeds */
            if (nlhs >= 4) {
                fielddim[0] = (cfg.issaveseed > 0) * RAND_WORD_LEN * sizeof(float);
                fielddim[1] = cfg.detectedcount; // his.savedphoton is for one repetition, should correct
                fielddim[2] = 0;
                fielddim[3] = 0;
                mxSetFieldByNumber(plhs[3], jstruct, 0, mxCreateNumericArray(2, fielddim, mxUINT8_CLASS, mxREAL));
                memcpy((unsigned char*)mxGetPr(mxGetFieldByNumber(plhs[3], jstruct, 0)), cfg.seeddata, fielddim[0]*fielddim[1]);
                free(cfg.seeddata);
                cfg.seeddata = NULL;
            }

            /** if the 3rd output presents, output the detector-masked medium volume, similar to the --dumpmask flag */
            if (nlhs >= 3) {
                fielddim[0] = cfg.dim.x;
                fielddim[1] = cfg.dim.y;
                fielddim[2] = cfg.dim.z;
                fielddim[3] = 0;

                if (cfg.vol) {
                    mxSetFieldByNumber(plhs[2], jstruct, 0, mxCreateNumericArray(3, fielddim, mxUINT32_CLASS, mxREAL));
                    memcpy((unsigned char*)mxGetPr(mxGetFieldByNumber(plhs[2], jstruct, 0)), cfg.vol,
                           fielddim[0]*fielddim[1]*fielddim[2]*sizeof(unsigned int));
                }
            }

            /** if the 2nd output presents, output the detected photon partialpath data */
            if (nlhs >= 2) {
                fielddim[0] = hostdetreclen;
                fielddim[1] = cfg.detectedcount;
                fielddim[2] = 0;
                fielddim[3] = 0;

                if (cfg.detectedcount > 0) {
                    mxSetFieldByNumber(plhs[1], jstruct, 0, mxCreateNumericArray(2, fielddim, mxSINGLE_CLASS, mxREAL));
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[1], jstruct, 0)), cfg.exportdetected,
                           fielddim[0]*fielddim[1]*sizeof(float));
                }

                free(cfg.exportdetected);
                cfg.exportdetected = NULL;
            }

            /** if the 1st output presents, output the fluence/energy-deposit volume data */
            if (nlhs >= 1) {
                size_t fieldlen;
                fielddim[0] = cfg.srcnum * cfg.dim.x;
                fielddim[1] = cfg.dim.y;
                fielddim[2] = cfg.dim.z;
                fielddim[3] = (int)((cfg.tend - cfg.tstart) / cfg.tstep + 0.5);

                if (cfg.replay.seed != NULL && cfg.replaydet == -1) {
                    fielddim[4] = cfg.detnum;
                }

                if (cfg.replay.seed != NULL && (cfg.outputtype == otRF || cfg.outputtype == otRFmus)) {
                    fielddim[5] = 2;
                }

                if (cfg.extrasrclen && cfg.srcid == -1) {
                    fielddim[5] *= (cfg.extrasrclen + 1);
                }

                fieldlen = (size_t)fielddim[0] * fielddim[1] * fielddim[2] * fielddim[3] * fielddim[4] * fielddim[5];

                if (cfg.issaveref && cfg.exportfield) {
                    int highdim = fielddim[3] * fielddim[4] * fielddim[5];
                    size_t voxellen = cfg.dim.x * cfg.dim.y * cfg.dim.z;
                    float* dref = (float*)malloc(fieldlen * sizeof(float));

                    memcpy(dref, cfg.exportfield, fieldlen * sizeof(float));

                    for (size_t voxelid = 0; voxelid < voxellen; voxelid++) {
                        if (cfg.vol[voxelid]) {
                            for (int gate = 0; gate < highdim; gate++)
                                for (int srcid = 0; srcid < cfg.srcnum; srcid++) {
                                    dref[(gate * voxellen + voxelid) * cfg.srcnum + srcid] = 0.f;
                                }
                        } else {
                            for (int gate = 0; gate < highdim; gate++)
                                for (int srcid = 0; srcid < cfg.srcnum; srcid++) {
                                    dref[(gate * voxellen + voxelid) * cfg.srcnum + srcid] = -dref[(gate * voxellen + voxelid) * cfg.srcnum + srcid];
                                    cfg.exportfield[(gate * voxellen + voxelid) * cfg.srcnum + srcid] = 0.f;
                                }
                        }
                    }

                    mxSetFieldByNumber(plhs[0], jstruct, 2, mxCreateNumericArray(((fielddim[5] > 1) ? 6 : (4 + (fielddim[4] > 1))), fielddim, mxSINGLE_CLASS, mxREAL));
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[0], jstruct, 2)), dref, fieldlen * sizeof(float));
                    free(dref);
                }

                if (cfg.issave2pt && cfg.exportfield) {
                    mxSetFieldByNumber(plhs[0], jstruct, 0, mxCreateNumericArray(((fielddim[5] > 1) ? 6 : (4 + (fielddim[4] > 1))), fielddim, mxSINGLE_CLASS, mxREAL));
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[0], jstruct, 0)), cfg.exportfield,
                           fieldlen * sizeof(float));
                }

                if (cfg.exportfield) {
                    free(cfg.exportfield);
                }

                cfg.exportfield = NULL;

                /** also return the run-time info in outut.runtime */
                mxArray* stat = mxCreateStructMatrix(1, 1, 7, statstruct);
                mxArray* val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.runtime;
                mxSetFieldByNumber(stat, 0, 0, val);

                /** return the total simulated photon number */
                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.nphoton * ((cfg.respin > 1) ? (cfg.respin) : 1);
                mxSetFieldByNumber(stat, 0, 1, val);

                /** return the total simulated energy */
                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.energytot;
                mxSetFieldByNumber(stat, 0, 2, val);

                /** return the total absorbed energy */
                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.energyabs;
                mxSetFieldByNumber(stat, 0, 3, val);

                /** return the normalization factor */
                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.normalizer;
                mxSetFieldByNumber(stat, 0, 4, val);

                /** return the voxel size unitinmm */
                val = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(val) = cfg.unitinmm;
                mxSetFieldByNumber(stat, 0, 5, val);

                /** return the relative workload between multiple GPUs */
                val = mxCreateDoubleMatrix(1, activedev, mxREAL);

                for (int i = 0; i < activedev; i++) {
                    *(mxGetPr(val) + i) = cfg.workload[i];
                }

                mxSetFieldByNumber(stat, 0, 6, val);

                mxSetFieldByNumber(plhs[0], jstruct, 1, stat);

                /** return the final optical properties for polarized MCX simulation */
                if (cfg.polprop) {
                    for (int i = 0; i < cfg.polmedianum; i++) {
                        // restore mua and mus values
                        cfg.prop[i + 1].mua /= cfg.unitinmm;
                        cfg.prop[i + 1].mus /= cfg.unitinmm;
                    }

                    dimtype propdim[2] = {4, 0};
                    propdim[1] = (dimtype)cfg.medianum;
                    mxSetFieldByNumber(plhs[0], jstruct, 3, mxCreateNumericArray(2, propdim, mxSINGLE_CLASS, mxREAL));
                    memcpy((float*)mxGetPr(mxGetFieldByNumber(plhs[0], jstruct, 3)), cfg.prop, cfg.medianum * 4 * sizeof(float));
                }
            }
        } catch (const char* err) {
            mexPrintf("Error: %s\n", err);
        } catch (const std::exception& err) {
            mexPrintf("C++ Error: %s\n", err.what());
        } catch (...) {
            mexPrintf("Unknown Exception");
        }

        /** Clear up simulation data structures by calling the destructors */
        if (detps) {
            free(detps);
        }

        mcx_cleargpuinfo(&gpuinfo);
        mcx_clearcfg(&cfg);
    }

    return;
}


/**
 * @brief Function to parse one subfield of the input structure
 *
 * This function reads in all necessary information from the cfg input structure.
 * it can handle single scalar inputs, short vectors (3-4 elem), strings and arrays.
 *
 * @param[in] root: the cfg input data structure
 * @param[in] item: the current element of the cfg input data structure
 * @param[in] idx: the index of the current element (starting from 0)
 * @param[out] cfg: the simulation configuration structure to store all input read from the parameters
 */

void mcx_set_field(const mxArray* root, const mxArray* item, int idx, Config* cfg) {
    const char* name = mxGetFieldNameByNumber(root, idx);
    const dimtype* arraydim;
    int i, j;

    if (strcmp(name, "nphoton") == 0 && cfg->replay.seed != NULL) {
        return;
    }

    cfg->flog = stderr;
    GET_1ST_FIELD(cfg, nphoton)
    GET_ONE_FIELD(cfg, nblocksize)
    GET_ONE_FIELD(cfg, nthread)
    GET_ONE_FIELD(cfg, tstart)
    GET_ONE_FIELD(cfg, tstep)
    GET_ONE_FIELD(cfg, tend)
    GET_ONE_FIELD(cfg, maxdetphoton)
    GET_ONE_FIELD(cfg, sradius)
    GET_ONE_FIELD(cfg, maxgate)
    GET_ONE_FIELD(cfg, respin)
    GET_ONE_FIELD(cfg, isreflect)
    GET_ONE_FIELD(cfg, isref3)
    GET_ONE_FIELD(cfg, isrefint)
    GET_ONE_FIELD(cfg, isnormalized)
    GET_ONE_FIELD(cfg, isgpuinfo)
    GET_ONE_FIELD(cfg, issrcfrom0)
    GET_ONE_FIELD(cfg, autopilot)
    GET_ONE_FIELD(cfg, minenergy)
    GET_ONE_FIELD(cfg, unitinmm)
    GET_ONE_FIELD(cfg, printnum)
    GET_ONE_FIELD(cfg, voidtime)
    GET_ONE_FIELD(cfg, issavedet)
    GET_ONE_FIELD(cfg, issaveseed)
    GET_ONE_FIELD(cfg, issaveref)
    GET_ONE_FIELD(cfg, issaveexit)
    GET_ONE_FIELD(cfg, ismomentum)
    GET_ONE_FIELD(cfg, isspecular)
    GET_ONE_FIELD(cfg, istrajstokes)
    GET_ONE_FIELD(cfg, replaydet)
    GET_ONE_FIELD(cfg, faststep)
    GET_ONE_FIELD(cfg, maxvoidstep)
    GET_ONE_FIELD(cfg, maxjumpdebug)
    GET_ONE_FIELD(cfg, gscatter)
    GET_ONE_FIELD(cfg, srcnum)
    GET_ONE_FIELD(cfg, srcid)
    GET_ONE_FIELD(cfg, omega)
    GET_ONE_FIELD(cfg, issave2pt)
    GET_ONE_FIELD(cfg, lambda)
    GET_VEC3_FIELD(cfg, steps)
    GET_VEC3_FIELD(cfg, crop0)
    GET_VEC3_FIELD(cfg, crop1)
    GET_VEC4_FIELD(cfg, srciquv)
    else if (strcmp(name, "srcpos") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] == 0 || arraydim[1] < 3 || arraydim[1] > 4) {
            mexErrMsgTxt("the 'srcpos' field must have 3 or 4 columns (x,y,z,w0)");
        }

        double* val = mxGetPr(item);

        for (i = 0; i < arraydim[1]; i++) {
            ((float*)(&cfg->srcpos.x))[i] = val[i * arraydim[0]];
        }

        printf("mcx.srcpos=[%g %g %g %g];\n", cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z, cfg->srcpos.w);

        if (arraydim[0] == 1 && cfg->extrasrclen == 0) {
            return;
        }

        if (cfg->extrasrclen && cfg->extrasrclen != arraydim[0] - 1) {
            mexErrMsgTxt("Length of sub-elements of srcpos/srcdir/srcparam1/srcparam2 must match");
        } else {
            cfg->extrasrclen = arraydim[0] - 1;
        }

        if (cfg->srcdata == NULL) {
            cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
        }

        for (j = 0; j < arraydim[1]; j++)
            for (i = 0; i < cfg->extrasrclen; i++) {
                ((float*)(&cfg->srcdata[i].srcpos.x))[j] = val[j * arraydim[0] + i + 1];
            }

        printf("mcx.extrasrclen=%d;\n", cfg->extrasrclen);
    } else if (strcmp(name, "srcdir") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] == 0 || arraydim[1] < 3 || arraydim[1] > 4) {
            mexErrMsgTxt("the 'srcdir' field must have 3 or 4 columns (vx,vy,vz,focallength)");
        }

        double* val = mxGetPr(item);

        for (i = 0; i < arraydim[1]; i++) {
            ((float*)(&cfg->srcdir.x))[i] = val[i * arraydim[0]];
        }

        printf("mcx.srcdir=[%g %g %g %g];\n", cfg->srcdir.x, cfg->srcdir.y, cfg->srcdir.z, cfg->srcdir.w);

        if (arraydim[0] == 1 && cfg->extrasrclen == 0) {
            return;
        }

        if (cfg->extrasrclen && cfg->extrasrclen != arraydim[0] - 1) {
            mexErrMsgTxt("Length of sub-elements of srcdir/srcdir/srcparam1/srcparam2 must match");
        } else {
            cfg->extrasrclen = arraydim[0] - 1;
        }

        if (cfg->srcdata == NULL) {
            cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
        }

        for (j = 0; j < arraydim[1]; j++)
            for (i = 0; i < cfg->extrasrclen; i++) {
                ((float*)(&cfg->srcdata[i].srcdir.x))[j] = val[j * arraydim[0] + i + 1];
            }

        printf("mcx.extrasrclen=%d;\n", cfg->extrasrclen);
    } else if (strcmp(name, "srcparam1") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] == 0 || (arraydim[1] == 0 && arraydim[1] > 4)) {
            mexErrMsgTxt("the 'srcparam1' field must have 1-4 columns");
        }

        double* val = mxGetPr(item);

        for (i = 0; i < arraydim[1]; i++) {
            ((float*)(&cfg->srcparam1.x))[i] = val[i * arraydim[0]];
        }

        printf("mcx.srcparam1=[%g %g %g %g];\n", cfg->srcparam1.x, cfg->srcparam1.y, cfg->srcparam1.z, cfg->srcparam1.w);

        if (arraydim[0] == 1 && cfg->extrasrclen == 0) {
            return;
        }

        if (cfg->extrasrclen && cfg->extrasrclen != arraydim[0] - 1) {
            mexErrMsgTxt("Length of sub-elements of srcparam1/srcparam1/srcparam1/srcparam2 must match");
        } else {
            cfg->extrasrclen = arraydim[0] - 1;
        }

        if (cfg->srcdata == NULL) {
            cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
        }

        for (j = 0; j < arraydim[1]; j++)
            for (i = 0; i < cfg->extrasrclen; i++) {
                ((float*)(&cfg->srcdata[i].srcparam1.x))[j] = val[j * arraydim[0] + i + 1];
            }

        printf("mcx.extrasrclen=%d;\n", cfg->extrasrclen);
    } else if (strcmp(name, "srcparam2") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] == 0 || (arraydim[1] == 0 && arraydim[1] > 4)) {
            mexErrMsgTxt("the 'srcparam2' field must have 1-4 columns");
        }

        double* val = mxGetPr(item);

        for (i = 0; i < arraydim[1]; i++) {
            ((float*)(&cfg->srcparam2.x))[i] = val[i * arraydim[0]];
        }

        printf("mcx.srcparam2=[%g %g %g %g];\n", cfg->srcparam2.x, cfg->srcparam2.y, cfg->srcparam2.z, cfg->srcparam2.w);

        if (arraydim[0] == 1 && cfg->extrasrclen == 0) {
            return;
        }

        if (cfg->extrasrclen && cfg->extrasrclen != arraydim[0] - 1) {
            mexErrMsgTxt("Length of sub-elements of srcparam2/srcparam2/srcparam2/srcparam2 must match");
        } else {
            cfg->extrasrclen = arraydim[0] - 1;
        }

        if (cfg->srcdata == NULL) {
            cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
        }

        for (j = 0; j < arraydim[1]; j++)
            for (i = 0; i < cfg->extrasrclen; i++) {
                ((float*)(&cfg->srcdata[i].srcparam2.x))[j] = val[j * arraydim[0] + i + 1];
            }

        printf("mcx.extrasrclen=%d;\n", cfg->extrasrclen);
    } else if (strcmp(name, "vol") == 0) {
        dimtype dimxyz;
        cfg->mediabyte = 0;
        arraydim = mxGetDimensions(item);

        if (mxGetNumberOfDimensions(item) <= 3) {
            if (mxIsUint8(item) || mxIsInt8(item)) { // input is a 3D byte array
                cfg->mediabyte = 1;
            } else if (mxIsUint16(item) || mxIsInt16(item)) { // input is a 3D short array
                cfg->mediabyte = 2;
            } else if (mxIsUint32(item) || mxIsInt32(item)) { // input is a 3D integer array
                cfg->mediabyte = 4;
            } else if (mxIsDouble(item)) { // input is a 3D double array
                cfg->mediabyte = 8;
            } else if (mxIsSingle(item)) { // input is a float32 array
                cfg->mediabyte = 14;
            }

            for (i = 0; i < mxGetNumberOfDimensions(item); i++) {
                ((unsigned int*)(&cfg->dim))[i] = arraydim[i];
            }

            if (i < 3) {
                cfg->dim.z = 1;
            }

        } else if (mxGetNumberOfDimensions(item) == 4) { // if dimension is 4D, 1st dim is the property records: mua/mus/g/n
            if ((mxIsUint8(item) || mxIsInt8(item)) && arraydim[0] == 4) { // if 4D byte array has a 1st dim of 4
                cfg->mediabyte = MEDIA_ASGN_BYTE;
            } else if (mxIsSingle(item) && arraydim[0] == 4) { // if 4D float array has a 1st dim of 4
                cfg->mediabyte = MEDIA_ASGN_F2H;
            } else if ((mxIsUint8(item) || mxIsInt8(item)) && arraydim[0] == 8) {
                cfg->mediabyte = MEDIA_2LABEL_SPLIT;
            } else if (mxIsSingle(item) && arraydim[0] == 3) {
                cfg->mediabyte = MEDIA_LABEL_HALF;
            } else if ((mxIsUint16(item) || mxIsInt16(item)) && arraydim[0] == 3) {
                cfg->mediabyte = MEDIA_2LABEL_MIX;
            } else if ((mxIsUint16(item) || mxIsInt16(item)) && arraydim[0] == 2) { // if 4D short array has a 1st dim of 2
                cfg->mediabyte = MEDIA_AS_SHORT;
            } else if (mxIsSingle(item) && arraydim[0] == 2) { // if 4D float32 array has a 1st dim of 2
                cfg->mediabyte = MEDIA_AS_F2H;
            } else if (mxIsSingle(item) && arraydim[0] == 1) { // if 4D byte array has a 1st dim of 1
                cfg->mediabyte = MEDIA_MUA_FLOAT;
            }

            for (i = 0; i < 3; i++) {
                ((unsigned int*)(&cfg->dim))[i] = arraydim[i + 1];
            }
        }

        if (cfg->mediabyte == 0) {
            mexErrMsgTxt("the 'vol' field must be a 3D or 4D array");
        }

        dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z;

        if (cfg->vol) {
            free(cfg->vol);
        }

        cfg->vol = static_cast<unsigned int*>(malloc(dimxyz * sizeof(unsigned int)));

        if (cfg->mediabyte == 4 || (cfg->mediabyte > 100 && cfg->mediabyte != MEDIA_MUA_FLOAT)) {
            memcpy(cfg->vol, mxGetData(item), dimxyz * sizeof(unsigned int));
        } else {
            if (cfg->mediabyte == 1) {
                unsigned char* val = (unsigned char*)mxGetPr(item);

                for (i = 0; i < dimxyz; i++) {
                    cfg->vol[i] = val[i];
                }
            } else if (cfg->mediabyte == 2) {
                unsigned short* val = (unsigned short*)mxGetPr(item);

                for (i = 0; i < dimxyz; i++) {
                    cfg->vol[i] = val[i];
                }
            } else if (cfg->mediabyte == 8) {
                double* val = (double*)mxGetPr(item);

                for (i = 0; i < dimxyz; i++) {
                    cfg->vol[i] = val[i];
                }

                cfg->mediabyte = 4;
            } else if (cfg->mediabyte == 14) {
                float* val = (float*)mxGetPr(item);

                for (i = 0; i < dimxyz; i++) {
                    cfg->vol[i] = val[i];
                }
            } else if (cfg->mediabyte == MEDIA_MUA_FLOAT) {
                union {
                    float f;
                    uint  i;
                } f2i;
                float* val = (float*)mxGetPr(item);

                for (i = 0; i < dimxyz; i++) {
                    f2i.f = val[i] * cfg->unitinmm;

                    if (f2i.i == 0) { /*avoid being detected as a 0-label voxel*/
                        f2i.f = EPS;
                    }

                    if (val[i] != val[i] || f2i.i == SIGN_BIT) { /*if input is nan in continuous medium, convert to 0-voxel*/
                        f2i.i = 0;
                    }

                    cfg->vol[i] = f2i.i;
                }
            } else if (cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_ASGN_F2H) {
                float* val = (float*)mxGetPr(item);
                float f2h[2];
                int offset = (cfg->mediabyte == MEDIA_ASGN_F2H);

                if (cfg->mediabyte == MEDIA_ASGN_F2H) {
                    cfg->vol = static_cast<unsigned int*>(realloc(cfg->vol, dimxyz * 2 * sizeof(unsigned int)));
                }

                for (i = 0; i < dimxyz; i++) {
                    f2h[0] = val[i << (1 + offset)] * cfg->unitinmm;       // mua
                    f2h[1] = val[(i << (1 + offset)) + 1] * cfg->unitinmm; // mus

                    if (f2h[0] != f2h[0] || f2h[1] != f2h[1]) { /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
                        cfg->vol[i] = 0;
                        continue;
                    }

                    if (cfg->mediabyte == MEDIA_ASGN_F2H) {
                        cfg->vol[i] = mcx_float2half2(f2h);
                        f2h[0] = val[(i << 2) + 2];   // g
                        f2h[1] = val[(i << 2) + 3];   // n
                        cfg->vol[i + dimxyz] = mcx_float2half2(f2h);
                    } else {
                        cfg->vol[i] = mcx_float2half2(f2h);
                    }
                }
            } else if (cfg->mediabyte == MEDIA_LABEL_HALF) {
                float* val = (float*)mxGetPr(item);
                union {
                    float f[3];
                    unsigned int i[3];
                    unsigned short h[2];
                    unsigned char c[4];
                } f2bh;
                unsigned short tmp;

                for (i = 0; i < dimxyz; i++) {
                    f2bh.f[2] = val[i * 3];        // mua/mus/g or n float-value, depending on f[1]
                    f2bh.f[1] = val[i * 3 + 1];    // if it has value 1-4, it replaces the 1st (mua), 2nd (mus), 3rd (g), 4th (n) value of the label by f[2]
                    f2bh.f[0] = val[i * 3 + 2];    // voxel label

                    if (f2bh.f[1] < 0.f || f2bh.f[1] >= 4.f || f2bh.f[0] < 0.f ) {
                        mexErrMsgTxt("the 2nd volume must have an integer value between 0 and 3");
                    }

                    if (f2bh.f[1] >= 1.f && f2bh.f[1] <= 2.f) { // if the values are mua or mus, scale by cfg->unitinmm
                        f2bh.f[2] *= cfg->unitinmm;
                    }

                    f2bh.h[0] = ( (((unsigned char)(f2bh.f[1]) & 0x3) << 14) | (unsigned short)(f2bh.f[0]) );

                    f2bh.h[1] = (f2bh.i[2] >> 31) << 5;
                    tmp = (f2bh.i[2] >> 23) & 0xff;
                    tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
                    f2bh.h[1] = (f2bh.h[1] | tmp) << 10;
                    f2bh.h[1] |= (f2bh.i[2] >> 13) & 0x3ff;

                    cfg->vol[i] = f2bh.i[0];
                }
            } else if (cfg->mediabyte == MEDIA_2LABEL_MIX) {
                unsigned short* val = (unsigned short*)mxGetPr(item);
                union {
                    unsigned short h[2];
                    unsigned char  c[4];
                    unsigned int   i[1];
                } f2bh;
                unsigned short tmp;

                for (i = 0; i < dimxyz; i++) {
                    f2bh.c[0] = val[i * 3]   & 0xFF;     // label 1
                    f2bh.c[1] = val[i * 3 + 1] & 0xFF;   // label 2
                    f2bh.h[1] = val[i * 3 + 2] & 0x7FFF; // label 1 mixing-percentage scaled to 32767 (32767 means 100%)
                    cfg->vol[i] = f2bh.i[0];
                }
            } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
                unsigned char* val = (unsigned char*)mxGetPr(item);

                if (cfg->vol) {
                    free(cfg->vol);
                }

                cfg->vol = static_cast<unsigned int*>(malloc(dimxyz << 3));
                memcpy(cfg->vol, val, (dimxyz << 3));
            }
        }

        printf("mcx.dim=[%d %d %d];\n", cfg->dim.x, cfg->dim.y, cfg->dim.z);
        printf("mcx.mediabyte=%d;\n", cfg->mediabyte);
    } else if (strcmp(name, "detpos") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] > 0 && arraydim[1] != 4) {
            mexErrMsgTxt("the 'detpos' field must have 4 columns (x,y,z,radius)");
        }

        double* val = mxGetPr(item);
        cfg->detnum = arraydim[0];

        if (cfg->detpos) {
            free(cfg->detpos);
        }

        cfg->detpos = (float4*)malloc(cfg->detnum * sizeof(float4));

        for (j = 0; j < 4; j++)
            for (i = 0; i < cfg->detnum; i++) {
                ((float*)(&cfg->detpos[i]))[j] = val[j * cfg->detnum + i];
            }

        printf("mcx.detnum=%d;\n", cfg->detnum);
    } else if (strcmp(name, "prop") == 0) {
        arraydim = mxGetDimensions(item);

        if (arraydim[0] > 0 && arraydim[1] != 4) {
            mexErrMsgTxt("the 'prop' field must have 4 columns (mua,mus,g,n)");
        }

        double* val = mxGetPr(item);
        cfg->medianum = arraydim[0];

        if (cfg->prop) {
            free(cfg->prop);
        }

        cfg->prop = (Medium*)malloc(cfg->medianum * sizeof(Medium));

        for (j = 0; j < 4; j++)
            for (i = 0; i < cfg->medianum; i++) {
                ((float*)(&cfg->prop[i]))[j] = val[j * cfg->medianum + i];
            }

        printf("mcx.medianum=%d;\n", cfg->medianum);
    } else if (strcmp(name, "polprop") == 0) {
        if (mxGetNumberOfDimensions(item) != 2) {
            mexErrMsgTxt("the 'polprop' field must a 2D array");
        }

        arraydim = mxGetDimensions(item);

        if (arraydim[0] > 0 && arraydim[1] != 5) {
            mexErrMsgTxt("the 'polprop' field must have 5 columns (mua,radius,rho,n_sph,n_bkg");
        }

        double* val = mxGetPr(item);
        cfg->polmedianum = arraydim[0];

        if (cfg->polprop) {
            free(cfg->polprop);
        }

        cfg->polprop = (POLMedium*)malloc(cfg->polmedianum * sizeof(POLMedium));

        for (j = 0; j < 5; j++)
            for (i = 0; i < cfg->polmedianum; i++) {
                ((float*)(&cfg->polprop[i]))[j] = val[j * arraydim[0] + i];
            }

        printf("mcx.polmedianum=%d;\n", cfg->polmedianum);
    } else if (strcmp(name, "session") == 0) {
        int len = mxGetNumberOfElements(item);

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'session' field must be a non-empty string");
        }

        if (len > MAX_SESSION_LENGTH) {
            mexErrMsgTxt("the 'session' field is too long");
        }

        int status = mxGetString(item, cfg->session, MAX_SESSION_LENGTH);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        printf("mcx.session='%s';\n", cfg->session);
    } else if (strcmp(name, "srctype") == 0) {
        int len = mxGetNumberOfElements(item);
        const char* srctypeid[] = {"pencil", "isotropic", "cone", "gaussian", "planar",
                                   "pattern", "fourier", "arcsine", "disk", "fourierx", "fourierx2d", "zgaussian",
                                   "line", "slit", "pencilarray", "pattern3d", "hyperboloid", "ring", ""
                                  };
        char strtypestr[MAX_SESSION_LENGTH] = {'\0'};

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'srctype' field must be a non-empty string");
        }

        if (len > MAX_SESSION_LENGTH) {
            mexErrMsgTxt("the 'srctype' field is too long");
        }

        int status = mxGetString(item, strtypestr, MAX_SESSION_LENGTH);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        cfg->srctype = mcx_keylookup(strtypestr, srctypeid);

        if (cfg->srctype == -1) {
            mexErrMsgTxt("the specified source type is not supported");
        }

        printf("mcx.srctype='%s';\n", strtypestr);
    } else if (strcmp(name, "outputtype") == 0) {
        int len = mxGetNumberOfElements(item);
        const char* outputtype[] = {"flux", "fluence", "energy", "jacobian", "nscat", "wl", "wp", "wm", "rf", "length", "rfmus", "wltof", "wptof", ""};
        char outputstr[MAX_SESSION_LENGTH] = {'\0'};

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'outputtype' field must be a non-empty string");
        }

        if (len > MAX_SESSION_LENGTH) {
            mexErrMsgTxt("the 'outputtype' field is too long");
        }

        int status = mxGetString(item, outputstr, MAX_SESSION_LENGTH);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        cfg->outputtype = mcx_keylookup(outputstr, outputtype);

        if (cfg->outputtype >= 5) { // map wl to jacobian, wp to nscat
            cfg->outputtype -= 2;
        }

        if (cfg->outputtype == -1) {
            mexErrMsgTxt("the specified output type is not supported");
        }

        printf("mcx.outputtype='%s';\n", outputstr);
    } else if (strcmp(name, "debuglevel") == 0) {
        int len = mxGetNumberOfElements(item);
        const char debugflag[] = {'R', 'M', 'P', 'T', '\0'};
        char debuglevel[MAX_SESSION_LENGTH] = {'\0'};

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'debuglevel' field must be a non-empty string");
        }

        if (len > MAX_SESSION_LENGTH) {
            mexErrMsgTxt("the 'debuglevel' field is too long");
        }

        int status = mxGetString(item, debuglevel, MAX_SESSION_LENGTH);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        cfg->debuglevel = mcx_parsedebugopt(debuglevel, debugflag);

        if (cfg->debuglevel == 0) {
            mexWarnMsgTxt("the specified debuglevel is not supported");
        }

        printf("mcx.debuglevel=%d;\n", cfg->debuglevel);
    } else if (strcmp(name, "savedetflag") == 0) {
        int len = mxGetNumberOfElements(item);
        const char saveflag[] = {'D', 'S', 'P', 'M', 'X', 'V', 'W', 'I', '\0'};
        char savedetflag[MAX_SESSION_LENGTH] = {'\0'};

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'savedetflag' field must be a non-empty string");
        }

        if (len > MAX_SESSION_LENGTH) {
            mexErrMsgTxt("the 'savedetflag' field is too long");
        }

        int status = mxGetString(item, savedetflag, MAX_SESSION_LENGTH);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        cfg->savedetflag = mcx_parsedebugopt(savedetflag, saveflag);
        printf("mcx.savedetflag=%d;\n", cfg->savedetflag);
    } else if (strcmp(name, "srcpattern") == 0) {
        arraydim = mxGetDimensions(item);
        dimtype dimz = 1;

        if (mxGetNumberOfDimensions(item) >= 3) {
            dimz = arraydim[2];
        }

        if (mxGetNumberOfDimensions(item) == 4) {
            dimz *= arraydim[3];
        }

        double* val = mxGetPr(item);

        if (cfg->srcpattern) {
            free(cfg->srcpattern);
        }

        cfg->srcpattern = (float*)malloc(arraydim[0] * arraydim[1] * dimz * sizeof(float));

        for (i = 0; i < arraydim[0]*arraydim[1]*dimz; i++) {
            cfg->srcpattern[i] = val[i];
        }

        printf("mcx.srcpattern=[%ld %ld %ld];\n", arraydim[0], arraydim[1], dimz);
    } else if (strcmp(name, "invcdf") == 0) {
        dimtype nphase = mxGetNumberOfElements(item);
        double* val = mxGetPr(item);

        if (cfg->invcdf) {
            free(cfg->invcdf);
        }

        cfg->nphase = (unsigned int)nphase + 2;
        cfg->invcdf = (float*)calloc(cfg->nphase, sizeof(float));

        for (i = 0; i < nphase; i++) {
            cfg->invcdf[i + 1] = val[i];

            if ((i > 0 && val[i] < val[i - 1]) || val[i] > 1.f || val[i] < -1.f) {
                mexErrMsgTxt("cfg.invcdf contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
            }
        }

        cfg->invcdf[0] = -1.f;
        cfg->invcdf[cfg->nphase - 1] = 1.f;
        printf("mcx.invcdf=[%d];\n", cfg->nphase);
    } else if (strcmp(name, "angleinvcdf") == 0) {
        dimtype nangle = mxGetNumberOfElements(item);
        double* val = mxGetPr(item);

        if (cfg->angleinvcdf) {
            free(cfg->angleinvcdf);
        }

        cfg->nangle = (unsigned int)nangle;
        cfg->angleinvcdf = (float*)calloc(cfg->nangle, sizeof(float));

        for (i = 0; i < nangle; i++) {
            cfg->angleinvcdf[i] = val[i];

            if ((i > 0 && val[i] < val[i - 1]) || val[i] > 1.f || val[i] < 0.f) {
                mexErrMsgTxt("cfg.angleinvcdf contains invalid data; it must be a monotonically increasing vector with all values between 0 and 1");
            }
        }

        printf("mcx.angleinvcdf=[%d];\n", cfg->nangle);
    } else if (strcmp(name, "shapes") == 0) {
        int len = mxGetNumberOfElements(item);

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'shapes' field must be a non-empty string");
        }

        cfg->shapedata = (char*)calloc(len + 2, 1);
        int status = mxGetString(item, cfg->shapedata, len + 1);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        printf("mcx.shapedata='%s';\n", cfg->shapedata);
    } else if (strcmp(name, "bc") == 0) {
        int len = mxGetNumberOfElements(item);

        if (!mxIsChar(item) || len == 0 || len > 12) {
            mexErrMsgTxt("the 'bc' field must be a non-empty string");
        }

        mxGetString(item, cfg->bc, len + 1); // copy the string, plus the ending NULL, max 13 char
        printf("mcx.bc='%s';\n", cfg->bc);
    } else if (strcmp(name, "detphotons") == 0) {
        arraydim = mxGetDimensions(item);
        dimdetps[0] = arraydim[0];
        dimdetps[1] = arraydim[1];
        detps = (float*)malloc(arraydim[0] * arraydim[1] * sizeof(float));
        memcpy(detps, mxGetData(item), arraydim[0]*arraydim[1]*sizeof(float));
        printf("mcx.detphotons=[%ld %ld];\n", arraydim[0], arraydim[1]);
    } else if (strcmp(name, "seed") == 0) {
        arraydim = mxGetDimensions(item);

        if (MAX(arraydim[0], arraydim[1]) == 0) {
            mexErrMsgTxt("the 'seed' field can not be empty");
        }

        if (!mxIsUint8(item)) {
            double* val = mxGetPr(item);
            cfg->seed = val[0];
            printf("mcx.seed=%d;\n", cfg->seed);
        } else {
            seedbyte = arraydim[0];
            cfg->replay.seed = malloc(arraydim[0] * arraydim[1]);

            if (arraydim[0] != sizeof(float)*RAND_WORD_LEN) {
                mexErrMsgTxt("the row number of cfg.seed does not match RNG seed byte-length");
            }

            memcpy(cfg->replay.seed, mxGetData(item), arraydim[0]*arraydim[1]);
            cfg->seed = SEED_FROM_FILE;
            cfg->nphoton = arraydim[1];
            printf("mcx.nphoton=%ld;\n", cfg->nphoton);
        }
    } else if (strcmp(name, "gpuid") == 0) {
        int len = mxGetNumberOfElements(item);

        if (mxIsChar(item)) {
            if (len == 0) {
                mexErrMsgTxt("the 'gpuid' field must be an integer or non-empty string");
            }

            if (len > MAX_DEVICE) {
                mexErrMsgTxt("the 'gpuid' field is too long");
            }

            int status = mxGetString(item, cfg->deviceid, MAX_DEVICE);

            if (status != 0) {
                mexWarnMsgTxt("not enough space. string is truncated.");
            }

            printf("mcx.gpuid='%s';\n", cfg->deviceid);
        } else {
            double* val = mxGetPr(item);
            cfg->gpuid = val[0];
            memset(cfg->deviceid, 0, MAX_DEVICE);

            if (cfg->gpuid > 0 && cfg->gpuid < MAX_DEVICE) {
                memset(cfg->deviceid, '0', cfg->gpuid - 1);
                cfg->deviceid[cfg->gpuid - 1] = '1';
            } else {
                mexErrMsgTxt("GPU id must be positive and can not be more than 256");
            }

            printf("mcx.gpuid=%d;\n", cfg->gpuid);
        }

        for (int i = 0; i < MAX_DEVICE; i++)
            if (cfg->deviceid[i] == '0') {
                cfg->deviceid[i] = '\0';
            }
    } else if (strcmp(name, "workload") == 0) {
        double* val = mxGetPr(item);
        arraydim = mxGetDimensions(item);

        if (arraydim[0]*arraydim[1] > MAX_DEVICE) {
            mexErrMsgTxt("the workload list can not be longer than 256");
        }

        for (i = 0; i < arraydim[0]*arraydim[1]; i++) {
            cfg->workload[i] = val[i];
        }

        printf("mcx.workload=<<%ld>>;\n", arraydim[0]*arraydim[1]);
    } else if (strcmp(name, "flog") == 0) {
        int len = mxGetNumberOfElements(item);
        char logfile[MAX_SESSION_LENGTH] = {'\0'};

        if (mxIsChar(item)) {
            if (len > 0) {
                mxGetString(item, logfile, MAX_SESSION_LENGTH);
                cfg->flog = fopen(logfile, "a+");

                if (cfg->flog == NULL) {
                    mexErrMsgTxt("Log output file can not be written");
                }
            } else {
                cfg->flog = stdout;
            }
        } else {
            double* val = mxGetPr(item);

            if (len > 0 && val[0] <= 2) {
                cfg->flog = ((int)val[0] == 2 ? stderr : ((int)val[0] == 1 ? stdout : (cfg->printnum = -1, stdout)));
            }
        }

        printf("mcx.flog=%d;\n", cfg->flog);
    } else if (strcmp(name, "lang") == 0) {
        int len = mxGetNumberOfElements(item);
        char langid[32] = {'\0'};

        if (!mxIsChar(item) || len == 0) {
            mexErrMsgTxt("the 'lang' field must be a non-empty string");
        }

        if (len > 5) {
            mexErrMsgTxt("the 'lang' field is too long");
        }

        int status = mxGetString(item, langid, 32);

        if (status != 0) {
            mexWarnMsgTxt("not enough space. string is truncated.");
        }

        int idx = mcx_keylookup(langid, languagename);

        if (idx == -1) {
            mexErrMsgTxt(T_("Unsupported language"));
        }

        mcx_lang = cJSON_Parse(translations[idx]);
        printf("mcx.lang='%s';\n", langid);
    } else {
        printf(S_RED "WARNING: redundant field '%s'\n" S_RESET, name);
    }
}

/**
 * @brief Error reporting function in the mex function, equivallent to mcx_error in binary mode
 *
 * @param[in] id: a single integer for the types of the error
 * @param[in] msg: the error message string
 * @param[in] filename: the unit file name where this error is raised
 * @param[in] linenum: the line number in the file where this error is raised
 */

extern "C" int mcx_throw_exception(const int id, const char* msg, const char* filename, const int linenum) {
    printf("MCXLAB ERROR %d in unit %s:%d: %s\n", id, filename, linenum, msg);
    throw msg;
    return id;
}

/**
 * @brief Print a brief help information if nothing is provided
 */

void mcxlab_usage() {
    printf("MCXLAB (" MCX_VERSION ")\nUsage:\n    [flux,detphoton,vol,seeds,traj]=mcxlab(cfg);\n\nPlease run 'help mcxlab' for more details.\n");
}

/**
 * @brief Force matlab refresh the command window to print all buffered messages
 */

extern "C" void mcx_matlab_flush() {
#ifdef _OPENMP
    #pragma omp master
    {
#endif

#ifndef MATLAB_MEX_FILE
        mexEvalString("fflush(stdout);");
#else
        mexEvalString("pause(.0001);");
#endif

#ifdef _OPENMP
    }
#endif

}
