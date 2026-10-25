/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2026
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
\file    mcx_utils.c

@brief   Simulation configuration and command line option handling
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>

#ifndef WIN32
    #include <sys/ioctl.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#ifdef WIN32
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
#endif
#include "mcx_utils.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include "mcx_core.h"
#include "mcx_bench.h"
#include "mcx_mie.h"
#include "mcx_neurojson.h"
#include "mcx_lang.h"

#if defined(_WIN32) && defined(USE_OS_TIMER) && !defined(MCX_CONTAINER)
    #include "mmc_tictoc.h"
#endif

#ifndef MCX_CONTAINER
    #include "zmat/zmatlib.h"
    #include "ubj/ubj.h"
#endif

/**
 * Macro to load JSON keys
 */
#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
     ((idfull==NULL || (tmp=cJSON_GetObjectItem(root,idfull))==0) ? fallback : tmp->val) \
     : tmp->val)

/**
 * Macro to load JSON object
 */
#define FIND_JSON_OBJ(id,idfull,parent) \
    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
     ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? NULL : tmp) \
     : tmp)

#define UBJ_WRITE_KEY(ctx, key,  type, val)    {ubjw_write_key( (ctx), (key)); ubjw_write_##type((ctx), (val));}
#define UBJ_WRITE_ARRAY(ctx, type, nlen, val)  {ubjw_write_buffer( (ctx), (unsigned char*)(val), (UBJ_TYPE)(JDB_##type), (nlen));}

#define ubjw_write_single ubjw_write_float32
#define ubjw_write_double ubjw_write_float64

#ifdef WIN32
    char pathsep = '\\';
#else
    char pathsep = '/';
#endif

/**
 * Macro to include unit name and line number in the error message
 */
#define MCX_ASSERT(a)  (!(a) && (mcx_error(-1,"input error",__FILE__,__LINE__),1) ); /* exit with a non-zero code */

#define MIN_HEADER_SIZE 348    /**< Analyze header size */
#define NII_HEADER_SIZE 352    /**< NIFTI header size */
#define GL_RGBA32F 0x8814

/**
 * Short command line options
 * If a short command line option is '-' that means it only has long/verbose option.
 * Array terminates with '\0'.
 * Currently un-used options: cCJoy0-9
 */

const char shortopt[] = {'h', 'i', 'f', 'n', 't', 'T', 's', 'a', 'g', 'b', '-', 'z', 'u', 'H', 'P',
                         'd', 'r', 'S', 'p', 'e', 'U', 'R', 'l', 'L', '-', 'I', '-', 'G', 'M', 'A', 'E', 'v', 'D',
                         'k', 'q', 'Y', 'O', 'F', '-', '-', 'x', 'X', '-', 'K', 'm', 'V', 'B', 'W', 'w', '-',
                         'Q', '-', 'Z', 'j', '-', '-', '-', 'N', 'y', '\0'
                        };

/**
 * Long command line options
 * The length of this array must match the length of shortopt[], terminates with ""
 */
const char* fullopt[] = {"--help", "--interactive", "--input", "--photon",
                         "--thread", "--blocksize", "--session", "--array",
                         "--gategroup", "--reflect", "--reflectin", "--srcfrom0",
                         "--unitinmm", "--maxdetphoton", "--shapes", "--savedet",
                         "--repeat", "--save2pt", "--printlen", "--minenergy",
                         "--normalize", "--skipradius", "--log", "--listgpu", "--faststep",
                         "--printgpu", "--root", "--gpu", "--dumpmask", "--autopilot",
                         "--seed", "--version", "--debug", "--voidtime", "--saveseed",
                         "--replaydet", "--outputtype", "--outputformat", "--maxjumpdebug",
                         "--maxvoidstep", "--saveexit", "--saveref", "--gscatter", "--mediabyte",
                         "--momentum", "--specular", "--bc", "--workload", "--savedetflag",
                         "--internalsrc", "--bench", "--dumpjson", "--zip", "--json", "--atomic",
                         "--srcid", "--trajstokes", "--net", "--lang", ""
                        };

/**
 * Output data types
 * x: fluence rate
 * f: fluence
 * e: energy deposition
 * j: jacobian for mua
 * p: scattering counts for computing Jacobians for mus
 * m: momentum transfer in replay
 * r: frequency domain/RF mua Jacobian by replay
 * l: total path lengths in each voxel
 * s : frequency domain/RF mus Jacobian by replay
 * t : weighted average of time-of-flight x total path length in each voxel
 * b : weighted average of time-of-flight x total scattering count in each voxel
 */

const char outputtype[] = {'x', 'f', 'e', 'j', 'p', 'm', 'r', 'l', 's', 't', 'b', 'a', 'd', 'u', 'v', 'w', 'q', '\0'};
/*                          flux flnc enrg jcbn wp  dcs  rf  plen rfms wltof wptof adj  D   mus  musp muad mamp */

/**
 * Debug flags
 * R: debug random number generator
 * M: record photon movement and trajectory
 * P: show progress bar
 */

const char debugflag[] = {'R', 'M', 'P', 'T', '\0'};

/**
 * Recorded fields for detected photons
 * D: detector ID (starting from 1) [1]
 * S: partial scattering event count [#media]
 * P: partial path lengths [#media]
 * M: momentum transfer [#media]
 * X: exit position [3]
 * V: exit direction vector [3]
 * W: initial weight [1]
 */

const char saveflag[] = {'D', 'S', 'P', 'M', 'X', 'V', 'W', 'I', '\0'};

/**
 * Output file format
 * mc2: binary mc2 format to store fluence volume data
 * nii: output fluence in nii format
 * hdr: output volume in Analyze hdr/img format
 * ubj: output volume in unversal binary json format (not implemented)
 * tx3: a simple 3D texture format
 * jnii: NeuroJSON JNIfTI format (JSON compatible)
 * bnii: NeuroJSON binary JNIfTI format (binary JSON format BJData compatible)
 */

const char* outputformat[] = {"mc2", "nii", "hdr", "ubj", "tx3", "jnii", "bnii", ""};

/**
 * Boundary condition (BC) types
 * _: no condition (fallback to isreflect)
 * r: Fresnel boundary
 * a: total absorption BC
 * m: total reflection (mirror) BC
 * c: cyclic BC
 */

const char boundarycond[] = {'_', 'r', 'a', 'm', 'c', '\0'};

/**
 * Boundary detection flags
 * 0: do not detect photon
 * 1: detect photon at that boundary
 */

const char boundarydetflag[] = {'0', '1', '\0'};

/**
 * Source type specifier
 * User can specify the source type using a string
 */

const char* srctypeid[] = {"pencil", "isotropic", "cone", "gaussian", "planar",
                           "pattern", "fourier", "arcsine", "disk", "fourierx", "fourierx2d", "zgaussian",
                           "line", "slit", "pencilarray", "pattern3d", "hyperboloid", "ring", ""
                          };


/**
 * Media byte format
 * User can specify the source type using a string
 */

const unsigned int mediaformatid[] = {1, 2, 4, 96, 97, 98, 99, 100, 101, 102, 103, 104, 0};
const char* mediaformat[] = {"byte", "short", "integer", "asgn_float", "svmc", "mixlabel", "labelplus",
                             "muamus_float", "mua_float", "muamus_half", "asgn_byte", "muamus_short", ""
                            };

/**
 * Flag to decide if parameter has been initialized over command line
 */

char flagset[256] = {'\0'};

const char* zipformat[] = {"zlib", "gzip", "base64", "lzip", "lzma", "lz4", "lz4hc", ""};


/**
 * JSON object to store translations for suppored languages
 */

cJSON* mcx_lang = NULL;

/**
 * @brief Initializing the simulation configuration with default values
 *
 * Constructor of the simulation configuration, initializing all field to default values
 */

void mcx_initcfg(Config* cfg) {
    cfg->medianum = 0;
    cfg->polmedianum = 0;
    cfg->mediabyte = 1;      /** expect 1-byte per medium index, use --mediabyte to set to 2 or 4 */
    cfg->detnum = 0;
    cfg->dim.x = 0;
    cfg->dim.y = 0;
    cfg->dim.z = 0;
    cfg->steps.x = 1.f;
    cfg->steps.y = 1.f;
    cfg->steps.z = 1.f;
    cfg->nblocksize = 64;    /** in theory, mcx can use min block size 32 because no communication between threads, but 64 seems to work the best */
    cfg->nphoton = 0;
    cfg->nthread = (1 << 14); /** launch many threads to saturate the device to maximize throughput */
    cfg->isrowmajor = 0;     /** default is Matlab array */
    cfg->maxgate = 0;
    cfg->isreflect = 1;
    cfg->isref3 = 1;
    cfg->isrefint = 0;
    cfg->isnormalized = 1;
    cfg->issavedet = 1;      /** output detected photon data by default, use -d 0 to disable */
    cfg->respin = 1;
    cfg->issave2pt = 1;
    cfg->isgpuinfo = 0;
    cfg->prop = NULL;
    cfg->polprop = NULL;
    cfg->detpos = NULL;
    cfg->detdir = NULL;
    cfg->smatrix = NULL;
    cfg->vol = NULL;
    cfg->session[0] = '\0';
    cfg->printnum = 0;
    cfg->minenergy = 0.f;
    cfg->flog = stdout;
    cfg->sradius = -2.f;
    cfg->rootpath[0] = '\0';
    cfg->gpuid = 0;
    cfg->issrcfrom0 = 0;
    cfg->tstart = 0.f;
    cfg->tend = 0.f;
    cfg->tstep = VERY_BIG;
    cfg->unitinmm = 1.f;
    cfg->isdumpmask = 0;
    cfg->isdumpjson = 0;
    cfg->srctype = 0;;       /** use pencil beam as default source type */
    cfg->maxdetphoton = 1000000;
    cfg->maxjumpdebug = 10000000;
    cfg->exportdebugdata = NULL;
    cfg->debugdatalen = 0;
    cfg->autopilot = 1;
    cfg->seed = 0x623F9A9E;  /** default RNG seed, a big integer, with a hidden meaning :) */
    cfg->exportfield = NULL;
    cfg->exportjacob = NULL;
    cfg->exportdetected = NULL;
    cfg->energytot = 0.f;
    cfg->energyabs = 0.f;
    cfg->energyesc = 0.f;
#ifndef MCX_CONTAINER
    cfg->zipid = zmZlib;
#endif
    cfg->omega = 0.f;
    cfg->lambda = 0.f;
    /*cfg->his=(History){{'M','C','X','H'},1,0,0,0,0,0,0,1.f,{0,0,0,0,0,0,0}};*/   /** This format is only supported by C99 */
    memset(&cfg->his, 0, sizeof(History));
    memcpy(cfg->his.magic, "MCXH", 4);
    cfg->his.version = 1;
    cfg->his.unitinmm = 1.f;
    cfg->his.normalizer = 1.f;
    cfg->his.respin = 1;
    cfg->his.srcnum = 1;
    cfg->savedetflag = 0x5;
    cfg->his.savedetflag = cfg->savedetflag;
    cfg->shapedata = NULL;
    cfg->extrajson = NULL;
    cfg->seeddata = NULL;
    cfg->maxvoidstep = 1000;
    cfg->voidtime = 1;
    cfg->srcpattern = NULL;
    cfg->srcnum = 1;
    cfg->debuglevel = 0;
    cfg->issaveseed = 0;
    cfg->issaveexit = 0;
    cfg->istrajstokes = 0;
    cfg->ismomentum = 0;
    cfg->internalsrc = 0;
    cfg->replay.seed = NULL;
    cfg->replay.weight = NULL;
    cfg->replay.tof = NULL;
    cfg->replay.detid = NULL;
    cfg->replaydet = 0;
    cfg->seedfile[0] = '\0';
    cfg->outputtype = otFlux;
    cfg->outputformat = ofJNifti;
    cfg->detectedcount = 0;
    cfg->runtime = 0;
    cfg->faststep = 0;
    cfg->srcpos.w = 1.f;
    cfg->srcdir.w = 0.f;
    cfg->issaveref = 0;
    cfg->isspecular = 0;
    cfg->dx = cfg->dy = cfg->dz = NULL;
    cfg->gscatter = 1e9;   /** by default, honor anisotropy for all scattering, use --gscatter to reduce it */
    cfg->nphase = 0;
    cfg->invcdf = NULL;
    cfg->nangle = 0;
    cfg->angleinvcdf = NULL;
    cfg->srcid = 0;
    cfg->extrasrclen = 0;
    cfg->his.totalsource = cfg->extrasrclen + 1;
    cfg->srcdata = NULL;
    memset(cfg->jsonfile, 0, MAX_PATH_LENGTH);
    memset(cfg->langid, 0, MAX_LANG_ID);
    memset(cfg->bc, 0, 13);
    memset(&(cfg->srcparam1), 0, sizeof(float4));
    memset(&(cfg->srcparam2), 0, sizeof(float4));
    memset(cfg->deviceid, 0, MAX_DEVICE);
    memset(cfg->workload, 0, MAX_DEVICE * sizeof(float));
    cfg->deviceid[0] = '1'; /** use the first GPU device by default*/
#if defined(MCX_CONTAINER)
#ifndef PYBIND11_VERSION_MAJOR
    cfg->parentid = mpMATLAB;
#else
    cfg->parentid = mpPython;
#endif
#else
    cfg->parentid = mpStandalone;
#endif

    if (!mcx_lang) {
        char* envlocale = getenv( "MCX_LANG" );

        if (envlocale) {
            int idx;
            char langid[6] = {'\0'};
            strncpy(langid, envlocale, 5);
            idx = mcx_keylookup(langid, languagename);

            if (idx == -1) {
                MCX_FPRINTF(cfg->flog, "Unsupported language ID (%s), fallback to default\n", langid);
            } else {
                mcx_lang = mcx_parsejson(translations[idx]);
            }
        }
    }
}

/**
 * @brief Reset and clear the GPU information data structure
 *
 * Clearing the GPU information data structure
 */

void mcx_cleargpuinfo(GPUInfo** gpuinfo) {
    if (*gpuinfo) {
        free(*gpuinfo);
        *gpuinfo = NULL;
    }
}

/**
 * @brief Clearing the simulation configuration data structure
 *
 * Destructor of the simulation configuration, delete all dynamically allocated members
 */

void mcx_clearcfg(Config* cfg) {
    if (cfg->medianum) {
        free(cfg->prop);
    }

    if (cfg->polmedianum) {
        free(cfg->polprop);
    }

    if (cfg->smatrix) {
        free(cfg->smatrix);
    }

    if (cfg->detnum) {
        free(cfg->detpos);

        if (cfg->detdir) {
            free(cfg->detdir);
        }
    }

    if (cfg->dim.x && cfg->dim.y && cfg->dim.z) {
        free(cfg->vol);
    }

    if (cfg->srcpattern) {
        free(cfg->srcpattern);
    }

    if (cfg->replay.weight) {
        free(cfg->replay.weight);
    }

    if (cfg->replay.seed) {
        free(cfg->replay.seed);
    }

    if (cfg->replay.tof) {
        free(cfg->replay.tof);
    }

    if (cfg->replay.detid) {
        free(cfg->replay.detid);
    }

    if (cfg->dx) {
        free(cfg->dx);
    }

    if (cfg->dy) {
        free(cfg->dy);
    }

    if (cfg->dz) {
        free(cfg->dz);
    }

    if (cfg->exportfield) {
        free(cfg->exportfield);
    }

    if (cfg->exportjacob) {
        free(cfg->exportjacob);
        cfg->exportjacob = NULL;
    }

    if (cfg->exportdetected) {
        free(cfg->exportdetected);
    }

    if (cfg->exportdebugdata) {
        free(cfg->exportdebugdata);
    }

    if (cfg->seeddata) {
        free(cfg->seeddata);
    }

    if (cfg->shapedata) {
        free(cfg->shapedata);
    }

    if (cfg->extrajson) {
        free(cfg->extrajson);
    }

    if (cfg->invcdf) {
        free(cfg->invcdf);
    }

    if (cfg->angleinvcdf) {
        free(cfg->angleinvcdf);
    }

    if (cfg->srcdata) {
        free(cfg->srcdata);
    }

    if (mcx_lang) {
        cJSON_Delete(mcx_lang);
        mcx_lang = NULL;
    }

    mcx_initcfg(cfg);
}

#ifndef MCX_CONTAINER

/**
 * @brief Save volumetric output (fluence etc) to an Nifty format binary file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] name: output file name (will append '.nii')
 * @param[in] type32bit: type of the data, only support 32bit per record
 * @param[in] outputformatid: decide if save as nii or analyze format
 * @param[in] cfg: simulation configuration
 */

void mcx_savenii(float* dat, size_t len, char* name, int type32bit, int outputformatid, Config* cfg) {
    FILE* fp;
    char fname[MAX_FULL_PATH] = {'\0'};
    nifti_1_header hdr;
    nifti1_extender pad = {{0, 0, 0, 0}};
    float* logval = dat;
    size_t i;

    memset((void*)&hdr, 0, sizeof(hdr));
    hdr.sizeof_hdr = MIN_HEADER_SIZE;
    hdr.dim[0] = 4;
    hdr.dim[1] = cfg->dim.x;
    hdr.dim[2] = cfg->dim.y;
    hdr.dim[3] = cfg->dim.z;
    hdr.dim[4] = len / (cfg->dim.x * cfg->dim.y * cfg->dim.z);
    hdr.datatype = type32bit;
    hdr.bitpix = 32;
    hdr.pixdim[1] = cfg->unitinmm;
    hdr.pixdim[2] = cfg->unitinmm;
    hdr.pixdim[3] = cfg->unitinmm;
    hdr.intent_code = NIFTI_INTENT_NONE;

    if (type32bit == NIFTI_TYPE_FLOAT32) {
        hdr.pixdim[4] = cfg->tstep * 1e6f;
    } else {
        /* write the label/detector mask pairs to a new buffer, dat may be cfg->vol in use */
        short* mask = (short*)malloc(len * 2 * sizeof(short));

        if (mask == NULL) {
            MCX_ERROR(-9, "can not allocate memory");
        }

        logval = (float*)mask;

        for (i = 0; i < len; i++) {
            mask[(i << 1)]    = (((unsigned int*)dat)[i] & MED_MASK);
            mask[(i << 1) + 1] = (((unsigned int*)dat)[i] & DET_MASK) >> 31;
        }

        hdr.datatype = NIFTI_TYPE_UINT16;
        hdr.bitpix = 16;
        hdr.dim[1] = 2;
        hdr.dim[2] = cfg->dim.x;
        hdr.dim[3] = cfg->dim.y;
        hdr.dim[4] = cfg->dim.z;
        hdr.pixdim[4] = cfg->unitinmm;
        hdr.pixdim[1] = 1.f;
    }

    if (outputformatid == ofNifti) {
        strncpy(hdr.magic, "n+1\0", 4);
        hdr.vox_offset = (float) NII_HEADER_SIZE;
    } else {
        strncpy(hdr.magic, "ni1\0", 4);
        hdr.vox_offset = (float)0;
    }

    hdr.scl_slope = 0.f;
    hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_USEC;

    sprintf(fname, "%s.%s", name, outputformat[outputformatid]);

    if (( fp = fopen(fname, "wb")) == NULL) {
        MCX_ERROR(-9, "Error opening header file for write");
    }

    if (fwrite(&hdr, MIN_HEADER_SIZE, 1, fp) != 1) {
        MCX_ERROR(-9, "Error writing header file");
    }

    if (outputformatid == ofNifti) {
        if (fwrite(&pad, 4, 1, fp) != 1) {
            MCX_ERROR(-9, "Error writing header file extension pad");
        }

        if (fwrite(logval, (size_t)(hdr.bitpix >> 3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) !=
                hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4]) {
            MCX_ERROR(-9, "Error writing data to file");
        }

        fclose(fp);
    } else if (outputformatid == ofAnalyze) {
        fclose(fp);  /* close .hdr file */

        sprintf(fname, "%s.img", name);

        fp = fopen(fname, "wb");

        if (fp == NULL) {
            MCX_ERROR(-9, "Error opening img file for write");
        }

        if (fwrite(logval, (size_t)(hdr.bitpix >> 3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) !=
                hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4]) {
            MCX_ERROR(-9, "Error writing img file");
        }

        fclose(fp);
    } else {
        MCX_ERROR(-9, "Output format is not supported");
    }

    if (logval != dat) {
        free(logval);
    }
}

/**
 * @brief Return the "Description" string of the JNIfTI header for an output type
 *
 * @param[in] type: output type, an enum TOutputType value
 */

static const char* mcx_outputdesc(int type) {
    static const char* typestr[] = {
        "MCX volumetric output: Fluence rate (W/mm^2)",
        "MCX volumetric output: Fluence (J/mm^2)",
        "MCX volumetric output: Voxel-wise energy deposit (J)",
        "MCX volumetric output: Jacobian for mua (J/mm)",
        "MCX volumetric output: Scattering count",
        "MCX volumetric output: Partial momentum transfer",
        "MCX volumetric output: Frequency-domain (RF) output, real and imaginary parts",
        "MCX volumetric output: Total path length (mm)",
        "MCX volumetric output: RF Jacobian for mus, real and imaginary parts",
        "MCX volumetric output: Weighted time-of-flight x path length",
        "MCX volumetric output: Weighted time-of-flight x scattering count",
        "MCX volumetric output: Adjoint Jacobian for mua",
        "MCX volumetric output: Adjoint Jacobian for the diffusion coefficient D",
        "MCX volumetric output: Adjoint Jacobian for mus",
        "MCX volumetric output: Adjoint Jacobian for mus'",
        "MCX volumetric output: Adjoint Jacobians for mua and D",
        "MCX volumetric output: Adjoint Jacobians for mua and mus'"
    };

    if (type < 0 || type >= (int)(sizeof(typestr) / sizeof(typestr[0]))) {
        return "MCX volumetric output";
    }

    return typestr[type];
}

/**
 * @brief Save volumetric output (fluence etc) to a binary JNIfTI/JSON/JData format file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] name: output file name (will append '.nii')
 * @param[in] type32bit: type of the data, only support 32bit per record
 * @param[in] outputformatid: decide if save as nii or analyze format
 * @param[in] cfg: simulation configuration
 */

void mcx_savebnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, int iscol, Config* cfg) {
    FILE* fp;
    char fname[MAX_FULL_PATH] = {'\0'};
    int affine[] = {0, 0, 1, 0, 0, 0};
    size_t datalen = sizeof(float), outputlen = 0;

    ubjw_context_t* root = NULL;
    uchar* jsonstr = NULL;

    for (int i = 0; i < ndim; i++) {
        datalen *= dims[i];
    }

    /* raw bytes (datalen) + 2x worst-case headroom for headers and BJData markers */
    datalen = (datalen << 1) + 4096;
    jsonstr = malloc(datalen);
    root = ubjw_open_memory(jsonstr, jsonstr + datalen);

    ubjw_begin_object(root, UBJ_MIXED, 0);
    /* the "_DataInfo_" section */
    ubjw_write_key(root, "_DataInfo_");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    UBJ_WRITE_KEY(root, "JNIFTIVersion", string, "0.5");
    UBJ_WRITE_KEY(root, "Comment", string, "Created by MCX (https://mcx.space)");
    UBJ_WRITE_KEY(root, "AnnotationFormat", string, "https://neurojson.org/jnifti/draft1");
    UBJ_WRITE_KEY(root, "SerialFormat", string, "https://neurojson.org/bjdata/draft2");
    ubjw_write_key(root, "Parser");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    ubjw_write_key(root, "Python");
    ubjw_begin_array(root, UBJ_STRING, 2);
    ubjw_write_string(root, "https://neurojson.org/download/pyjdata");
    ubjw_write_string(root, "https://neurojson.org/download/pybjdata");
    ubjw_end(root);
    ubjw_write_key(root, "MATLAB");
    ubjw_begin_array(root, UBJ_STRING, 2);
    ubjw_write_string(root, "https://neurojson.org/download/jnifty");
    ubjw_write_string(root, "https://neurojson.org/download/jsonlab");
    ubjw_end(root);
    ubjw_write_key(root, "JavaScript");
    ubjw_begin_array(root, UBJ_STRING, 2);
    ubjw_write_string(root, "https://www.npmjs.com/package/jda");
    ubjw_write_string(root, "https://www.npmjs.com/package/bjd");
    ubjw_end(root);
    UBJ_WRITE_KEY(root, "CPP", string, "https://github.com/NeuroJSON/json");
    UBJ_WRITE_KEY(root, "C", string, "https://github.com/NeuroJSON/ubj");
    ubjw_end(root);
    ubjw_end(root);

    /* the "NIFTIHeader" section */
    ubjw_write_key(root, "NIFTIHeader");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    UBJ_WRITE_KEY(root, "NIIHeaderSize", uint16, 348);
    ubjw_write_key(root, "Dim");
    UBJ_WRITE_ARRAY(root, uint32, ndim, dims);
    UBJ_WRITE_KEY(root, "Param1", uint8, 0);
    UBJ_WRITE_KEY(root, "Param2", uint8, 0);
    UBJ_WRITE_KEY(root, "Param3", uint8, 0);
    UBJ_WRITE_KEY(root, "Intent", uint8, 0);
    UBJ_WRITE_KEY(root, "DataType", string, ((isfloat ? "single" : "uint32")));
    UBJ_WRITE_KEY(root, "BitDepth", uint8, 32);
    UBJ_WRITE_KEY(root, "FirstSliceID", uint8, 0);
    ubjw_write_key(root, "VoxelSize");
    UBJ_WRITE_ARRAY(root, single, ndim, voxelsize);
    ubjw_write_key(root, "Orientation");
    ubjw_begin_object(root, UBJ_MIXED, 3);
    UBJ_WRITE_KEY(root, "x", char, 'r');
    UBJ_WRITE_KEY(root, "y", char, 'a');
    UBJ_WRITE_KEY(root, "z", char, 's');
    ubjw_end(root);
    UBJ_WRITE_KEY(root, "ScaleSlope", uint8, 1);
    UBJ_WRITE_KEY(root, "ScaleOffset", uint8, 1);
    UBJ_WRITE_KEY(root, "LastSliceID", uint32, cfg->maxgate);
    UBJ_WRITE_KEY(root, "SliceType", uint8, 1);
    ubjw_write_key(root, "Unit");
    ubjw_begin_object(root, UBJ_MIXED, 2);
    UBJ_WRITE_KEY(root, "L", string, "mm");
    UBJ_WRITE_KEY(root, "T", string, "s");
    ubjw_end(root);
    UBJ_WRITE_KEY(root, "MaxIntensity", uint32, 1);
    UBJ_WRITE_KEY(root, "MinIntensity", uint32, 0);
    UBJ_WRITE_KEY(root, "SliceTime", uint8, 0);
    UBJ_WRITE_KEY(root, "TimeOffset", uint8, 0);

    UBJ_WRITE_KEY(root, "Description", string, mcx_outputdesc(cfg->outputtype));

    UBJ_WRITE_KEY(root, "AuxFile", string, "");
    UBJ_WRITE_KEY(root, "QForm", uint8, 0);
    UBJ_WRITE_KEY(root, "SForm", uint8, 1);
    ubjw_write_key(root, "Quatern");
    ubjw_begin_object(root, UBJ_MIXED, 3);
    UBJ_WRITE_KEY(root, "b", uint8, 0);
    UBJ_WRITE_KEY(root, "c", uint8, 0);
    UBJ_WRITE_KEY(root, "d", uint8, 0);
    ubjw_end(root);
    ubjw_write_key(root, "QuaternOffset");
    ubjw_begin_object(root, UBJ_MIXED, 3);
    UBJ_WRITE_KEY(root, "x", uint8, 0);
    UBJ_WRITE_KEY(root, "y", uint8, 0);
    UBJ_WRITE_KEY(root, "z", uint8, 0);
    ubjw_end(root);
    ubjw_write_key(root, "Affine");
    ubjw_begin_array(root, UBJ_MIXED, 0);
    UBJ_WRITE_ARRAY(root, int32, 4, affine + 2);
    UBJ_WRITE_ARRAY(root, int32, 4, affine + 1);
    UBJ_WRITE_ARRAY(root, int32, 4, affine);
    ubjw_end(root);
    UBJ_WRITE_KEY(root, "Name", string, cfg->session);
    UBJ_WRITE_KEY(root, "NIIFormat", string, "jnifti");
    ubjw_end(root);

    ubjw_write_key(root, "NIFTIData");

    /* the "NIFTIData" section stores volumetric data */
    ubjw_begin_object(root, UBJ_MIXED, 0);

    if (mcx_jdataencode(vol, ndim, dims, (isfloat ? "single" : "uint32"), 4, cfg->zipid, root, 1, iscol, cfg)) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    ubjw_end(root);
    ubjw_end(root);

    /* now save JSON to file */
    outputlen = ubjw_close_context(root);

    if (jsonstr == NULL) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    sprintf(fname, "%s.bnii", name);

    fp = fopen(fname, "wb");

    if (fp == NULL) {
        MCX_ERROR(-1, "error opening file to write");
    }

    fwrite(jsonstr, outputlen, 1, fp);
    fclose(fp);

    if (jsonstr) {
        free(jsonstr);
    }
}


/**
 * @brief Save volumetric output (fluence etc) to a JNIfTI/JSON/JData format file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] name: output file name (will append '.nii')
 * @param[in] type32bit: type of the data, only support 32bit per record
 * @param[in] outputformatid: decide if save as nii or analyze format
 * @param[in] cfg: simulation configuration
 */

void mcx_savejnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, int iscol, Config* cfg) {
    FILE* fp;
    char fname[MAX_FULL_PATH] = {'\0'};
    int affine[] = {0, 0, 1, 0, 0, 0};
    const char* libpy[] = {"https://neurojson.org/download/pyjdata", "https://neurojson.org/download/pybjdata"};
    const char* libmat[] = {"https://neurojson.org/download/jnifty", "https://neurojson.org/download/jsonlab"};
    const char* libjs[] = {"https://www.npmjs.com/package/jda", "https://www.npmjs.com/package/bjd"};
    const char* libc[]  = {"https://github.com/DaveGamble/cJSON", "https://github.com/NeuroJSON/ubj"};

    cJSON* root = NULL, *hdr = NULL, *dat = NULL, *sub = NULL, *info = NULL, *parser = NULL;
    char* jsonstr = NULL;
    root = cJSON_CreateObject();

    /* the "_DataInfo_" section */
    cJSON_AddItemToObject(root, "_DataInfo_", info = cJSON_CreateObject());
    cJSON_AddStringToObject(info, "JNIFTIVersion", "0.5");
    cJSON_AddStringToObject(info, "Comment", "Created by MCX (https://mcx.space)");
    cJSON_AddStringToObject(info, "AnnotationFormat", "https://neurojson.org/jnifti/draft1");
    cJSON_AddStringToObject(info, "SerialFormat", "https://json.org");
    cJSON_AddItemToObject(info, "Parser", parser = cJSON_CreateObject());
    cJSON_AddItemToObject(parser, "Python", cJSON_CreateStringArray(libpy, 2));
    cJSON_AddItemToObject(parser, "MATLAB", cJSON_CreateStringArray(libmat, 2));
    cJSON_AddItemToObject(parser, "JavaScript", cJSON_CreateStringArray(libjs, 2));
    cJSON_AddStringToObject(parser, "CPP", "https://github.com/NeuroJSON/json");
    cJSON_AddItemToObject(parser, "C", cJSON_CreateStringArray(libc, 2));

    /* the "NIFTIHeader" section */
    cJSON_AddItemToObject(root, "NIFTIHeader", hdr = cJSON_CreateObject());
    cJSON_AddNumberToObject(hdr, "NIIHeaderSize", 348);
    cJSON_AddItemToObject(hdr, "Dim", cJSON_CreateIntArray((int*)dims, ndim));
    cJSON_AddNumberToObject(hdr, "Param1", 0);
    cJSON_AddNumberToObject(hdr, "Param2", 0);
    cJSON_AddNumberToObject(hdr, "Param3", 0);
    cJSON_AddNumberToObject(hdr, "Intent", 0);
    cJSON_AddStringToObject(hdr, "DataType", (isfloat ? "single" : "uint32"));
    cJSON_AddNumberToObject(hdr, "BitDepth", 32);
    cJSON_AddNumberToObject(hdr, "FirstSliceID", 0);
    cJSON_AddItemToObject(hdr, "VoxelSize", cJSON_CreateFloatArray(voxelsize, ndim));
    cJSON_AddItemToObject(hdr, "Orientation", sub = cJSON_CreateObject());
    cJSON_AddStringToObject(sub, "x", "r");
    cJSON_AddStringToObject(sub, "y", "a");
    cJSON_AddStringToObject(sub, "z", "s");
    cJSON_AddNumberToObject(hdr, "ScaleSlope", 1);
    cJSON_AddNumberToObject(hdr, "ScaleOffset", 0);
    cJSON_AddNumberToObject(hdr, "LastSliceID", cfg->maxgate);
    cJSON_AddNumberToObject(hdr, "SliceType", 1);
    cJSON_AddItemToObject(hdr, "Unit", sub = cJSON_CreateObject());
    cJSON_AddStringToObject(sub, "L", "mm");
    cJSON_AddStringToObject(sub, "T", "s");
    cJSON_AddNumberToObject(hdr, "MaxIntensity", 1);
    cJSON_AddNumberToObject(hdr, "MinIntensity", 0);
    cJSON_AddNumberToObject(hdr, "SliceTime", 0);
    cJSON_AddNumberToObject(hdr, "TimeOffset", 0);

    cJSON_AddStringToObject(hdr, "Description", mcx_outputdesc(cfg->outputtype));

    cJSON_AddStringToObject(hdr, "AuxFile", "");
    cJSON_AddNumberToObject(hdr, "QForm", 0);
    cJSON_AddNumberToObject(hdr, "SForm", 1);
    cJSON_AddItemToObject(hdr, "Quatern", sub = cJSON_CreateObject());
    cJSON_AddNumberToObject(sub, "b", 0);
    cJSON_AddNumberToObject(sub, "c", 0);
    cJSON_AddNumberToObject(sub, "d", 0);
    cJSON_AddItemToObject(hdr, "QuaternOffset", sub = cJSON_CreateObject());
    cJSON_AddNumberToObject(sub, "x", 0);
    cJSON_AddNumberToObject(sub, "y", 0);
    cJSON_AddNumberToObject(sub, "z", 0);
    cJSON_AddItemToObject(hdr, "Affine", sub = cJSON_CreateArray());
    cJSON_AddItemToArray(sub, cJSON_CreateIntArray(affine + 2, 4));
    cJSON_AddItemToArray(sub, cJSON_CreateIntArray(affine + 1, 4));
    cJSON_AddItemToArray(sub, cJSON_CreateIntArray(affine, 4));
    cJSON_AddStringToObject(hdr, "Name", cfg->session);
    cJSON_AddStringToObject(hdr, "NIIFormat", "jnifti");

    /* the "NIFTIData" section stores volumetric data */
    cJSON_AddItemToObject(root, "NIFTIData",   dat = cJSON_CreateObject());

    if (mcx_jdataencode(vol, ndim, dims, (isfloat ? "single" : "uint32"), 4, cfg->zipid, dat, 0, iscol, cfg)) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    /* now save JSON to file */
    jsonstr = cJSON_Print(root);

    if (jsonstr == NULL) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    sprintf(fname, "%s.jnii", name);

    fp = fopen(fname, "wt");

    if (fp == NULL) {
        MCX_ERROR(-1, "error opening file to write");
    }

    fprintf(fp, "%s\n", jsonstr);
    fclose(fp);

    if (jsonstr) {
        free(jsonstr);
    }

    if (root) {
        cJSON_Delete(root);
    }
}


/**
 * @brief Save volumetric output (fluence etc) to mc2 format binary file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] cfg: simulation configuration
 */

void mcx_savedata(float* dat, size_t len, Config* cfg) {
    FILE* fp;
    char name[MAX_FULL_PATH];
    char fname[MAX_FULL_PATH + 10];
    unsigned int glformat = GL_RGBA32F;

    if (cfg->rootpath[0]) {
        sprintf(name, "%s%c%s", cfg->rootpath, pathsep, cfg->session);
    } else {
        sprintf(name, "%s", cfg->session);
    }

    if (cfg->outputformat == ofNifti || cfg->outputformat == ofAnalyze) {
        mcx_savenii(dat, len * (1 + (cfg->outputtype == otRF || cfg->outputtype == otRFmus)), name, NIFTI_TYPE_FLOAT32, cfg->outputformat, cfg);
        return;
    } else if (cfg->outputformat == ofJNifti || cfg->outputformat == ofBJNifti) {
        uint dims[6] = {cfg->dim.x, cfg->dim.y, cfg->dim.z, cfg->maxgate, cfg->srcnum, 1};
        float voxelsize[6] = {cfg->steps.x, cfg->steps.y, cfg->steps.z, cfg->tstep, 1, 1};

        if (cfg->extrasrclen && cfg->srcid < 0) {
            dims[5] = cfg->extrasrclen + 1;
        }

        if (cfg->seed == SEED_FROM_FILE && (cfg->replaydet == -1 && cfg->detnum > 1)) {
            dims[5] *= cfg->detnum;
        }

        if ((cfg->seed == SEED_FROM_FILE && (cfg->outputtype == otRF || cfg->outputtype == otRFmus)) ||
                (cfg->outputtype == otRF && cfg->seed != SEED_FROM_FILE && cfg->omega > 0.f)) {
            dims[5] *= 2;
        }

        if (MCX_IS_ADJOINT_TYPE(cfg->outputtype) && cfg->seed != SEED_FROM_FILE && cfg->detdir != NULL) {
            unsigned int Ns = cfg->extrasrclen + 1 - cfg->detnum;
            unsigned int Nd = cfg->detnum;
            dims[0] = cfg->dim.x;
            dims[1] = cfg->dim.y;
            dims[2] = cfg->dim.z;
            dims[3] = Ns * Nd;
            dims[4] = MCX_IS_DUAL_ADJOINT_TYPE(cfg->outputtype) ? 2 : 1;
            dims[5] = (cfg->omega > 0.f) ? 2 : 1;  /* RF: the real and imaginary parts */
        }

        if (cfg->outputformat == ofJNifti) {
            mcx_savejnii(dat, 5 + (dims[5] > 1), dims, voxelsize, name, 1, 1, cfg);
        } else {
            mcx_savebnii(dat, 5 + (dims[5] > 1), dims, voxelsize, name, 1, 1, cfg);
        }

        return;
    }

    sprintf(fname, "%s.%s", name, outputformat[(int)cfg->outputformat]);
    fp = fopen(fname, "wb");

    if (fp == NULL) {
        MCX_ERROR(-2, "can not save data to disk");
    }

    if (cfg->outputformat == ofTX3) {
        fwrite(&glformat, sizeof(unsigned int), 1, fp);
        fwrite(&(cfg->dim.x), sizeof(int), 3, fp);
    }

    fwrite(dat, sizeof(float), len * (1 + (cfg->outputtype == otRF || cfg->outputtype == otRFmus)), fp);
    fclose(fp);
}

/**
 * @brief Save detected photon data to mch format binary file
 *
 * @param[in] ppath: buffer pointing to the detected photon data (partial path etc)
 * @param[in] seeds: buffer pointing to the detected photon seed data
 * @param[in] count: number of detected photons
 * @param[in] doappend: flag if the new data is appended or write from the begining
 * @param[in] cfg: simulation configuration
 */

void mcx_savedetphoton(float* ppath, void* seeds, int count, int doappend, Config* cfg) {
    FILE* fp;
    char fhistory[MAX_FULL_PATH], filetag;

    if (cfg->outputformat == ofJNifti) {
        mcx_savejdet(ppath, seeds, count, doappend, cfg);
        return;
    }

    if (cfg->outputformat == ofBJNifti) {
        mcx_savejdb(ppath, seeds, count, doappend, cfg);
        return;
    }

    filetag = ((cfg->his.detected == 0  && cfg->his.savedphoton) ? 't' : 'h');

    if (cfg->rootpath[0]) {
        sprintf(fhistory, "%s%c%s.mc%c", cfg->rootpath, pathsep, cfg->session, filetag);
    } else {
        sprintf(fhistory, "%s.mc%c", cfg->session, filetag);
    }

    if (doappend) {
        fp = fopen(fhistory, "ab");
    } else {
        fp = fopen(fhistory, "wb");
    }

    if (fp == NULL) {
        MCX_ERROR(-2, "can not save data to disk");
    }

    fwrite(&(cfg->his), sizeof(History), 1, fp);
    fwrite(ppath, sizeof(float), count * cfg->his.colcount, fp);

    if (cfg->issaveseed && seeds != NULL) {
        fwrite(seeds, cfg->his.seedbyte, count, fp);
    }

    fclose(fp);
}

/**
 * @brief Save detected photon data to mch format binary file
 *
 * @param[in] ppath: buffer pointing to the detected photon data (partial path etc)
 * @param[in] seeds: buffer pointing to the detected photon seed data
 * @param[in] count: number of detected photons
 * @param[in] doappend: flag if the new data is appended or write from the begining
 * @param[in] cfg: simulation configuration
 */

void mcx_savejdet(float* ppath, void* seeds, uint count, int doappend, Config* cfg) {
    FILE* fp;
    char fhistory[MAX_FULL_PATH], filetag;
    cJSON* root = NULL, *obj = NULL, *hdr = NULL, *dat = NULL, *sub = NULL;
    char* jsonstr = NULL;
    int col = 0;

    root = cJSON_CreateObject();

    /* the "NIFTIHeader" section */
    cJSON_AddItemToObject(root, "MCXData", obj = cJSON_CreateObject());
    cJSON_AddItemToObject(obj, "Info", hdr = cJSON_CreateObject());
    cJSON_AddNumberToObject(hdr, "Version", cfg->his.version);
    cJSON_AddNumberToObject(hdr, "MediaNum", cfg->his.maxmedia);
    cJSON_AddNumberToObject(hdr, "DetNum", cfg->his.detnum);
    cJSON_AddNumberToObject(hdr, "ColumnNum", cfg->his.colcount);
    cJSON_AddNumberToObject(hdr, "TotalPhoton", cfg->his.totalphoton);
    cJSON_AddNumberToObject(hdr, "DetectedPhoton", count);
    cJSON_AddNumberToObject(hdr, "SavedPhoton", cfg->his.savedphoton);
    cJSON_AddNumberToObject(hdr, "LengthUnit", cfg->his.unitinmm);
    cJSON_AddNumberToObject(hdr, "SeedByte", cfg->his.seedbyte);
    cJSON_AddNumberToObject(hdr, "Normalizer", cfg->his.normalizer);
    cJSON_AddNumberToObject(hdr, "Repeat", cfg->his.respin);
    cJSON_AddNumberToObject(hdr, "SrcNum", cfg->his.srcnum);
    cJSON_AddNumberToObject(hdr, "SaveDetFlag", cfg->his.savedetflag);
    cJSON_AddNumberToObject(hdr, "TotalSource", cfg->his.totalsource);
    cJSON_AddItemToObject(hdr, "Media", sub = cJSON_CreateArray());

    for (int i = 0; i < cfg->medianum; i++) {
        cJSON_AddItemToArray(sub, dat = cJSON_CreateObject());
        cJSON_AddNumberToObject(dat, "mua", cfg->prop[i].mua / cfg->unitinmm);
        cJSON_AddNumberToObject(dat, "mus", cfg->prop[i].mus / cfg->unitinmm);
        cJSON_AddNumberToObject(dat, "g",   cfg->prop[i].g);
        cJSON_AddNumberToObject(dat, "n",   cfg->prop[i].n);
    }

    if (cfg->his.detected == 0  && cfg->his.savedphoton) {
        char colnum[] = {1, 3, 1, 1, 4};
        char* dtype[] = {"uint32", "single", "single", "uint32", "single"};
        char* dname[] = {"photonid", "p", "w0", "srcid", "iquv"};
        /* srcid is always written (the released layout); iquv only exists when
           istrajstokes extends each record to MCX_DEBUG_REC_LEN + 4 floats */
        int activecol = sizeof(colnum) - (cfg->his.colcount < MCX_DEBUG_REC_LEN + 4);
        cJSON_AddItemToObject(obj, "Trajectory", dat = cJSON_CreateObject());

        for (int id = 0; id < activecol; id++) {
            uint dims[2] = {count, colnum[id]};
            float* buf = (float*)calloc(dims[0] * dims[1], sizeof(float));

            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++) {
                    buf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                }

            cJSON_AddItemToObject(dat, dname[id], sub = cJSON_CreateObject());

            if (mcx_jdataencode(buf, 2, dims, dtype[id], 4, cfg->zipid, sub, 0, 0, cfg)) {
                MCX_ERROR(-1, "error when converting to JSON");
            }

            free(buf);
            col += dims[1];
        }
    } else {
        int colnum[] = {1, cfg->his.maxmedia, cfg->his.maxmedia, cfg->his.maxmedia, 3, 3, 1, 4};
        char* dtype[] = {"uint32", "uint32", "single", "single", "single", "single", "single", "single"};
        char* dname[] = {"detid", "nscat", "ppath", "mom", "p", "v", "w0", "s"};
        cJSON_AddItemToObject(obj, "PhotonData", dat = cJSON_CreateObject());

        for (int id = 0; id < sizeof(colnum) / sizeof(int); id++) {
            if ((cfg->savedetflag >> id) & 0x1) {
                uint dims[2] = {count, colnum[id]};
                void* val = NULL;
                float* fbuf = NULL;
                uint*  ibuf = NULL;
                int hassrcid = ((cfg->his.totalsource > 1) && id == 0); /* srcid is packed in the upper 16 bits of detid */

                if (!strcmp(dtype[id], "uint32")) {
                    ibuf = (uint*)calloc(dims[0] * dims[1], sizeof(uint));

                    for (int i = 0; i < dims[0]; i++)
                        for (int j = 0; j < dims[1]; j++) {
                            if (id == 1) { /* nscat is accumulated by the kernel as raw uint bits */
                                memcpy(ibuf + i * dims[1] + j, ppath + i * cfg->his.colcount + col + j, sizeof(uint));
                            } else {
                                ibuf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                            }
                        }

                    val = (void*)ibuf;
                } else {
                    fbuf = (float*)calloc(dims[0] * dims[1], sizeof(float));

                    for (int i = 0; i < dims[0]; i++)
                        for (int j = 0; j < dims[1]; j++) {
                            fbuf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                        }

                    val = (void*)fbuf;
                }

                if (hassrcid) {
                    uint* srcid = (uint*)calloc(dims[0] * dims[1], sizeof(uint));

                    for (int i = 0; i < dims[0] * dims[1]; i++) {
                        srcid[i] = ibuf[i] >> 16;
                        ibuf[i] &= 0xFFFF;
                    }

                    cJSON_AddItemToObject(dat, "srcid", sub = cJSON_CreateObject());

                    if (mcx_jdataencode((void*)srcid, 2, dims, dtype[id], 4, cfg->zipid, sub, 0, 0, cfg)) {
                        MCX_ERROR(-1, "error when converting to JSON");
                    }

                    free(srcid);
                }

                cJSON_AddItemToObject(dat, dname[id], sub = cJSON_CreateObject());

                if (mcx_jdataencode(val, 2, dims, dtype[id], 4, cfg->zipid, sub, 0, 0, cfg)) {
                    MCX_ERROR(-1, "error when converting to JSON");
                }

                free(val);
                col += dims[1];
            }
        }
    }

    if (cfg->issaveseed && seeds != NULL) {
        uint dims[2] = {count, cfg->his.seedbyte};
        cJSON_AddItemToObject(dat, "seed", sub = cJSON_CreateObject());

        if (mcx_jdataencode(seeds, 2, dims, "uint8", 1, cfg->zipid, sub, 0, 0, cfg)) {
            MCX_ERROR(-1, "error when converting to JSON");
        }
    }

    /* now save JSON to file */
    jsonstr = cJSON_Print(root);

    if (jsonstr == NULL) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    filetag = ((cfg->his.detected == 0  && cfg->his.savedphoton) ? 't' : 'h');

    if (cfg->rootpath[0]) {
        sprintf(fhistory, "%s%c%s_%s.jdt", cfg->rootpath, pathsep, cfg->session, (filetag == 't' ? "traj" : "detp"));
    } else {
        sprintf(fhistory, "%s_%s.jdt", cfg->session, (filetag == 't' ? "traj" : "detp"));
    }

    if (doappend) {
        fp = fopen(fhistory, "at");
    } else {
        fp = fopen(fhistory, "wt");
    }

    if (fp == NULL) {
        MCX_ERROR(-2, "can not save data to disk");
    }

    fprintf(fp, "%s\n", jsonstr);
    fclose(fp);

    if (jsonstr) {
        free(jsonstr);
    }

    if (root) {
        cJSON_Delete(root);
    }
}

/**
 * @brief Save detected photon data to a binary JData (.jdb) format file
 *
 * @param[in] ppath: buffer pointing to the detected photon data (partial path etc)
 * @param[in] seeds: buffer pointing to the detected photon seed data
 * @param[in] count: number of detected photons
 * @param[in] doappend: flag if the new data is appended or write from the begining
 * @param[in] cfg: simulation configuration
 */

void mcx_savejdb(float* ppath, void* seeds, uint count, int doappend, Config* cfg) {
    FILE* fp;
    char fhistory[MAX_FULL_PATH], filetag;
    ubjw_context_t* root = NULL;
    uchar* jsonstr = NULL;
    size_t datalen, outputlen = 0;
    int col = 0;

    datalen = (size_t)count * cfg->his.colcount * sizeof(float)
              + (size_t)count * cfg->his.seedbyte + 4096;
    jsonstr = malloc(datalen << 1);
    root = ubjw_open_memory(jsonstr, jsonstr + (datalen << 1));

    ubjw_begin_object(root, UBJ_MIXED, 0);

    /* the "MCXData" section */
    ubjw_write_key(root, "MCXData");
    ubjw_begin_object(root, UBJ_MIXED, 0);

    ubjw_write_key(root, "Info");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    UBJ_WRITE_KEY(root, "Version",        uint32, cfg->his.version);
    UBJ_WRITE_KEY(root, "MediaNum",       uint32, cfg->his.maxmedia);
    UBJ_WRITE_KEY(root, "DetNum",         uint32, cfg->his.detnum);
    UBJ_WRITE_KEY(root, "ColumnNum",      uint32, cfg->his.colcount);
    UBJ_WRITE_KEY(root, "TotalPhoton",    uint32, cfg->his.totalphoton);
    UBJ_WRITE_KEY(root, "DetectedPhoton", uint32, count);
    UBJ_WRITE_KEY(root, "SavedPhoton",    uint32, cfg->his.savedphoton);
    UBJ_WRITE_KEY(root, "LengthUnit",     single, cfg->his.unitinmm);
    UBJ_WRITE_KEY(root, "SeedByte",       uint32, cfg->his.seedbyte);
    UBJ_WRITE_KEY(root, "Normalizer",     single, cfg->his.normalizer);
    UBJ_WRITE_KEY(root, "Repeat",         int32,  cfg->his.respin);
    UBJ_WRITE_KEY(root, "SrcNum",         uint32, cfg->his.srcnum);
    UBJ_WRITE_KEY(root, "SaveDetFlag",    uint32, cfg->his.savedetflag);
    UBJ_WRITE_KEY(root, "TotalSource",    uint32, cfg->his.totalsource);
    ubjw_write_key(root, "Media");
    ubjw_begin_array(root, UBJ_MIXED, 0);

    for (int i = 0; i < cfg->medianum; i++) {
        ubjw_begin_object(root, UBJ_MIXED, 4);
        UBJ_WRITE_KEY(root, "mua", single, cfg->prop[i].mua / cfg->unitinmm);
        UBJ_WRITE_KEY(root, "mus", single, cfg->prop[i].mus / cfg->unitinmm);
        UBJ_WRITE_KEY(root, "g",   single, cfg->prop[i].g);
        UBJ_WRITE_KEY(root, "n",   single, cfg->prop[i].n);
        ubjw_end(root);
    }

    ubjw_end(root); /* end of "Media" */
    ubjw_end(root); /* end of "Info" */

    if (cfg->his.detected == 0 && cfg->his.savedphoton) {
        char colnum[] = {1, 3, 1, 1, 4};
        char* dtype[] = {"uint32", "single", "single", "uint32", "single"};
        char* dname[] = {"photonid", "p", "w0", "srcid", "iquv"};
        /* srcid is always written (the released layout); iquv only exists when
           istrajstokes extends each record to MCX_DEBUG_REC_LEN + 4 floats */
        int activecol = sizeof(colnum) - (cfg->his.colcount < MCX_DEBUG_REC_LEN + 4);

        ubjw_write_key(root, "Trajectory");
        ubjw_begin_object(root, UBJ_MIXED, 0);

        for (int id = 0; id < activecol; id++) {
            uint dims[2] = {count, colnum[id]};
            float* buf = (float*)calloc(dims[0] * dims[1], sizeof(float));

            for (int i = 0; i < (int)dims[0]; i++)
                for (int j = 0; j < (int)dims[1]; j++) {
                    buf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                }

            ubjw_write_key(root, dname[id]);
            ubjw_begin_object(root, UBJ_MIXED, 0);

            if (mcx_jdataencode(buf, 2, dims, dtype[id], 4, cfg->zipid, root, 1, 0, cfg)) {
                MCX_ERROR(-1, "error when converting to JSON");
            }

            ubjw_end(root);
            free(buf);
            col += dims[1];
        }

        if (cfg->issaveseed && seeds != NULL) {
            uint dims[2] = {count, cfg->his.seedbyte};
            ubjw_write_key(root, "seed");
            ubjw_begin_object(root, UBJ_MIXED, 0);

            if (mcx_jdataencode(seeds, 2, dims, "uint8", 1, cfg->zipid, root, 1, 0, cfg)) {
                MCX_ERROR(-1, "error when converting to JSON");
            }

            ubjw_end(root);
        }

        ubjw_end(root); /* end of "Trajectory" */
    } else {
        int colnum[] = {1, cfg->his.maxmedia, cfg->his.maxmedia, cfg->his.maxmedia, 3, 3, 1, 4};
        char* dtype[] = {"uint32", "uint32", "single", "single", "single", "single", "single", "single"};
        char* dname[] = {"detid", "nscat", "ppath", "mom", "p", "v", "w0", "s"};

        ubjw_write_key(root, "PhotonData");
        ubjw_begin_object(root, UBJ_MIXED, 0);

        for (int id = 0; id < (int)(sizeof(colnum) / sizeof(int)); id++) {
            if ((cfg->savedetflag >> id) & 0x1) {
                uint dims[2] = {count, colnum[id]};
                void* val = NULL;
                float* fbuf = NULL;
                uint*  ibuf = NULL;
                int hassrcid = ((cfg->his.totalsource > 1) && id == 0); /* srcid is packed in the upper 16 bits of detid */

                if (!strcmp(dtype[id], "uint32")) {
                    ibuf = (uint*)calloc(dims[0] * dims[1], sizeof(uint));

                    for (int i = 0; i < (int)dims[0]; i++)
                        for (int j = 0; j < (int)dims[1]; j++) {
                            if (id == 1) { /* nscat is accumulated by the kernel as raw uint bits */
                                memcpy(ibuf + i * dims[1] + j, ppath + i * cfg->his.colcount + col + j, sizeof(uint));
                            } else {
                                ibuf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                            }
                        }

                    val = (void*)ibuf;
                } else {
                    fbuf = (float*)calloc(dims[0] * dims[1], sizeof(float));

                    for (int i = 0; i < (int)dims[0]; i++)
                        for (int j = 0; j < (int)dims[1]; j++) {
                            fbuf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                        }

                    val = (void*)fbuf;
                }

                if (hassrcid) {
                    uint* srcid = (uint*)calloc(dims[0] * dims[1], sizeof(uint));

                    for (int i = 0; i < (int)(dims[0] * dims[1]); i++) {
                        srcid[i] = ibuf[i] >> 16;
                        ibuf[i] &= 0xFFFF;
                    }

                    ubjw_write_key(root, "srcid");
                    ubjw_begin_object(root, UBJ_MIXED, 0);

                    if (mcx_jdataencode((void*)srcid, 2, dims, dtype[id], 4, cfg->zipid, root, 1, 0, cfg)) {
                        MCX_ERROR(-1, "error when converting to JSON");
                    }

                    ubjw_end(root);
                    free(srcid);
                }

                ubjw_write_key(root, dname[id]);
                ubjw_begin_object(root, UBJ_MIXED, 0);

                if (mcx_jdataencode(val, 2, dims, dtype[id], 4, cfg->zipid, root, 1, 0, cfg)) {
                    MCX_ERROR(-1, "error when converting to JSON");
                }

                ubjw_end(root);
                free(val);
                col += dims[1];
            }
        }

        if (cfg->issaveseed && seeds != NULL) {
            uint dims[2] = {count, cfg->his.seedbyte};
            ubjw_write_key(root, "seed");
            ubjw_begin_object(root, UBJ_MIXED, 0);

            if (mcx_jdataencode(seeds, 2, dims, "uint8", 1, cfg->zipid, root, 1, 0, cfg)) {
                MCX_ERROR(-1, "error when converting to JSON");
            }

            ubjw_end(root);
        }

        ubjw_end(root); /* end of "PhotonData" */
    }

    ubjw_end(root); /* end of "MCXData" */
    ubjw_end(root); /* end of root object */

    outputlen = ubjw_close_context(root);

    if (jsonstr == NULL) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    filetag = ((cfg->his.detected == 0 && cfg->his.savedphoton) ? 't' : 'h');

    if (cfg->rootpath[0]) {
        sprintf(fhistory, "%s%c%s_%s.jdb", cfg->rootpath, pathsep, cfg->session, (filetag == 't' ? "traj" : "detp"));
    } else {
        sprintf(fhistory, "%s_%s.jdb", cfg->session, (filetag == 't' ? "traj" : "detp"));
    }

    if (doappend) {
        fp = fopen(fhistory, "ab");
    } else {
        fp = fopen(fhistory, "wb");
    }

    if (fp == NULL) {
        MCX_ERROR(-2, "can not save data to disk");
    }

    fwrite(jsonstr, outputlen, 1, fp);
    fclose(fp);

    if (jsonstr) {
        free(jsonstr);
    }
}

#endif

/**
 * @brief Print a message to the console or a log file
 *
 * @param[in] cfg: simulation configuration
 * @param[in] str: a string to be printed
 */

void mcx_printlog(Config* cfg, char* str) {
    if (cfg->flog) { /*stdout is 1*/
        MCX_FPRINTF(cfg->flog, "%s\n", str);
    }
}

/**
 * @brief Normalize the solution by multiplying a scaling factor
 *
 * @param[in,out] field: volumetric data before normalization
 * @param[in] scale: the scaling factor (or normalization factor) to be applied
 * @param[in] fieldlen: the length (floating point) of elements in the volume
 * @param[in] pidx: the index of the pattern to normalize when photon sharing is used
 * @param[in] srcnum: the stride between consecutive voxels when photon sharing is used
 */

void mcx_normalize(float field[], float scale, size_t fieldlen, int pidx, int srcnum) {
    int i;

    for (i = 0; i < fieldlen; i++) {
        field[i * srcnum + pidx] *= scale;
    }
}

/**
 * @brief Kahan summation: Add a sequence of finite precision floating point numbers
 *
 * Source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 * @param[in,out] sum: sum of the squence before and after adding the next element
 * @param[in,out] kahanc: a running compensation for lost low-order bits
 * @param[in] input: the next element of the sequence
 */

void mcx_kahanSum(float* sum, float* kahanc, float input) {
    float kahany = input - *kahanc;
    float kahant = *sum + kahany;
    *kahanc = kahant - *sum - kahany;
    *sum = kahant;
}

/**
* @brief Retrieve mua for different cfg.vol formats to convert fluence back to energy in post-processing
*
* @param[out] output: medium absorption coefficient for the current voxel
* @param[in] mediaid: medium index of the current voxel
* @param[in] cfg: simulation configuration
*/

float mcx_updatemua(unsigned int mediaid, Config* cfg) {
    float mua = 0.f;

    if (cfg->mediabyte <= 4) {
        mua = cfg->prop[mediaid & MED_MASK].mua;
    } else if (cfg->mediabyte == MEDIA_MUA_FLOAT) {
        mua = fabs(*((float*)&mediaid));
    } else if (cfg->mediabyte == MEDIA_ASGN_BYTE) {
        union {
            unsigned i;
            unsigned char h[4];
        } val;
        val.i = mediaid & MED_MASK;
        mua = val.h[0] * (1.f / 255.f) * (cfg->prop[2].mua - cfg->prop[1].mua) + cfg->prop[1].mua;
    } else if (cfg->mediabyte == MEDIA_AS_SHORT) {
        union {
            unsigned int i;
            unsigned short h[2];
        } val;
        val.i = mediaid & MED_MASK;
        mua = val.h[0] * (1.f / 65535.f) * (cfg->prop[2].mua - cfg->prop[1].mua) + cfg->prop[1].mua;
    }

    return mua;
}

/**
 * @brief Force flush the command line to print the message
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_flush(Config* cfg) {
#if defined(MCX_CONTAINER) && (defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER))
    mcx_matlab_flush();
#elif defined(PYBIND11_VERSION_MAJOR)
    mcx_python_flush();
#else
    fflush(cfg->flog);
#endif
}

/**
 * @brief Error reporting function
 *
 * @param[in] id: a single integer for the types of the error
 * @param[in] msg: the error message string
 * @param[in] file: the unit file name where this error is raised
 * @param[in] linenum: the line number in the file where this error is raised
 */

void mcx_error(const int id, const char* msg, const char* file, const int linenum) {
#ifdef MCX_CONTAINER
    mcx_throw_exception(id, msg, file, linenum);
#else
    MCX_FPRINTF(stdout, S_RED "\nMCX ERROR(%d):%s in unit %s:%d\n" S_RESET, id, msg, file, linenum);

    if (id == -MCX_CUDA_ERROR_LAUNCH_FAILED) {
        MCX_FPRINTF(stdout, S_RED "MCX is terminated by your graphics driver. If you use windows, \n\
please modify TdrDelay value in the registry. Please checkout FAQ #1 for more details:\n\
URL: https://mcx.space/wiki/index.cgi?Doc/FAQ\n" S_RESET);
    }

    exit(id);
#endif
}

/**
 * @brief Function to recursively create output folder
 *
 * Source: https://stackoverflow.com/questions/2336242/recursive-mkdir-system-call-on-unix
 * @param[in] dir_path: folder name to be created
 * @param[in] mode: mode of the created folder
 */

int mkpath(char* dir_path, int mode) {
    char* p = dir_path;
    p[strlen(p) + 1] = '\0';
    p[strlen(p)] = pathsep;

    for (p = strchr(dir_path + 1, pathsep); p; p = strchr(p + 1, pathsep)) {
        *p = '\0';

        if (mkdir(dir_path, mode) == -1) {
            if (errno != EEXIST) {
                *p = pathsep;
                return -1;
            }
        }

        *p = pathsep;
    }

    if (dir_path[strlen(dir_path) - 1] == pathsep) { /* p is NULL after the loop */
        dir_path[strlen(dir_path) - 1] = '\0';
    }

    return 0;
}

/**
 * @brief Function to raise a CUDA error
 *
 * @param[in] ret: CUDA function return value, non-zero means an error
 */

void mcx_assert(int ret) {
    if (!ret) {
        MCX_ERROR(-1, "assert error");  /* ret is 0 here, which would exit as a success */
    }
}


/**
 * @brief Function to parse a JSON string using cJSON and print errors if detected
 *
 * @param[in] jbuf: a JSON string
 * @param[out] output: the parsed cJSON* object; if error is detected, it returns NULL
 */

cJSON* mcx_parsejson(const char* jbuf) {
    cJSON* jroot = cJSON_Parse(jbuf);

    if (!jroot) {
        char* ptrold, *ptr = (char*)cJSON_GetErrorPtr();

        if (ptr) {
            ptrold = strstr(jbuf, ptr);
        }

        if (ptr && ptrold) {
            char* offs = (ptrold - jbuf >= 50) ? ptrold - 50 : (char*)jbuf;

            while (offs < ptrold) {
                MCX_FPRINTF(stderr, "%c", *offs);
                offs++;
            }

            MCX_FPRINTF(stderr, "<error>%.50s\n", ptrold);
        }
    }

    return jroot;
}


#ifndef MCX_CONTAINER

/**
 * @brief Read simulation settings from a configuration file (.inp or .json)
 *
 * @param[in] fname: the name of the input file (.inp or .json)
 * @param[in] cfg: simulation configuration
 */

void mcx_readconfig(char* fname, Config* cfg) {
    if (fname[0] == 0) {
        mcx_loadconfig(stdin, cfg);
    } else {
        FILE* fp = fopen(fname, "rt");

        if (fp == NULL && fname[0] != '{') {
            MCX_ERROR(-2, "can not load the specified config file");
        }

        if (strstr(fname, ".json") != NULL || fname[0] == '{') {
            char* jbuf;
            int len;
            cJSON* jroot;

            if (fp != NULL) {
                fclose(fp);
                fp = fopen(fname, "rb");
                fseek (fp, 0, SEEK_END);
                len = ftell(fp) + 1;
                jbuf = (char*)malloc(len);
                rewind(fp);

                if (fread(jbuf, len - 1, 1, fp) != 1) {
                    MCX_ERROR(-2, "reading input file is terminated");
                }

                jbuf[len - 1] = '\0';
            } else {
                jbuf = fname;
            }

            jroot = mcx_parsejson(jbuf);

            if (jroot) {
                mcx_loadjson(jroot, cfg);
                cJSON_Delete(jroot);
            } else {
                if (fp != NULL) {
                    fclose(fp);
                    free(jbuf);
                }

                MCX_ERROR(-9, "invalid JSON input file");
            }

            if (fp != NULL) {
                free(jbuf);
            }
        } else {
            mcx_loadconfig(fp, cfg);
        }

        if (fp != NULL) {
            fclose(fp);
        }

        if (cfg->session[0] == '\0') {
            strncpy(cfg->session, fname, MAX_SESSION_LENGTH - 1);
        }
    }

    if (cfg->rootpath[0] != '\0') {
        struct stat st = {0};

        if (stat((const char*)cfg->rootpath, &st) == -1) {
            if (mkpath(cfg->rootpath, 0755)) {
                MCX_ERROR(-9, "can not create output folder");
            }
        }
    }
}

/**
 * @brief Write simulation settings to an inp file
 *
 * @param[in] fname: the name of the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_writeconfig(char* fname, Config* cfg) {
    if (fname[0] == 0) {
        mcx_saveconfig(stdout, cfg);
    } else {
        FILE* fp = fopen(fname, "wt");

        if (fp == NULL) {
            MCX_ERROR(-2, "can not write to the specified config file");
        }

        mcx_saveconfig(fp, cfg);
        fclose(fp);
    }
}

#endif

/**
 * @brief Preprocess user input and prepare the cfg data structure
 *
 * This function preprocess the user input and prepare the domain for the simulation.
 * It loads the media index array from file, add detector masks for easy detection, and
 * check inconsistency between the user specified inputs.
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_preprocess(Config* cfg) {
    int isbcdet = 0;

    /* the optical properties the kernel divides by and takes logarithms of: a
       refractive index of nought (a "1.37" written as a string reads as 0) or
       an infinite scattering coefficient has every photon step forever */
    for (uint i = 0; i < cfg->medianum; i++) {
        if (!isfinite(cfg->prop[i].mua) || !isfinite(cfg->prop[i].mus) || !isfinite(cfg->prop[i].g) || !isfinite(cfg->prop[i].n) ||
                cfg->prop[i].mua < 0.f || cfg->prop[i].mus < 0.f || cfg->prop[i].n <= 0.f || fabsf(cfg->prop[i].g) > 1.f) {
            char msg[MAX_PATH_LENGTH];
            snprintf(msg, MAX_PATH_LENGTH, "medium %u: mua and mus must be finite and non-negative, g within [-1,1], and n positive (got %g %g %g %g)",
                     i, cfg->prop[i].mua, cfg->prop[i].mus, cfg->prop[i].g, cfg->prop[i].n);
            MCX_ERROR(-1, msg);
        }
    }

    double tmp = sqrt(cfg->srcdir.x * cfg->srcdir.x + cfg->srcdir.y * cfg->srcdir.y + cfg->srcdir.z * cfg->srcdir.z);

    if (tmp < EPS) {
        MCX_ERROR(-4, "source initial direction vector can not have a length of 0");
    }

    tmp = 1.f / tmp;
    cfg->srcdir.x *= tmp;
    cfg->srcdir.y *= tmp;
    cfg->srcdir.z *= tmp;

    if (cfg->debuglevel & MCX_DEBUG_MOVE_ONLY) {
        cfg->issave2pt = 0;
        cfg->issavedet = 0;
    }

    if ((cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otRF || cfg->outputtype == otRFmus || cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF)
            && cfg->seed != SEED_FROM_FILE && !(cfg->outputtype == otRF && cfg->omega > 0.f)) {
        MCX_ERROR(-6, "Jacobian output is only valid in the reply mode. Please define cfg.seed");
    }

    /**
     * Append detectors as reversed sources for adjoint/srcid=-2 mode.
     * In the mex container (mcxlab/pmcx), this is handled in the host wrapper
     * before mcx_validatecfg is called, so we skip it here to avoid double-appending.
     */
#ifndef MCX_CONTAINER

    if ((MCX_IS_ADJOINT_TYPE(cfg->outputtype) || cfg->srcid == -2) && cfg->seed != SEED_FROM_FILE && cfg->detnum > 0 && cfg->detdir != NULL) {
        unsigned int origextrasrclen = cfg->extrasrclen;

        cfg->srcdata = (ExtraSrc*)realloc(cfg->srcdata, (cfg->extrasrclen + cfg->detnum) * sizeof(ExtraSrc));
        memset(cfg->srcdata + cfg->extrasrclen, 0, cfg->detnum * sizeof(ExtraSrc));

        for (unsigned int i = 0; i < cfg->detnum; i++) {
            cfg->srcdata[origextrasrclen + i].srcpos.x = cfg->detpos[i].x;
            cfg->srcdata[origextrasrclen + i].srcpos.y = cfg->detpos[i].y;
            cfg->srcdata[origextrasrclen + i].srcpos.z = cfg->detpos[i].z;
            cfg->srcdata[origextrasrclen + i].srcpos.w = 1.f;
            cfg->srcdata[origextrasrclen + i].srcdir   = cfg->detdir[i];
            cfg->srcdata[origextrasrclen + i].srcparam1.x = cfg->detpos[i].w; /**< detector radius stored in srcparam1.x for disk source */
            cfg->srcdata[origextrasrclen + i].srcparam1.y = 0.f;
            cfg->srcdata[origextrasrclen + i].srcparam1.z = 0.f;
            cfg->srcdata[origextrasrclen + i].srcparam1.w = 0.f;
            /* srcparam2 is left zeroed; .z/.w used for precomputed voxel idx, filled later */
        }

        cfg->extrasrclen += cfg->detnum;

        if (MCX_IS_ADJOINT_TYPE(cfg->outputtype)) {
            cfg->srcid = -1;
        }
    }

#endif

    if (cfg->outputtype == otRF && cfg->seed != SEED_FROM_FILE && cfg->omega == 0.f) {
        MCX_ERROR(-6, "You must set cfg.omega (=2*pi*freq) to use outputtype 'rf' in forward mode");
    }


    // if neither trajectory or polarization is enabled, disable istrajstokes flag
    if (!(cfg->debuglevel & (MCX_DEBUG_MOVE | MCX_DEBUG_MOVE_ONLY)) || !((cfg->mediabyte <= 4) && (cfg->polmedianum > 0))) {
        cfg->istrajstokes = 0;
    }

    for (int i = 0; i < 6; i++)
        if (cfg->bc[i] && mcx_lookupindex(cfg->bc + i, boundarycond)) {
            MCX_ERROR(-4, "unknown boundary condition specifier");
        }

    for (int i = 6; i < 12; i++) {
        if (cfg->bc[i] && mcx_lookupindex(cfg->bc + i, boundarydetflag)) {
            MCX_ERROR(-4, "unknown boundary detection flags");
        }

        if (cfg->bc[i]) {
            isbcdet = 1;
        }
    }

    if (cfg->vol && cfg->polprop) {
        if (!(cfg->mediabyte <= 4)) {
            MCX_ERROR(-1, "Unsupported media format");
        }

        if (cfg->medianum != cfg->polmedianum + 1) {
            MCX_ERROR(-6, "number of particle types does not match number of media");
        }

        if (cfg->lambda == 0.f) {
            MCX_ERROR(-1, "you must specify light wavelength lambda to run polarized photon simulation");
        }

        if (cfg->srciquv.x < 0.f) {
            MCX_ERROR(-4, "initial total light intensity must not be negative");
        }

        if (cfg->srciquv.y < -1.f || cfg->srciquv.y > 1.f || cfg->srciquv.z < -1.f || cfg->srciquv.z > 1.f || cfg->srciquv.w < -1.f || cfg->srciquv.w > 1.f) {
            MCX_ERROR(-4, "initial Q, U and V must be floating-point numbers between -1 and 1");
        }

        mcx_prep_polarized(cfg); // cfg->medianum will be updated
    }

    if (cfg->medianum == 0) {
        MCX_ERROR(-4, "you must define the 'prop' field in the input structure");
    }

    if (cfg->dim.x == 0 || cfg->dim.y == 0 || cfg->dim.z == 0) {
        MCX_ERROR(-4, "the 'vol' field in the input structure can not be empty");
    }

    if ((cfg->srctype == MCX_SRC_PATTERN || cfg->srctype == MCX_SRC_PATTERN3D) && cfg->srcpattern == NULL) {
        MCX_ERROR(-4, "the 'srcpattern' field can not be empty when your 'srctype' is 'pattern'");
    }

    if (cfg->steps.x != 1.f && cfg->unitinmm == 1.f) {
        cfg->unitinmm = cfg->steps.x;
    }

    if (cfg->unitinmm != 1.f) {
        cfg->steps.x = cfg->unitinmm;
        cfg->steps.y = cfg->unitinmm;
        cfg->steps.z = cfg->unitinmm;

        for (int i = 1; i < cfg->medianum; i++) {
            cfg->prop[i].mus *= cfg->unitinmm;
            cfg->prop[i].mua *= cfg->unitinmm;
        }
    }

    if (cfg->isrowmajor) {
        /*from here on, the array is always col-major*/
        if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
            mcx_convertrow2col64((size_t*) (cfg->vol), &(cfg->dim));
        } else if (cfg->mediabyte == MEDIA_ASGN_F2H) {
            mcx_convertrow2col(cfg->vol, &(cfg->dim));
            mcx_convertrow2col(cfg->vol + cfg->dim.x * cfg->dim.y * cfg->dim.z, &(cfg->dim));
        } else {
            mcx_convertrow2col(cfg->vol, &(cfg->dim));
        }

        cfg->isrowmajor = 0;
    }

    if (cfg->issavedet && cfg->detnum == 0 && isbcdet == 0) {
        cfg->issavedet = 0;
    }

    if (cfg->issavedet == 0) {
        cfg->issaveexit = 0;
        cfg->ismomentum = 0;

        if (cfg->seed != SEED_FROM_FILE) {
            cfg->savedetflag = 0;
        }
    }

    if (cfg->respin == 0) {
        MCX_ERROR(-1, "respin number can not be 0, check your -r/--repeat input or cfg.respin value");
    }

    if (cfg->seed == SEED_FROM_FILE && cfg->seedfile[0]) {
        if (cfg->respin > 1 || cfg->respin < 0) {
            cfg->respin = 1;
            fprintf(stderr, S_RED "WARNING: respin is disabled in the replay mode\n" S_RESET);
        }
    }

    if (cfg->replaydet > (int)cfg->detnum) {
        MCX_ERROR(-4, "replay detector ID exceeds the maximum detector number");
    }

    if (cfg->replaydet == -1 && cfg->detnum == 1) {
        cfg->replaydet = 1;
    }

    if (cfg->medianum) {
        for (int i = 0; i < cfg->medianum; i++) {
            if (cfg->prop[i].mus == 0.f) {
                cfg->prop[i].mus = EPS;
                cfg->prop[i].g = 1.f;
            }
        }
    }

    if (cfg->srcpos.w == 0.f) {
        cfg->srcpos.w = 1.f;
    }

    if (cfg->extrasrclen) {
        if (cfg->srcnum > 1) {
            MCX_ERROR(-4, "simulating multiple sources currently can not be used with photon-sharing");
        }

        if (cfg->srcid > (int)cfg->extrasrclen + 1) {
            MCX_ERROR(-4, "srcid exceeds total defined source count");
        }

        for (int i = 0; i < cfg->extrasrclen; i++) {
            if (cfg->srcdata[i].srcpos.w == 0.f) {
                cfg->srcdata[i].srcpos.w = 1.f;
            }
        }
    }

    if (cfg->vol) {
        unsigned int dimxyz = cfg->dim.x * cfg->dim.y * cfg->dim.z;

        if (cfg->mediabyte <= 4) {
            unsigned int maxlabel = 0;

            for (uint i = 0; i < dimxyz; i++) {
                maxlabel = MAX(maxlabel, (cfg->vol[i] & MED_MASK));
            }

            if (cfg->medianum <= maxlabel) {
                MCX_ERROR(-4, "input media optical properties are less than the labels in the volume");
            }
        } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
            unsigned char* val = (unsigned char*)(cfg->vol);
            unsigned int* newvol = (unsigned int*)malloc(dimxyz << 3);
            union {
                unsigned char c[8];
                unsigned int  i[2];
            } b2u;

            for (int i = 0; i < dimxyz; i++) {
                b2u.c[2] = val[(i << 3) + 5]; // encoding normal vector nx, ny, nz
                b2u.c[1] = val[(i << 3) + 6];
                b2u.c[0] = val[(i << 3) + 7];

                b2u.c[5] = val[(i << 3) + 2]; // encoding reference point px, py, pz
                b2u.c[4] = val[(i << 3) + 3];
                b2u.c[3] = val[(i << 3) + 4];

                b2u.c[7] = val[(i << 3)]; // lower label and upper label
                b2u.c[6] = val[(i << 3) + 1];

                newvol[i] = b2u.i[1]; // first half: high 4 byte, second half: low 4 bytes
                newvol[i + dimxyz] = b2u.i[0];
            }

            memcpy(cfg->vol, newvol, (dimxyz << 3));
            free(newvol);
        }
    }

    // for all point sources, precompute launch voxel index and media value and store those in srcparam2.z/.w internally
    if (cfg->srctype <= MCX_SRC_CONE || cfg->srctype == MCX_SRC_ARCSINE || cfg->srctype == MCX_SRC_ZGAUSSIAN) {
        if (cfg->srcpos.x < 0.f || cfg->srcpos.y < 0.f || cfg->srcpos.z < 0.f || cfg->srcpos.x >= cfg->dim.x || cfg->srcpos.y >= cfg->dim.y || cfg->srcpos.z >= cfg->dim.z) {
            *((uint*)&cfg->srcparam2.z) = 0;
            *((uint*)&cfg->srcparam2.w) = 0;
        } else {
            uint idx1dorig = ((int)(floorf(cfg->srcpos.z)) * (cfg->dim.y * cfg->dim.x) + (int)(floorf(cfg->srcpos.y)) * cfg->dim.x + (int)(floorf(cfg->srcpos.x)));
            *((uint*)&cfg->srcparam2.z) = idx1dorig;
            *((uint*)&cfg->srcparam2.w) = (cfg->vol[idx1dorig] & MED_MASK);
        }

        if (cfg->extrasrclen) {
            for (int i = 0; i < cfg->extrasrclen; i++) {
                if (cfg->srcdata[i].srcpos.x < 0.f || cfg->srcdata[i].srcpos.y < 0.f || cfg->srcdata[i].srcpos.z < 0.f || cfg->srcdata[i].srcpos.x >= cfg->dim.x || cfg->srcdata[i].srcpos.y >= cfg->dim.y || cfg->srcdata[i].srcpos.z >= cfg->dim.z) {
                    *((uint*)&cfg->srcdata[i].srcparam2.z) = 0;
                    *((uint*)&cfg->srcdata[i].srcparam2.w) = 0;
                } else {
                    uint idx1dorig = ((int)(floorf(cfg->srcdata[i].srcpos.z)) * (cfg->dim.y * cfg->dim.x) + (int)(floorf(cfg->srcdata[i].srcpos.y)) * cfg->dim.x + (int)(floorf(cfg->srcdata[i].srcpos.x)));
                    *((uint*)&cfg->srcdata[i].srcparam2.z) = idx1dorig;
                    *((uint*)&cfg->srcdata[i].srcparam2.w) = (cfg->vol[idx1dorig] & MED_MASK);
                }
            }
        }
    }

    if (cfg->issavedet) {
        mcx_maskdet(cfg);
    }

    for (int i = 0; i < MAX_DEVICE; i++)
        if (cfg->deviceid[i] == '0') {
            cfg->deviceid[i] = '\0';
        }

    if ((cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_MUA_FLOAT || cfg->mediabyte == MEDIA_AS_HALF) && cfg->medianum < 2) {
        MCX_ERROR(-4, "the 'prop' field must contain at least 2 rows for the requested media format");
    }

    if ((cfg->mediabyte == MEDIA_ASGN_BYTE || cfg->mediabyte == MEDIA_AS_SHORT) && cfg->medianum < 3) {
        MCX_ERROR(-4, "the 'prop' field must contain at least 3 rows for the requested media format");
    }

    if (cfg->ismomentum) {
        cfg->savedetflag = SET_SAVE_MOM(cfg->savedetflag);
    }

    if (cfg->issaveexit) {
        cfg->savedetflag = SET_SAVE_PEXIT(cfg->savedetflag);
        cfg->savedetflag = SET_SAVE_VEXIT(cfg->savedetflag);
    }

    if (cfg->polmedianum) {
        cfg->savedetflag = SET_SAVE_PPATH(cfg->savedetflag);
        cfg->savedetflag = SET_SAVE_VEXIT(cfg->savedetflag);
        cfg->savedetflag = SET_SAVE_W0(cfg->savedetflag);
        cfg->savedetflag = SET_SAVE_IQUV(cfg->savedetflag);
    } else {
        cfg->savedetflag = UNSET_SAVE_IQUV(cfg->savedetflag);
    }

    if (cfg->issavedet && cfg->savedetflag == 0) {
        cfg->savedetflag = 0x5;
    }

    /** the continuous-property formats have no tissue label to sort the per-medium data by */
    if ((cfg->mediabyte >= 100 || cfg->mediabyte == MEDIA_ASGN_F2H) && cfg->savedetflag) {
        cfg->savedetflag = UNSET_SAVE_NSCAT(cfg->savedetflag);
        cfg->savedetflag = UNSET_SAVE_PPATH(cfg->savedetflag);
        cfg->savedetflag = UNSET_SAVE_MOM(cfg->savedetflag);
        cfg->savedetflag = UNSET_SAVE_IQUV(cfg->savedetflag);
    }

    if (cfg->issaveref > 1) {
        if (cfg->issavedet == 0) {
            MCX_ERROR(-4, "you must have at least two outputs if issaveref is greater than 1");
        }

        if (cfg->dim.x * cfg->dim.y * cfg->dim.z > cfg->maxdetphoton) {
            MCX_FPRINTF(cfg->flog, "you must set --maxdetphoton larger than the total size of the voxels when --issaveref is greater than 1; autocorrecting ...\n");
            cfg->maxdetphoton = cfg->dim.x * cfg->dim.y * cfg->dim.z;
        }

        cfg->savedetflag = 0x5;
    }
}

/**
 * @brief Preprocess media to prepare polarized photon simulation
 *
 * This function precompute the scattering coefficent and smatrix for
 * different sphere-medium combinations.
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_prep_polarized(Config* cfg) {
    /* precompute cosine of discretized scattering angles */
    double* mu = (double*)malloc(NANGLES * sizeof(double));

    for (int i = 0; i < NANGLES; i++) {
        mu[i] = cos(ONE_PI * i / (NANGLES - 1));
    }

    cfg->smatrix = (float4*)malloc(cfg->polmedianum * NANGLES * sizeof(float4));
    Medium* prop = cfg->prop;
    POLMedium* polprop = cfg->polprop;

    for (int i = 0; i < cfg->polmedianum; i++) {
        prop[i + 1].mua = polprop[i].mua;
        prop[i + 1].n = polprop[i].nmed;

        /* for (i-1)th sphere(r, rho, nsph)-background medium(nmed) combination, compute mus and s-matrix */
        double x, A, qsca, g;
        x = TWO_PI * polprop[i].r * polprop[i].nmed / (cfg->lambda * 1e-3); // size parameter (unitless)
        A = ONE_PI * polprop[i].r * polprop[i].r;                     // cross-sectional area in micron^2
        Mie(x, polprop[i].nsph / polprop[i].nmed, mu, cfg->smatrix + i * NANGLES, &qsca, &g);

        if (prop[i + 1].mus > EPS) {
            float target_mus = prop[i + 1].mus; // achieve target mus

            if (prop[i + 1].g < 1.f - EPS) {
                float target_musp = prop[i + 1].mus * (1.0f - prop[i + 1].g); // achieve target mus(1-g)
                target_mus = target_musp / (1.0 - g);
            }

            polprop[i].rho = target_mus / qsca / A * 1e-3;
        }

        /* compute scattering coefficient (in mm^-1) */
        prop[i + 1].mus = qsca * A * polprop[i].rho * 1e3;

        /* store anisotropy g (not used in polarized MCX simulation) */
        prop[i + 1].g = g;
    }

    free(mu);
}

#ifndef MCX_CONTAINER

/**
 * @brief Preprocess user input and prepare the volumetric domain for simulation
 *
 * This function preprocess the user input and prepare the domain for the simulation.
 * It loads the media index array from file, add detector masks for easy detection, and
 * check inconsistency between the user specified inputs.
 *
 * @param[in] filename: the name of the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_prepdomain(char* filename, Config* cfg) {
    if (filename[0] || cfg->vol) {
        if (cfg->vol == NULL) {
            mcx_loadvolume(filename, cfg, 0);

            if (cfg->shapedata && strstr(cfg->shapedata, ":") != NULL) {
                int status;

                if (cfg->mediabyte > 4) {
                    MCX_ERROR(-10, "rasterization of shapes must be used with label-based mediatype");
                }

                Grid3D grid = {&(cfg->vol), &(cfg->dim), {1.f, 1.f, 1.f}, cfg->isrowmajor};

                if (cfg->issrcfrom0) {
                    memset(&(grid.orig.x), 0, sizeof(float3));
                }

                status = mcx_parse_shapestring(&grid, cfg->shapedata);

                if (status) {
                    MCX_ERROR(status, mcx_last_shapeerror());
                }
            }
        }

        if (cfg->isdumpjson == 2) { /* dump the inputs before mcx_preprocess, whether the volume came from a file or the JSON input */
            mcx_savejdata(cfg->jsonfile, cfg);
            exit(0);
        }
    } else {
        MCX_ERROR(-4, "one must specify a binary volume file in order to run the simulation");
    }

    if (cfg->seed == SEED_FROM_FILE && cfg->seedfile[0]) {
        if (strstr(cfg->seedfile, ".jdt") != NULL || strstr(cfg->seedfile, ".jdat") != NULL) {
            mcx_loadseedjdat(cfg->seedfile, cfg);
        } else {
            mcx_loadseedfile(cfg);
        }
    }

    if (cfg->isdumpjson == 3) {
        mcx_preprocess(cfg);
        mcx_savejdata(cfg->jsonfile, cfg);
        exit(0);
    }
}

/**
 * @brief Load user inputs from a .inp input file
 *
 * This function loads user input from a simple text input format in a .inp extension
 *
 * @param[in] in: file handle to the .inp file
 * @param[in] cfg: simulation configuration
 */

void mcx_loadconfig(FILE* in, Config* cfg) {
    uint i, gates, itmp;
    size_t count;
    float dtmp;
    char filename[MAX_PATH_LENGTH] = {'\0'}, comment[MAX_FULL_PATH], strtypestr[MAX_FULL_PATH] = {'\0'}, *comm;

    if (in == stdin) {
        fprintf(stdout, "Please specify the total number of photons: [1000000]\n\t");
    }

    MCX_ASSERT(fscanf(in, "%ld", &(count) ) == 1);

    if (cfg->nphoton == 0) {
        cfg->nphoton = count;
    }

    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%ld\nPlease specify the random number generator seed: [1234567]\n\t", cfg->nphoton);
    }

    if (cfg->seed == 0) {
        MCX_ASSERT(fscanf(in, "%d", &(cfg->seed) ) == 1);
    } else {
        MCX_ASSERT(fscanf(in, "%u", &itmp ) == 1);
    }

    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%d\nPlease specify the position of the source (in grid unit): [10 10 5]\n\t", cfg->seed);
    }

    MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->srcpos.x), &(cfg->srcpos.y), &(cfg->srcpos.z) ) == 3);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (cfg->issrcfrom0 == 0 && comm != NULL && sscanf(comm, "%u", &itmp) == 1) {
        cfg->issrcfrom0 = itmp;
    }

    if (in == stdin)
        fprintf(stdout, "%f %f %f\nPlease specify the normal direction of the source fiber: [0 0 1]\n\t",
                cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z);

    if (!cfg->issrcfrom0) {
        cfg->srcpos.x--;
        cfg->srcpos.y--;
        cfg->srcpos.z--; /*convert to C index, grid center*/
    }

    MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->srcdir.x), &(cfg->srcdir.y), &(cfg->srcdir.z) ) == 3);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (comm != NULL && sscanf(comm, "%f", &dtmp) == 1) {
        cfg->srcdir.w = dtmp;
    }

    if (in == stdin)
        fprintf(stdout, "%f %f %f %f\nPlease specify the time gates (format: start end step) in seconds [0.0 1e-9 1e-10]\n\t",
                cfg->srcdir.x, cfg->srcdir.y, cfg->srcdir.z, cfg->srcdir.w);

    MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->tstart), &(cfg->tend), &(cfg->tstep) ) == 3);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin)
        fprintf(stdout, "%f %f %f\nPlease specify the path to the volume binary file:\n\t",
                cfg->tstart, cfg->tend, cfg->tstep);

    if (cfg->tstart > cfg->tend || !(cfg->tstep > 0.f)) {
        MCX_ERROR(-9, "incorrect time gate settings");
    }

    if (!((cfg->tend - cfg->tstart) / cfg->tstep < 1e6)) {
        MCX_ERROR(-9, "too many time gates: (T1-T0)/Dt must be below one million");
    }

    gates = (uint)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);

    if (cfg->maxgate == 0) {
        cfg->maxgate = gates;
    } else if (cfg->maxgate > gates) {
        cfg->maxgate = gates;
    }

    MCX_ASSERT(fscanf(in, "%1023s", filename) == 1);

    if (cfg->rootpath[0]) {
#ifdef WIN32
        sprintf(comment, "%s\\%s", cfg->rootpath, filename);
#else
        sprintf(comment, "%s/%s", cfg->rootpath, filename);
#endif
        strncpy(filename, comment, MAX_FULL_PATH - 1);
    }

    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t", filename);
    }

    MCX_ASSERT(fscanf(in, "%f %u %u %u", &(cfg->steps.x), &(cfg->dim.x), &(cfg->crop0.x), &(cfg->crop1.x)) == 4);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin)
        fprintf(stdout, "%f %d %d %d\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                cfg->steps.x, cfg->dim.x, cfg->crop0.x, cfg->crop1.x);

    MCX_ASSERT(fscanf(in, "%f %u %u %u", &(cfg->steps.y), &(cfg->dim.y), &(cfg->crop0.y), &(cfg->crop1.y)) == 4);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin)
        fprintf(stdout, "%f %d %d %d\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                cfg->steps.y, cfg->dim.y, cfg->crop0.y, cfg->crop1.y);

    MCX_ASSERT(fscanf(in, "%f %u %u %u", &(cfg->steps.z), &(cfg->dim.z), &(cfg->crop0.z), &(cfg->crop1.z)) == 4);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (cfg->steps.x != cfg->steps.y || cfg->steps.y != cfg->steps.z) {
        MCX_ERROR(-9, "MCX currently does not support anisotropic voxels");
    }

    if (cfg->steps.x != 1.f && cfg->unitinmm == 1.f) {
        cfg->unitinmm = cfg->steps.x;
    }

    if (cfg->sradius > 0.f) {
        cfg->crop0.x = MAX((uint)(cfg->srcpos.x - cfg->sradius), 0);
        cfg->crop0.y = MAX((uint)(cfg->srcpos.y - cfg->sradius), 0);
        cfg->crop0.z = MAX((uint)(cfg->srcpos.z - cfg->sradius), 0);
        cfg->crop1.x = MIN((uint)(cfg->srcpos.x + cfg->sradius), cfg->dim.x - 1);
        cfg->crop1.y = MIN((uint)(cfg->srcpos.y + cfg->sradius), cfg->dim.y - 1);
        cfg->crop1.z = MIN((uint)(cfg->srcpos.z + cfg->sradius), cfg->dim.z - 1);
    } else if (cfg->sradius == 0.f) {
        memset(&(cfg->crop0), 0, sizeof(uint3));
        memset(&(cfg->crop1), 0, sizeof(uint3));
    } else {
        /*
           if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
        */
        if (!cfg->issrcfrom0) {
            cfg->crop0.x--;
            cfg->crop0.y--;
            cfg->crop0.z--;  /*convert to C index*/
            cfg->crop1.x--;
            cfg->crop1.y--;
            cfg->crop1.z--;
        }
    }

    if (in == stdin)
        fprintf(stdout, "%f %d %d %d\nPlease specify the total types of media:\n\t",
                cfg->steps.z, cfg->dim.z, cfg->crop0.z, cfg->crop1.z);

    MCX_ASSERT(fscanf(in, "%u", &(cfg->medianum)) == 1);
    cfg->medianum++;
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%d\n", cfg->medianum);
    }

    cfg->prop = (Medium*)malloc(sizeof(Medium) * cfg->medianum);
    cfg->prop[0].mua = 0.f; /*property 0 is already air*/
    cfg->prop[0].mus = 0.f;
    cfg->prop[0].g = 1.f;
    cfg->prop[0].n = 1.f;

    for (i = 1; i < cfg->medianum; i++) {
        if (in == stdin) {
            fprintf(stdout, "Please define medium #%d: mus(1/mm), anisotropy, mua(1/mm) and refractive index: [1.01 0.01 0.04 1.37]\n\t", i);
        }

        MCX_ASSERT(fscanf(in, "%f %f %f %f", &(cfg->prop[i].mus), &(cfg->prop[i].g), &(cfg->prop[i].mua), &(cfg->prop[i].n)) == 4);
        comm = fgets(comment, MAX_PATH_LENGTH, in);

        if (in == stdin) {
            fprintf(stdout, "%f %f %f %f\n", cfg->prop[i].mus, cfg->prop[i].g, cfg->prop[i].mua, cfg->prop[i].n);
        }
    }

    if (in == stdin) {
        fprintf(stdout, "Please specify the total number of detectors and fiber diameter (in grid unit):\n\t");
    }

    MCX_ASSERT(fscanf(in, "%u %f", &(cfg->detnum), &(cfg->detradius)) == 2);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%d %f\n", cfg->detnum, cfg->detradius);
    }

    if (cfg->medianum + cfg->detnum + (cfg->extrasrclen << 2) > MAX_PROP_AND_DETECTORS) {
        MCX_ERROR(-4, "input media types plus detector number plus additional sources exceeds the maximum total (4000)");
    }

    cfg->detpos = (float4*)malloc(sizeof(float4) * cfg->detnum);

    for (i = 0; i < cfg->detnum; i++) {
        if (in == stdin) {
            fprintf(stdout, "Please define detector #%d: x,y,z (in grid unit): [5 5 5 1]\n\t", i);
        }

        MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->detpos[i].x), &(cfg->detpos[i].y), &(cfg->detpos[i].z)) == 3);
        cfg->detpos[i].w = cfg->detradius;

        if (!cfg->issrcfrom0) {
            cfg->detpos[i].x--;
            cfg->detpos[i].y--;
            cfg->detpos[i].z--;  /*convert to C index*/
        }

        comm = fgets(comment, MAX_PATH_LENGTH, in);

        if (comm != NULL && sscanf(comm, "%f", &dtmp) == 1) {
            cfg->detpos[i].w = dtmp;
        }

        if (in == stdin) {
            fprintf(stdout, "%f %f %f\n", cfg->detpos[i].x, cfg->detpos[i].y, cfg->detpos[i].z);
        }
    }

    mcx_prepdomain(filename, cfg);
    cfg->his.maxmedia = cfg->medianum - 1; /*skip media 0*/
    cfg->his.detnum = cfg->detnum;
    cfg->his.srcnum = cfg->srcnum;
    cfg->his.savedetflag = cfg->savedetflag;
    cfg->his.totalsource = ((cfg->srcid <= 0) ? cfg->extrasrclen + 1 : 1);
    //cfg->his.colcount=2+(cfg->medianum-1)*(2+(cfg->ismomentum>0))+(cfg->issaveexit>0)*6; /*column count=maxmedia+2*/

    if (in == stdin) {
        fprintf(stdout, "Please specify the source type[pencil|cone|gaussian]:\n\t");
    }

    if (fscanf(in, "%32s", strtypestr) == 1 && strtypestr[0]) {
        int srctype = mcx_keylookup(strtypestr, srctypeid);

        if (srctype == -1) {
            MCX_ERROR(-6, "the specified source type is not supported");
        }

        if (srctype >= 0) {
            comm = fgets(comment, MAX_PATH_LENGTH, in);
            cfg->srctype = srctype;

            if (in == stdin) {
                fprintf(stdout, "Please specify the source parameters set 1 (4 floating-points):\n\t");
            }

            mcx_assert(fscanf(in, "%f %f %f %f", &(cfg->srcparam1.x),
                              &(cfg->srcparam1.y), &(cfg->srcparam1.z), &(cfg->srcparam1.w)) == 4);

            if (in == stdin) {
                fprintf(stdout, "Please specify the source parameters set 2 (4 floating-points):\n\t");
            }

            mcx_assert(fscanf(in, "%f %f %f %f", &(cfg->srcparam2.x),
                              &(cfg->srcparam2.y), &(cfg->srcparam2.z), &(cfg->srcparam2.w)) == 4);

            if (cfg->srctype == MCX_SRC_PATTERN && cfg->srcparam1.w * cfg->srcparam2.w > 0) {
                char patternfile[MAX_PATH_LENGTH];
                FILE* fp;

                if (cfg->srcpattern) {
                    free(cfg->srcpattern);
                }

                cfg->srcpattern = (float*)calloc((cfg->srcparam1.w * cfg->srcparam2.w * cfg->srcnum), sizeof(float));
                MCX_ASSERT(fscanf(in, "%1023s", patternfile) == 1);
                fp = fopen(patternfile, "rb");

                if (fp == NULL) {
                    MCX_ERROR(-6, "pattern file can not be opened");
                }

                MCX_ASSERT(fread(cfg->srcpattern, cfg->srcparam1.w * cfg->srcparam2.w * cfg->srcnum, sizeof(float), fp) == sizeof(float));
                fclose(fp);
            } else if (cfg->srctype == MCX_SRC_PATTERN3D && cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z > 0) {
                char patternfile[MAX_PATH_LENGTH];
                FILE* fp;

                if (cfg->srcpattern) {
                    free(cfg->srcpattern);
                }

                cfg->srcpattern = (float*)calloc((int)(cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z * cfg->srcnum), sizeof(float));
                MCX_ASSERT(fscanf(in, "%1023s", patternfile) == 1);
                fp = fopen(patternfile, "rb");

                if (fp == NULL) {
                    MCX_ERROR(-6, "pattern file can not be opened");
                }

                MCX_ASSERT(fread(cfg->srcpattern, cfg->srcparam1.x * cfg->srcparam1.y * cfg->srcparam1.z * cfg->srcnum, sizeof(float), fp) == sizeof(float));
                fclose(fp);
            }
        } else {
            return;
        }
    } else {
        return;
    }
}

/**
 * Stops with a message naming APath unless ARow is an array of at least
 * AMinLen elements.  The readers walk a row as child->next->next...; one
 * that is short, empty or not an array at all walks off the end into NULL.
 */
static void mcx_json_needrow(cJSON* row, int minlen, const char* path) {
    char msg[MAX_PATH_LENGTH];

    if (row == NULL || !cJSON_IsArray(row) || cJSON_GetArraySize(row) < minlen) {
        snprintf(msg, MAX_PATH_LENGTH, "%s must be an array of at least %d numbers", path, minlen);
        MCX_ERROR(-1, msg);
    }
}

/**
 * @brief Load user inputs from a .json input file
 *
 * This function loads user input from a JSON format in a .json extension
 *
 * @param[out] root: json data structure pointer
 * @param[in] cfg: simulation configuration
 */

int mcx_loadjson(cJSON* root, Config* cfg) {
    int i;
    cJSON* Domain, *Optode, *Forward, *Session, *Shapes, *tmp, *subitem;
    int isnested = 0;  /* whether the current Optode.Source field is an Nsrc-row array */
    char filename[MAX_FULL_PATH] = {'\0'};
    Domain  = cJSON_GetObjectItem(root, "Domain");
    Optode  = cJSON_GetObjectItem(root, "Optode");
    Session = cJSON_GetObjectItem(root, "Session");
    Forward = cJSON_GetObjectItem(root, "Forward");

    /* RootPath locates the input files, such as Domain.VolumeFile, which is read before the Session section */
    if (Session && cfg->rootpath[0] == '\0') {
        strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath", "Session.RootPath", Session, "", valuestring), MAX_PATH_LENGTH - 1);
    }

    Shapes  = cJSON_GetObjectItem(root, "Shapes");

    if (Domain) {
        char volfile[MAX_PATH_LENGTH];
        cJSON* meds, *val, *vv;
        val = FIND_JSON_OBJ("VolumeFile", "Domain.VolumeFile", Domain);

        if (val) {
            strncpy(volfile, val->valuestring, MAX_PATH_LENGTH - 1);

            if (cfg->rootpath[0]) {
#ifdef WIN32
                sprintf(filename, "%s\\%s", cfg->rootpath, volfile);
#else
                sprintf(filename, "%s/%s", cfg->rootpath, volfile);
#endif
            } else {
                strncpy(filename, volfile, MAX_PATH_LENGTH - 1);
            }
        }

        val = FIND_JSON_OBJ("MediaFormat", "Domain.MediaFormat", Domain);

        if (!flagset['K'] && val) {
            cfg->mediabyte = mcx_keylookup((char*)FIND_JSON_KEY("MediaFormat", "Domain.MediaFormat", Domain, "byte", valuestring), mediaformat);

            if (cfg->mediabyte == -1) {
                MCX_ERROR(-1, "Unsupported media format.");
            }

            cfg->mediabyte = mediaformatid[cfg->mediabyte];
            flagset['K'] = 1;
        }

        if (!flagset['u']) {
            cfg->unitinmm = FIND_JSON_KEY("LengthUnit", "Domain.LengthUnit", Domain, 1.f, valuedouble);
        }

        meds = FIND_JSON_OBJ("Media", "Domain.Media", Domain);

        if (meds) {
            cJSON* med = meds->child;

            if (med) {
                cfg->medianum = cJSON_GetArraySize(meds);

                if (cfg->prop) {
                    free(cfg->prop);
                }

                cfg->prop = (Medium*)malloc(sizeof(Medium) * cfg->medianum);

                for (i = 0; i < cfg->medianum; i++) {
                    if (cJSON_IsObject(med)) {
                        cJSON* val = FIND_JSON_OBJ("mua", (MCX_ERROR(-1, "You must specify absorption coeff, default in 1/mm"), ""), med);

                        if (val) {
                            cfg->prop[i].mua = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("mus", (MCX_ERROR(-1, "You must specify scattering coeff, default in 1/mm"), ""), med);

                        if (val) {
                            cfg->prop[i].mus = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("g", (MCX_ERROR(-1, "You must specify anisotropy [0-1]"), ""), med);

                        if (val) {
                            cfg->prop[i].g = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("n", (MCX_ERROR(-1, "You must specify refractive index"), ""), med);

                        if (val) {
                            cfg->prop[i].n = val->valuedouble;
                        }
                    } else if (cJSON_IsArray(med)) {
                        mcx_json_needrow(med, 4, "each Domain.Media row");
                        cfg->prop[i].mua = med->child->valuedouble;
                        cfg->prop[i].mus = med->child->next->valuedouble;
                        cfg->prop[i].g = med->child->next->next->valuedouble;
                        cfg->prop[i].n = med->child->next->next->next->valuedouble;
                    } else {
                        MCX_ERROR(-1, "Session.Media must be either an array of objects or array of 4-elem numerical arrays");
                    }

                    med = med->next;

                    if (med == NULL) {
                        break;
                    }
                }
            }
        }

        meds = FIND_JSON_OBJ("MieScatter", "Domain.MieScatter", Domain);

        if (meds) {
            cJSON* med = meds->child;

            if (med) {
                cfg->polmedianum = cJSON_GetArraySize(meds);

                if (cfg->polprop) {
                    free(cfg->polprop);
                }

                cfg->polprop = (POLMedium*)malloc(cfg->polmedianum * sizeof(POLMedium));

                for (i = 0; i < cfg->polmedianum; i++) {
                    if (cJSON_IsObject(med)) {
                        cJSON* val = FIND_JSON_OBJ("mua", (MCX_ERROR(-1, "You must specify absorption coeff, default in 1/mm"), ""), med);

                        if (val) {
                            cfg->polprop[i].mua = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("radius", (MCX_ERROR(-1, "You must specify sphere particle radius, default in micron"), ""), med);

                        if (val) {
                            cfg->polprop[i].r = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("rho", (MCX_ERROR(-1, "You must specify particle volume density default in 1/micron^3"), ""), med);

                        if (val) {
                            cfg->polprop[i].rho = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("nsph", (MCX_ERROR(-1, "You must specify particle sphere refractive index"), ""), med);

                        if (val) {
                            cfg->polprop[i].nsph = val->valuedouble;
                        }

                        val = FIND_JSON_OBJ("nmed", (MCX_ERROR(-1, "You must specify particle background medium refractive index"), ""), med);

                        if (val) {
                            cfg->polprop[i].nmed = val->valuedouble;
                        }
                    } else if (cJSON_IsArray(med)) {
                        cfg->polprop[i].mua = med->child->valuedouble;
                        cfg->polprop[i].r = med->child->next->valuedouble;
                        cfg->polprop[i].rho = med->child->next->next->valuedouble;
                        cfg->polprop[i].nsph = med->child->next->next->next->valuedouble;
                        cfg->polprop[i].nmed = med->child->next->next->next->next->valuedouble;
                    } else {
                        MCX_ERROR(-1, "Domain.MieScatter must be either an array of objects or array of 5-elem numerical arrays");
                    }

                    med = med->next;

                    if (med == NULL) {
                        break;
                    }
                }
            }
        }

        val = FIND_JSON_OBJ("Dim", "Domain.Dim", Domain);

        if (val && cJSON_GetArraySize(val) >= 3) {
            cfg->dim.x = val->child->valueint;
            cfg->dim.y = val->child->next->valueint;
            cfg->dim.z = val->child->next->next->valueint;
        } else {
            if (!Shapes && (!(cfg->extrajson && cfg->extrajson[0] == '_')) ) {
                MCX_ERROR(-1, "You must specify the dimension of the volume");
            }
        }

        val = FIND_JSON_OBJ("Step", "Domain.Step", Domain);

        if (val) {
            if (cJSON_GetArraySize(val) >= 3) {
                cfg->steps.x = val->child->valuedouble;
                cfg->steps.y = val->child->next->valuedouble;
                cfg->steps.z = val->child->next->next->valuedouble;
            } else {
                MCX_ERROR(-1, "Domain::Step has incorrect element numbers");
            }
        }

        if (cfg->steps.x != cfg->steps.y || cfg->steps.y != cfg->steps.z) {
            MCX_ERROR(-9, "MCX currently does not support anisotropic voxels");
        }

        if (cfg->steps.x != 1.f && cfg->unitinmm == 1.f) {
            cfg->unitinmm = cfg->steps.x;
        }

        val = FIND_JSON_OBJ("InverseCDF", "Domain.InverseCDF", Domain);

        if (val) {
            int nphase = cJSON_GetArraySize(val);
            cfg->nphase = nphase + 2; /*left-/right-ends are excluded, so added 2*/

            if (cfg->invcdf) {
                free(cfg->invcdf);
            }

            cfg->invcdf = (float*)calloc(cfg->nphase, sizeof(float));
            cfg->invcdf[0] = -1.f; /*left end is always -1.f,right-end is always 1.f*/
            vv = val->child;

            for (i = 1; i <= nphase; i++) {
                cfg->invcdf[i] = vv->valuedouble;
                vv = vv->next;

                if (cfg->invcdf[i] < cfg->invcdf[i - 1] || cfg->invcdf[i] > 1.f || cfg->invcdf[i] < -1.f) {
                    MCX_ERROR(-1, "Domain.InverseCDF contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
                }
            }

            cfg->invcdf[nphase + 1] = 1.f; /*left end is always -1.f,right-end is always 1.f*/
            cfg->invcdf[cfg->nphase - 1] = 1.f;
        }

        val = FIND_JSON_OBJ("VoxelSize", "Domain.VoxelSize", Domain);

        if (val) {
            cJSON* voxelsize = val;
            val = FIND_JSON_OBJ("Dx", "Domain.VoxelSize.Dx", voxelsize);

            if (cJSON_GetArraySize(val) >= 1) {
                int len = cJSON_GetArraySize(val);

                if (len == 1) {
                    cfg->steps.x = -1.0f;
                } else if (len == cfg->dim.x) {
                    cfg->steps.x = -2.0f;
                } else {
                    MCX_ERROR(-1, "Domain::VoxelSize::Dx has incorrect element numbers");
                }

                cfg->dx = malloc(sizeof(float) * len);
                vv = val->child;

                for (i = 0; i < len; i++) {
                    cfg->dx[i] = vv->valuedouble;
                    vv = vv->next;
                }
            }

            val = FIND_JSON_OBJ("Dy", "Domain.VoxelSize.Dy", voxelsize);

            if (cJSON_GetArraySize(val) >= 1) {
                int len = cJSON_GetArraySize(val);

                if (len == 1) {
                    cfg->steps.y = -1.0f;
                } else if (len == cfg->dim.y) {
                    cfg->steps.y = -2.0f;
                } else {
                    MCX_ERROR(-1, "Domain::VoxelSize::Dy has incorrect element numbers");
                }

                cfg->dy = malloc(sizeof(float) * len);
                vv = val->child;

                for (i = 0; i < len; i++) {
                    cfg->dy[i] = vv->valuedouble;
                    vv = vv->next;
                }
            }

            val = FIND_JSON_OBJ("Dz", "Domain.VoxelSize.Dz", voxelsize);

            if (cJSON_GetArraySize(val) >= 1) {
                int len = cJSON_GetArraySize(val);

                if (len == 1) {
                    cfg->steps.z = -1.0f;
                } else if (len == cfg->dim.z) {
                    cfg->steps.z = -2.0f;
                } else {
                    MCX_ERROR(-1, "Domain::VoxelSize::Dz has incorrect element numbers");
                }

                cfg->dz = malloc(sizeof(float) * len);
                vv = val->child;

                for (i = 0; i < len; i++) {
                    cfg->dz[i] = vv->valuedouble;
                    vv = vv->next;
                }
            }
        }

        val = FIND_JSON_OBJ("CacheBoxP0", "Domain.CacheBoxP0", Domain);

        if (val) {
            if (cJSON_GetArraySize(val) >= 3) {
                cfg->crop0.x = val->child->valueint;
                cfg->crop0.y = val->child->next->valueint;
                cfg->crop0.z = val->child->next->next->valueint;
            } else {
                MCX_ERROR(-1, "Domain::CacheBoxP0 has incorrect element numbers");
            }
        }

        val = FIND_JSON_OBJ("CacheBoxP1", "Domain.CacheBoxP1", Domain);

        if (val) {
            if (cJSON_GetArraySize(val) >= 3) {
                cfg->crop1.x = val->child->valueint;
                cfg->crop1.y = val->child->next->valueint;
                cfg->crop1.z = val->child->next->next->valueint;
            } else {
                MCX_ERROR(-1, "Domain::CacheBoxP1 has incorrect element numbers");
            }
        }

        val = FIND_JSON_OBJ("OriginType", "Domain.OriginType", Domain);

        if (val && cfg->issrcfrom0 == 0) {
            cfg->issrcfrom0 = val->valueint;
        }

        if (cfg->sradius > 0.f) {
            cfg->crop0.x = MAX((uint)(cfg->srcpos.x - cfg->sradius), 0);
            cfg->crop0.y = MAX((uint)(cfg->srcpos.y - cfg->sradius), 0);
            cfg->crop0.z = MAX((uint)(cfg->srcpos.z - cfg->sradius), 0);
            cfg->crop1.x = MIN((uint)(cfg->srcpos.x + cfg->sradius), cfg->dim.x - 1);
            cfg->crop1.y = MIN((uint)(cfg->srcpos.y + cfg->sradius), cfg->dim.y - 1);
            cfg->crop1.z = MIN((uint)(cfg->srcpos.z + cfg->sradius), cfg->dim.z - 1);
        } else if (cfg->sradius == 0.f) {
            memset(&(cfg->crop0), 0, sizeof(uint3));
            memset(&(cfg->crop1), 0, sizeof(uint3));
        } else {
            /**
             *  if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
             */
            if (!cfg->issrcfrom0) {
                cfg->crop0.x--;
                cfg->crop0.y--;
                cfg->crop0.z--;  /*convert to C index*/
                cfg->crop1.x--;
                cfg->crop1.y--;
                cfg->crop1.z--;
            }
        }
    }

    if (Optode) {
        cJSON* dets, *vv, *src = FIND_JSON_OBJ("Source", "Optode.Source", Optode);

        if (src) {
            subitem = FIND_JSON_OBJ("Pos", "Optode.Source.Pos", src);

            if (subitem && cJSON_IsArray(subitem)) {
                if (!cJSON_IsArray(subitem->child)) {
                    cfg->extrasrclen = 0;  /* a single source; an Nsrc-row Pos must match the existing extra sources */
                }

                if (cJSON_IsArray(subitem->child)) {
                    if (cfg->srcdata && cfg->extrasrclen != cJSON_GetArraySize(subitem) - 1) {
                        MCX_ERROR(-1, "Length of sub-elements of Pos/Dir/Param1/Param2 must match");
                    }

                    if (cfg->srcdata == NULL) {
                        cfg->extrasrclen = cJSON_GetArraySize(subitem) - 1;
                    }

                    subitem = subitem->child;
                }

                mcx_json_needrow(subitem, 3, "Optode.Source.Pos");

                cfg->srcpos.x = subitem->child->valuedouble;
                cfg->srcpos.y = subitem->child->next->valuedouble;
                cfg->srcpos.z = subitem->child->next->next->valuedouble;

                if (subitem->child->next->next->next) {
                    cfg->srcpos.w = subitem->child->next->next->next->valuedouble;
                }

                if (cfg->extrasrclen > 0) {
                    int count = 0;

                    if (cfg->srcdata == NULL) {
                        cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
                    }

                    while (subitem->next && count < cfg->extrasrclen) {

                        subitem = subitem->next;
                        mcx_json_needrow(subitem, 3, "Optode.Source.Pos");
                        cfg->srcdata[count].srcpos.x = subitem->child->valuedouble - (!cfg->issrcfrom0);
                        cfg->srcdata[count].srcpos.y = subitem->child->next->valuedouble - (!cfg->issrcfrom0);
                        cfg->srcdata[count].srcpos.z = subitem->child->next->next->valuedouble - (!cfg->issrcfrom0);

                        if (subitem->child->next->next->next) {
                            cfg->srcdata[count].srcpos.w = subitem->child->next->next->next->valuedouble;
                        }

                        count++;
                    }
                }
            }

            if (FIND_JSON_OBJ("Weight", "Optode.Source.Weight", src)) {
                cfg->srcpos.w = FIND_JSON_KEY("Weight", "Optode.Source.Weight", src, cfg->srcpos.w, valuedouble);
            }

            subitem = FIND_JSON_OBJ("Dir", "Optode.Source.Dir", src);
            isnested = 0;

            if (subitem && cJSON_IsArray(subitem)) {
                if (cJSON_IsArray(subitem->child)) {
                    if (cfg->srcdata && cfg->extrasrclen != cJSON_GetArraySize(subitem) - 1) {
                        MCX_ERROR(-1, "Length of sub-elements of Pos/Dir/Param1/Param2 must match");
                    }

                    if (cfg->srcdata == NULL) {
                        cfg->extrasrclen = cJSON_GetArraySize(subitem) - 1;
                    }

                    subitem = subitem->child;
                    isnested = 1;
                }

                mcx_json_needrow(subitem, 3, "Optode.Source.Dir");

                cfg->srcdir.x = subitem->child->valuedouble;
                cfg->srcdir.y = subitem->child->next->valuedouble;
                cfg->srcdir.z = subitem->child->next->next->valuedouble;

                if (subitem->child->next->next->next) {

                    if (cJSON_IsString(subitem->child->next->next->next)) {
                        if (strcmp(subitem->child->next->next->next->valuestring, "_NaN_") == 0) {
                            cfg->srcdir.w = NAN;
                        } else if (strcmp(subitem->child->next->next->next->valuestring, "_Inf_") == 0) {
                            cfg->srcdir.w = INFINITY;
                        } else if (strcmp(subitem->child->next->next->next->valuestring, "-_Inf_") == 0) {
                            cfg->srcdir.w = -INFINITY;
                        }
                    } else {
                        cfg->srcdir.w = subitem->child->next->next->next->valuedouble;
                    }
                }

                if (cfg->extrasrclen > 0) {
                    int count = 0;

                    if (cfg->srcdata == NULL) {
                        cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
                    }

                    /* a single row given with multiple sources applies to all of them; walking
                       subitem->next of a 1-D array would read the next key of Source instead */
                    if (!isnested) {
                        for (count = 0; count < cfg->extrasrclen; count++) {
                            cfg->srcdata[count].srcdir = cfg->srcdir;
                        }
                    }

                    while (isnested && subitem->next && count < cfg->extrasrclen) {

                        subitem = subitem->next;
                        mcx_json_needrow(subitem, 3, "Optode.Source.Dir");
                        cfg->srcdata[count].srcdir.x = subitem->child->valuedouble;
                        cfg->srcdata[count].srcdir.y = subitem->child->next->valuedouble;
                        cfg->srcdata[count].srcdir.z = subitem->child->next->next->valuedouble;

                        if (subitem->child->next->next->next) {
                            if (cJSON_IsString(subitem->child->next->next->next)) {
                                if (strcmp(subitem->child->next->next->next->valuestring, "_NaN_") == 0) {
                                    cfg->srcdata[count].srcdir.w = NAN;
                                } else if (strcmp(subitem->child->next->next->next->valuestring, "_Inf_") == 0) {
                                    cfg->srcdata[count].srcdir.w = INFINITY;
                                } else if (strcmp(subitem->child->next->next->next->valuestring, "-_Inf_") == 0) {
                                    cfg->srcdata[count].srcdir.w = -INFINITY;
                                }
                            } else {
                                cfg->srcdata[count].srcdir.w = subitem->child->next->next->next->valuedouble;
                            }
                        }

                        count++;
                    }
                }
            }

            subitem = FIND_JSON_OBJ("IQUV", "Optode.Source.IQUV", src);

            if (subitem) {
                mcx_json_needrow(subitem, 4, "Optode.Source.IQUV");
                cfg->srciquv.x = subitem->child->valuedouble;
                cfg->srciquv.y = subitem->child->next->valuedouble;
                cfg->srciquv.z = subitem->child->next->next->valuedouble;
                cfg->srciquv.w = subitem->child->next->next->next->valuedouble;
            }

            if (!cfg->issrcfrom0) {
                cfg->srcpos.x--;
                cfg->srcpos.y--;
                cfg->srcpos.z--; /*convert to C index, grid center*/
            }

            subitem = FIND_JSON_OBJ("Type", "Optode.Source.Type", src);

            if (subitem) {
                cfg->srctype = mcx_keylookup((char*)FIND_JSON_KEY("Type", "Optode.Source.Type", src, "pencil", valuestring), srctypeid);
            }

            subitem = FIND_JSON_OBJ("Param1", "Optode.Source.Param1", src);
            isnested = 0;

            if (subitem && cJSON_IsArray(subitem)) {
                if (cJSON_IsArray(subitem->child)) {
                    if (cfg->srcdata && cfg->extrasrclen != cJSON_GetArraySize(subitem) - 1) {
                        MCX_ERROR(-1, "Length of sub-elements of Pos/Dir/Param1/Param2 must match");
                    }

                    if (cfg->srcdata == NULL) {
                        cfg->extrasrclen = cJSON_GetArraySize(subitem) - 1;
                    }

                    subitem = subitem->child;
                    isnested = 1;
                }

                mcx_json_needrow(subitem, 1, "Optode.Source.Param1");

                cfg->srcparam1.x = subitem->child->valuedouble;

                if (subitem->child->next) {
                    cfg->srcparam1.y = subitem->child->next->valuedouble;

                    if (subitem->child->next->next) {
                        cfg->srcparam1.z = subitem->child->next->next->valuedouble;

                        if (subitem->child->next->next->next) {
                            cfg->srcparam1.w = subitem->child->next->next->next->valuedouble;
                        }
                    }
                }

                if (cfg->extrasrclen > 0) {
                    int count = 0;

                    if (cfg->srcdata == NULL) {
                        cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
                    }

                    /* a single row given with multiple sources applies to all of them; walking
                       subitem->next of a 1-D array would read the next key of Source instead */
                    if (!isnested) {
                        for (count = 0; count < cfg->extrasrclen; count++) {
                            cfg->srcdata[count].srcparam1 = cfg->srcparam1;
                        }
                    }

                    while (isnested && subitem->next && count < cfg->extrasrclen) {

                        subitem = subitem->next;
                        mcx_json_needrow(subitem, 1, "Optode.Source.Param1");
                        cfg->srcdata[count].srcparam1.x = subitem->child->valuedouble;

                        if (subitem->child->next) {
                            cfg->srcdata[count].srcparam1.y = subitem->child->next->valuedouble;

                            if (subitem->child->next->next) {
                                cfg->srcdata[count].srcparam1.z = subitem->child->next->next->valuedouble;

                                if (subitem->child->next->next->next) {
                                    cfg->srcdata[count].srcparam1.w = subitem->child->next->next->next->valuedouble;
                                }
                            }
                        }

                        count++;
                    }
                }
            }

            subitem = FIND_JSON_OBJ("Param2", "Optode.Source.Param2", src);
            isnested = 0;

            if (subitem && cJSON_IsArray(subitem)) {
                if (cJSON_IsArray(subitem->child)) {
                    if (cfg->srcdata && cfg->extrasrclen != cJSON_GetArraySize(subitem) - 1) {
                        MCX_ERROR(-1, "Length of sub-elements of Pos/Dir/param2/Param2 must match");
                    }

                    if (cfg->srcdata == NULL) {
                        cfg->extrasrclen = cJSON_GetArraySize(subitem) - 1;
                    }

                    subitem = subitem->child;
                    isnested = 1;
                }

                mcx_json_needrow(subitem, 1, "Optode.Source.Param2");

                cfg->srcparam2.x = subitem->child->valuedouble;

                if (subitem->child->next) {
                    cfg->srcparam2.y = subitem->child->next->valuedouble;

                    if (subitem->child->next->next) {
                        cfg->srcparam2.z = subitem->child->next->next->valuedouble;

                        if (subitem->child->next->next->next) {
                            cfg->srcparam2.w = subitem->child->next->next->next->valuedouble;
                        }
                    }
                }

                if (cfg->extrasrclen > 0) {
                    int count = 0;

                    if (cfg->srcdata == NULL) {
                        cfg->srcdata = (ExtraSrc*)calloc(sizeof(ExtraSrc), cfg->extrasrclen);
                    }

                    /* a single row given with multiple sources applies to all of them; walking
                       subitem->next of a 1-D array would read the next key of Source instead */
                    if (!isnested) {
                        for (count = 0; count < cfg->extrasrclen; count++) {
                            cfg->srcdata[count].srcparam2 = cfg->srcparam2;
                        }
                    }

                    while (isnested && subitem->next && count < cfg->extrasrclen) {

                        subitem = subitem->next;
                        mcx_json_needrow(subitem, 1, "Optode.Source.Param2");
                        cfg->srcdata[count].srcparam2.x = subitem->child->valuedouble;

                        if (subitem->child->next) {
                            cfg->srcdata[count].srcparam2.y = subitem->child->next->valuedouble;

                            if (subitem->child->next->next) {
                                cfg->srcdata[count].srcparam2.z = subitem->child->next->next->valuedouble;

                                if (subitem->child->next->next->next) {
                                    cfg->srcdata[count].srcparam2.w = subitem->child->next->next->next->valuedouble;
                                }
                            }
                        }

                        count++;
                    }
                }
            }

            if (FIND_JSON_OBJ("Frequency", "Optode.Source.Frequency", src)) {
                cfg->omega = FIND_JSON_KEY("Frequency", "Optode.Source.Frequency", src, 0.f, valuedouble);
                cfg->omega *= TWO_PI;
            }

            if (FIND_JSON_OBJ("WaveLength", "Optode.Source.WaveLength", src)) {
                cfg->lambda = FIND_JSON_KEY("WaveLength", "Optode.Source.WaveLength", src, 0.f, valuedouble);
            }

            if (FIND_JSON_OBJ("SrcNum", "Optode.Source.SrcNum", src)) {
                cfg->srcnum = FIND_JSON_KEY("SrcNum", "Optode.Source.SrcNum", src, cfg->srcnum, valueint);

                /* read as an int into an unsigned count: nought or less sized the
                   normalization and output arrays at nothing, or wrapped around */
                if (FIND_JSON_KEY("SrcNum", "Optode.Source.SrcNum", src, 1, valuedouble) < 1 || cfg->srcnum > 1000000) {
                    MCX_ERROR(-1, "Optode.Source.SrcNum (the number of patterns) must be between 1 and 1000000");
                }
            }

            if (FIND_JSON_OBJ("ID", "Optode.Source.ID", src)) {
                cfg->srcid = FIND_JSON_KEY("ID", "Optode.Source.ID", src, cfg->srcid, valueint);
            }

            subitem = FIND_JSON_OBJ("Pattern", "Optode.Source.Pattern", src);

            if (subitem) {
                if (FIND_JSON_OBJ("_ArrayZipData_", "Optode.Source.Pattern._ArrayZipData_", subitem)) {
                    int ndim;
                    uint dims[3] = {1, 1, 1};
                    char* type = NULL;

                    if (cfg->srcpattern) {
                        free(cfg->srcpattern);
                    }

                    mcx_jdatadecode((void**)&cfg->srcpattern, &ndim, dims, 3, &type, subitem, cfg);

                    if (type == NULL || strcmp(type, "single")) {
                        if (cfg->srcpattern) {
                            free(cfg->srcpattern);
                        }

                        MCX_ERROR(-1, "Optode.Source.Pattern JData-annotated array must be in the 'single' format");
                    }

                    if (ndim == 3 && dims[2] > 1 && dims[0] > 1 && cfg->srctype == MCX_SRC_PATTERN) { /* a 3-D pattern3d array is a single pattern */
                        cfg->srcnum = dims[0];
                    }

                    /* the JData array is row-major, the source patterns are used in column-major order */
                    mcx_convertrow2col((unsigned int*)cfg->srcpattern, (uint3*)dims);
                } else {
                    int nx = FIND_JSON_KEY("Nx", "Optode.Source.Pattern.Nx", subitem, 0, valueint);
                    int ny = FIND_JSON_KEY("Ny", "Optode.Source.Pattern.Ny", subitem, 0, valueint);
                    int nz = FIND_JSON_KEY("Nz", "Optode.Source.Pattern.Nz", subitem, 1, valueint);

                    if (nx > 0 && ny > 0) {
                        cJSON* pat = FIND_JSON_OBJ("Data", "Optode.Source.Pattern.Data", subitem);

                        if (pat && pat->child) {
                            int i;
                            pat = pat->child;

                            if (cfg->srcpattern) {
                                free(cfg->srcpattern);
                            }

                            cfg->srcpattern = (float*)calloc(nx * ny * nz * cfg->srcnum, sizeof(float));

                            for (i = 0; i < nx * ny * nz * cfg->srcnum; i++) {
                                if (pat == NULL) {
                                    MCX_ERROR(-1, "Incomplete pattern data");
                                }

                                cfg->srcpattern[i] = pat->valuedouble;
                                pat = pat->next;
                            }
                        } else if (pat) {
                            FILE* fid = fopen(pat->valuestring, "rb");

                            if (fid != NULL) {
                                if (cfg->srcpattern) {
                                    free(cfg->srcpattern);
                                }

                                cfg->srcpattern = (float*)calloc(nx * ny * nz * cfg->srcnum, sizeof(float));
                                fread((void*)cfg->srcpattern, sizeof(float), nx * ny * nz * cfg->srcnum, fid);
                                fclose(fid);
                            }
                        }
                    }
                }
            }

            subitem = FIND_JSON_OBJ("AngleInverseCDF", "Optode.Source.AngleInverseCDF", src);

            if (!subitem && Domain) { /* older mcx2json/pmcx versions saved it in Domain */
                subitem = FIND_JSON_OBJ("AngleInverseCDF", "Domain.AngleInverseCDF", Domain);
            }

            if (subitem) {
                int nangle = cJSON_GetArraySize(subitem);

                if (cfg->angleinvcdf) {
                    free(cfg->angleinvcdf);
                }

                cfg->nangle = nangle;
                cfg->angleinvcdf = (float*)calloc(cfg->nangle, sizeof(float));
                vv = subitem->child;

                for (i = 0; i < nangle; i++) {
                    cfg->angleinvcdf[i] = vv->valuedouble;
                    vv = vv->next;

                    if ((i > 0 && cfg->angleinvcdf[i] < cfg->angleinvcdf[i - 1]) || cfg->angleinvcdf[i] > 1.f || cfg->angleinvcdf[i] < 0.f) {
                        MCX_ERROR(-1, "Optode.Source.AngleInverseCDF contains invalid data; it must be a monotonically increasing vector with all values between 0 and 1");
                    }
                }
            }

        }

        dets = FIND_JSON_OBJ("Detector", "Optode.Detector", Optode);

        if (dets) {
            cJSON* det = dets->child;

            if (det) {
                cfg->detnum = cJSON_GetArraySize(dets);
                cfg->detpos = (float4*)malloc(sizeof(float4) * cfg->detnum);

                for (i = 0; i < cfg->detnum; i++) {
                    cJSON* pos = dets, *rad = NULL, *dir = NULL;
                    rad = FIND_JSON_OBJ("R", "Optode.Detector.R", det);
                    dir = FIND_JSON_OBJ("Dir", "Optode.Detector.Dir", det);

                    if (cJSON_IsObject(det)) { /* {"Pos":[..],"R":r,"Dir":[..]}; may carry more than 2 fields */
                        pos = FIND_JSON_OBJ("Pos", "Optode.Detector.Pos", det);
                    }

                    if (pos) {
                        mcx_json_needrow(pos, 3, "Optode.Detector.Pos");
                        cfg->detpos[i].x = pos->child->valuedouble;
                        cfg->detpos[i].y = pos->child->next->valuedouble;
                        cfg->detpos[i].z = pos->child->next->next->valuedouble;
                    }

                    if (rad) {
                        cfg->detpos[i].w = rad->valuedouble;
                    }

                    if (dir && cJSON_IsArray(dir)) {
                        if (cfg->detdir == NULL) {
                            cfg->detdir = (float4*)calloc(cfg->detnum, sizeof(float4));
                        }

                        mcx_json_needrow(dir, 3, "Optode.Detector.Dir");
                        cfg->detdir[i].x = dir->child->valuedouble;
                        cfg->detdir[i].y = dir->child->next->valuedouble;
                        cfg->detdir[i].z = dir->child->next->next->valuedouble;
                        cfg->detdir[i].w = (dir->child->next->next->next) ? dir->child->next->next->next->valuedouble : 0.f;
                    }

                    if (!cfg->issrcfrom0) {
                        cfg->detpos[i].x--;
                        cfg->detpos[i].y--;
                        cfg->detpos[i].z--;  /*convert to C index*/
                    }

                    det = det->next;

                    if (det == NULL) {
                        break;
                    }
                }
            }
        }
    }

    if (cfg->medianum + cfg->detnum + (cfg->extrasrclen << 2) > MAX_PROP_AND_DETECTORS) {
        MCX_ERROR(-4, "input media types plus detector number plus additional sources exceeds the maximum total (4000)");
    }

    if (Session) {
        char val[1];

        if (!flagset['E']) {
            cfg->seed = FIND_JSON_KEY("RNGSeed", "Session.RNGSeed", Session, -1, valueint);
        }

        if (!flagset['n']) {
            cfg->nphoton = FIND_JSON_KEY("Photons", "Session.Photons", Session, 0, valuedouble);
        }

        if (!flagset['H']) {
            cfg->maxdetphoton = FIND_JSON_KEY("MaxDetPhoton", "Session.MaxDetPhoton", Session, cfg->maxdetphoton, valuedouble);
        }

        if (cfg->session[0] == '\0') {
            strncpy(cfg->session, FIND_JSON_KEY("ID", "Session.ID", Session, "default", valuestring), MAX_SESSION_LENGTH - 1);
        }

        if (cfg->rootpath[0] == '\0') {
            strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath", "Session.RootPath", Session, "", valuestring), MAX_PATH_LENGTH - 1);
        }

        if (!flagset['B']) {
            char* bc = FIND_JSON_KEY("BCFlags", "Session.BCFlags", Session, NULL, valuestring);

            if (bc) {
                strncpy(cfg->bc, bc, 12);
            }
        }

        if (!flagset['b']) {
            cfg->isreflect = FIND_JSON_KEY("DoMismatch", "Session.DoMismatch", Session, cfg->isreflect, valueint);
        }

        if (!flagset['S']) {
            cfg->issave2pt = FIND_JSON_KEY("DoSaveVolume", "Session.DoSaveVolume", Session, cfg->issave2pt, valueint);
        }

        if (!flagset['U']) {
            cfg->isnormalized = FIND_JSON_KEY("DoNormalize", "Session.DoNormalize", Session, cfg->isnormalized, valueint);
        }

        if (!flagset['d']) {
            cfg->issavedet = FIND_JSON_KEY("DoPartialPath", "Session.DoPartialPath", Session, cfg->issavedet, valueint);
        }

        if (!flagset['X']) {
            cfg->issaveref = FIND_JSON_KEY("DoSaveRef", "Session.DoSaveRef", Session, cfg->issaveref, valueint);
        }

        if (!flagset['x']) {
            cfg->issaveexit = FIND_JSON_KEY("DoSaveExit", "Session.DoSaveExit", Session, cfg->issaveexit, valueint);
        }

        if (!flagset['q']) {
            cfg->issaveseed = FIND_JSON_KEY("DoSaveSeed", "Session.DoSaveSeed", Session, cfg->issaveseed, valueint);
        }

        if (!flagset['A']) {
            cfg->autopilot = FIND_JSON_KEY("DoAutoThread", "Session.DoAutoThread", Session, cfg->autopilot, valueint);
        }

        if (!flagset['m']) {
            cfg->ismomentum = FIND_JSON_KEY("DoDCS", "Session.DoDCS", Session, cfg->ismomentum, valueint);
        }

        if (!flagset['V']) {
            cfg->isspecular = FIND_JSON_KEY("DoSpecular", "Session.DoSpecular", Session, cfg->isspecular, valueint);
        }

        if (!flagset['D']) {
            if (FIND_JSON_KEY("DebugFlag", "Session.DebugFlag", Session, "", valuestring)) {
                cfg->debuglevel = mcx_parsedebugopt(FIND_JSON_KEY("DebugFlag", "Session.DebugFlag", Session, "", valuestring), debugflag);
            } else {
                cfg->debuglevel = FIND_JSON_KEY("DebugFlag", "Session.DebugFlag", Session, 0, valueint);
            }
        }

        if (!flagset['w']) {
            if (FIND_JSON_KEY("SaveDataMask", "Session.SaveDataMask", Session, "DP", valuestring)) {
                cfg->savedetflag = mcx_parsedebugopt(FIND_JSON_KEY("SaveDataMask", "Session.SaveDataMask", Session, "DP", valuestring), saveflag);
            } else {
                cfg->savedetflag = FIND_JSON_KEY("SaveDataMask", "Session.SaveDataMask", Session, 5, valueint);
            }
        }

        if (!flagset['F']) {
            cfg->outputformat = mcx_keylookup((char*)FIND_JSON_KEY("OutputFormat", "Session.OutputFormat", Session, "jnii", valuestring), outputformat);
        }

        if (cfg->outputformat < 0) {
            MCX_ERROR(-2, "the specified output format is not recognized");
        }

        strncpy(val, FIND_JSON_KEY("OutputType", "Session.OutputType", Session, outputtype + cfg->outputtype, valuestring), 1);

        if (mcx_lookupindex(val, outputtype)) {
            MCX_ERROR(-2, "the specified output data type is not recognized");
        }

        if (!flagset['O']) {
            cfg->outputtype = val[0];
        }

        if (!flagset['e']) {
            cfg->minenergy = FIND_JSON_KEY("MinEnergy", "Session.MinEnergy", Session, cfg->minenergy, valuedouble);
        }
    }

    if (Forward) {
        uint gates;
        /* missing keys keep the current values, so that a partial --json fragment does not reset them */
        cfg->tstart = FIND_JSON_KEY("T0", "Forward.T0", Forward, cfg->tstart, valuedouble);
        cfg->tend  = FIND_JSON_KEY("T1", "Forward.T1", Forward, cfg->tend, valuedouble);
        cfg->tstep = FIND_JSON_KEY("Dt", "Forward.Dt", Forward, cfg->tstep, valuedouble);

        if (cfg->tstart > cfg->tend || !(cfg->tstep > 0.f)) {
            MCX_ERROR(-9, "incorrect time gate settings");
        }

        if (!((cfg->tend - cfg->tstart) / cfg->tstep < 1e6)) {
            MCX_ERROR(-9, "too many time gates: (T1-T0)/Dt must be below one million");
        }

        gates = (uint)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);

        if (cfg->maxgate == 0) {
            cfg->maxgate = gates;
        } else if (cfg->maxgate > gates) {
            cfg->maxgate = gates;
        }
    }

    if (filename[0] == '\0') {
        if (Shapes) {
            if (!FIND_JSON_OBJ("_ArraySize_", "Volume._ArraySize_", Shapes) && !cfg->shapedata) {
                cfg->shapedata = cJSON_Print(Shapes);
            }

            if (cfg->extrajson && cfg->extrajson[0] == '_') {
                if (cfg->shapedata) {
                    free(cfg->shapedata);
                }

                cfg->shapedata = cJSON_Print(Shapes);
            }

            if (FIND_JSON_OBJ("_ArraySize_", "Volume._ArraySize_", Shapes)) {
                int ndim;
                uint dims[4] = {1, 1, 1, 1};
                char* type = NULL, *buf = NULL;

                if (mcx_jdatadecode((void**)&buf, &ndim, dims, 4, &type, Shapes, cfg) == 0) {
                    memcpy(&(cfg->dim.x), dims, sizeof(uint) * 3);
                    mcx_loadvolume(buf, cfg, 1);
                }

                if (buf) {
                    free(buf);
                }
            } else {
                int status;
                Grid3D grid = {&(cfg->vol), &(cfg->dim), {1.f, 1.f, 1.f}, cfg->isrowmajor};

                if (cfg->issrcfrom0) {
                    memset(&(grid.orig.x), 0, sizeof(float3));
                }

                status = mcx_parse_jsonshapes(root, &grid);

                if (status) {
                    MCX_ERROR(status, mcx_last_shapeerror());
                }

                /* shapes give one 32-bit label per voxel; the two formats that
                   store two words per voxel would have the kernel read a
                   volume twice the size of the one rasterized */
                if (cfg->mediabyte == MEDIA_2LABEL_SPLIT || cfg->mediabyte == MEDIA_ASGN_F2H) {
                    MCX_ERROR(-1, "Domain.MediaFormat asgn_float and svmc need a volume array or file; Shapes give one label per voxel");
                }
            }
        } else if (!(cfg->extrajson && cfg->extrajson[0] == '_')) {
            MCX_ERROR(-1, "You must either define Domain.VolumeFile, or define a Shapes section");
        }
    } else if (Shapes) {
        /* shape objects next to Domain.VolumeFile (mcx2json writes them for cfg.vol plus cfg.shapes)
           are rasterized over the loaded volume, the same way as -P/--shapes */
        cJSON* shaperoot;

        if (FIND_JSON_OBJ("_ArraySize_", "Volume._ArraySize_", Shapes) || cfg->shapedata) {
            MCX_ERROR(-1, "You can not specify both Domain.VolumeFile and Shapes sections");
        }

        shaperoot = cJSON_CreateObject();
        cJSON_AddItemReferenceToObject(shaperoot, "Shapes", Shapes);
        cfg->shapedata = cJSON_PrintUnformatted(shaperoot);
        cJSON_Delete(shaperoot);
    }

    mcx_prepdomain(filename, cfg);
    cfg->his.maxmedia = cfg->medianum - 1; /*skip media 0*/
    cfg->his.detnum = cfg->detnum;
    cfg->his.srcnum = cfg->srcnum;
    cfg->his.savedetflag = cfg->savedetflag;
    cfg->his.totalsource = ((cfg->srcid <= 0) ? cfg->extrasrclen + 1 : 1);
    //cfg->his.colcount=2+(cfg->medianum-1)*(2+(cfg->ismomentum>0))+(cfg->issaveexit>0)*6; /*column count=maxmedia+2*/
    return 0;
}

/**
 * @brief Save simulation settings to an inp file
 *
 * @param[in] out: handle to the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_saveconfig(FILE* out, Config* cfg) {
    uint i;

    fprintf(out, "%ld\n", (cfg->nphoton) );
    fprintf(out, "%d\n", (cfg->seed) );
    fprintf(out, "%f %f %f\n", (cfg->srcpos.x), (cfg->srcpos.y), (cfg->srcpos.z) );
    fprintf(out, "%f %f %f\n", (cfg->srcdir.x), (cfg->srcdir.y), (cfg->srcdir.z) );
    fprintf(out, "%e %e %e\n", (cfg->tstart), (cfg->tend), (cfg->tstep) );
    fprintf(out, "%f %d %d %d\n", (cfg->steps.x), (cfg->dim.x), (cfg->crop0.x), (cfg->crop1.x));
    fprintf(out, "%f %d %d %d\n", (cfg->steps.y), (cfg->dim.y), (cfg->crop0.y), (cfg->crop1.y));
    fprintf(out, "%f %d %d %d\n", (cfg->steps.z), (cfg->dim.z), (cfg->crop0.z), (cfg->crop1.z));
    fprintf(out, "%d\n", (cfg->medianum));

    for (i = 0; i < cfg->medianum; i++) {
        fprintf(out, "%f %f %f %f\n", (cfg->prop[i].mus), (cfg->prop[i].g), (cfg->prop[i].mua), (cfg->prop[i].n));
    }

    fprintf(out, "%d", (cfg->detnum));

    for (i = 0; i < cfg->detnum; i++) {
        fprintf(out, "%f %f %f %f\n", (cfg->detpos[i].x), (cfg->detpos[i].y), (cfg->detpos[i].z), (cfg->detpos[i].w));
    }
}

/**
 * @brief Create the JSON array of a source direction, writing a NaN or Inf focal length as "_NaN_"/"_Inf_"
 *
 * cJSON prints a non-finite number as null; the "Dir" reader of mcx_loadjson accepts these strings
 *
 * @param[in] dir: the 4-element source direction vector
 */

static cJSON* mcx_dirtojson(const float* dir) {
    cJSON* arr = cJSON_CreateFloatArray(dir, 3);

    if (isnan(dir[3])) {
        cJSON_AddItemToArray(arr, cJSON_CreateString("_NaN_"));
    } else if (isinf(dir[3])) {
        cJSON_AddItemToArray(arr, cJSON_CreateString(dir[3] > 0.f ? "_Inf_" : "-_Inf_"));
    } else {
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(dir[3]));
    }

    return arr;
}

/**
 * @brief Save simulation settings to an inp file
 *
 * @param[in] out: handle to the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_savejdata(char* filename, Config* cfg) {
    cJSON* root = NULL, *obj = NULL, *sub = NULL, *tmp = NULL;
    char* jsonstr = NULL;
    root = cJSON_CreateObject();

    /* the "Session" section */
    cJSON_AddItemToObject(root, "Session", obj = cJSON_CreateObject());
    cJSON_AddStringToObject(obj, "ID", cfg->session);
    cJSON_AddNumberToObject(obj, "Photons", cfg->nphoton);
    cJSON_AddNumberToObject(obj, "RNGSeed", (uint)cfg->seed);

    if (cfg->isreflect > 1) {
        cJSON_AddNumberToObject(obj, "DoMismatch", cfg->isreflect);
    } else {
        cJSON_AddBoolToObject(obj, "DoMismatch", cfg->isreflect);
    }

    cJSON_AddBoolToObject(obj, "DoSaveVolume", cfg->issave2pt);

    if (cfg->isreflect > 1) {
        cJSON_AddNumberToObject(obj, "DoNormalize", cfg->isnormalized);
    } else {
        cJSON_AddBoolToObject(obj, "DoNormalize", cfg->isnormalized);
    }

    cJSON_AddBoolToObject(obj, "DoPartialPath", cfg->issavedet);

    if (cfg->issaveref) {
        cJSON_AddNumberToObject(obj, "DoSaveRef", cfg->issaveref);
    } else {
        cJSON_AddBoolToObject(obj, "DoSaveRef", cfg->issaveref);
    }

    cJSON_AddBoolToObject(obj, "DoSaveExit", cfg->issaveexit);
    cJSON_AddBoolToObject(obj, "DoSaveSeed", cfg->issaveseed);
    cJSON_AddBoolToObject(obj, "DoAutoThread", cfg->autopilot);
    cJSON_AddBoolToObject(obj, "DoDCS", cfg->ismomentum);
    cJSON_AddBoolToObject(obj, "DoSpecular", cfg->isspecular);

    if (cfg->rootpath[0] != '\0') {
        cJSON_AddStringToObject(obj, "RootPath", cfg->rootpath);
    }

    cJSON_AddNumberToObject(obj, "DebugFlag", cfg->debuglevel);
    cJSON_AddNumberToObject(obj, "SaveDataMask", cfg->savedetflag);
    cJSON_AddNumberToObject(obj, "MaxDetPhoton", cfg->maxdetphoton);

    if (cfg->minenergy > 0.f) {
        cJSON_AddNumberToObject(obj, "MinEnergy", cfg->minenergy);
    }

    {
        /* mcx_preprocess converts the boundary flags to indices of boundarycond/boundarydetflag */
        char bcstr[13] = {'\0'};
        int hasbc = 0;

        for (int i = 0; i < 12; i++) {
            char c = cfg->bc[i];
            const char* table = (i < 6) ? boundarycond : boundarydetflag;
            int nentry = (int)strlen(table);

            bcstr[i] = (c >= 0 && c < nentry) ? table[(int)c] : c;
            hasbc |= (c != 0 && bcstr[i] != table[0]);
        }

        if (hasbc) {
            cJSON_AddStringToObject(obj, "BCFlags", bcstr);
        }
    }

    if (cfg->outputformat >= 0) {
        cJSON_AddStringToObject(obj, "OutputFormat", outputformat[(int)cfg->outputformat]);
    }

    if (cfg->outputtype >= 0) {
        char outputtypestr[2] = {'\0'};
        outputtypestr[0] = outputtype[(int)cfg->outputtype];
        cJSON_AddStringToObject(obj, "OutputType", outputtypestr);
    }

    /* the "Forward" section */
    cJSON_AddItemToObject(root, "Forward", obj = cJSON_CreateObject());
    cJSON_AddNumberToObject(obj, "T0", cfg->tstart);
    cJSON_AddNumberToObject(obj, "T1", cfg->tend);
    cJSON_AddNumberToObject(obj, "Dt", cfg->tstep);

    /* the "Domain" section */
    cJSON_AddItemToObject(root, "Domain", obj = cJSON_CreateObject());

    for (int i = 0; i < sizeof(mediaformatid) / sizeof(int); i++) {
        if (cfg->mediabyte == mediaformatid[i]) {
            cJSON_AddStringToObject(obj, "MediaFormat", mediaformat[i]);
            break;
        }
    }

    cJSON_AddNumberToObject(obj, "LengthUnit", cfg->unitinmm);
    cJSON_AddItemToObject(obj, "Media", sub = cJSON_CreateArray());

    for (int i = 0; i < cfg->medianum; i++) {
        cJSON_AddItemToArray(sub, tmp = cJSON_CreateObject());
        cJSON_AddNumberToObject(tmp, "mua", cfg->prop[i].mua / cfg->unitinmm);
        cJSON_AddNumberToObject(tmp, "mus", cfg->prop[i].mus / cfg->unitinmm);
        cJSON_AddNumberToObject(tmp, "g",   cfg->prop[i].g);
        cJSON_AddNumberToObject(tmp, "n",   cfg->prop[i].n);
    }

    if (cfg->polmedianum > 0 && cfg->polprop) {
        cJSON_AddItemToObject(obj, "MieScatter", sub = cJSON_CreateArray());

        for (int i = 0; i < cfg->polmedianum; i++) {
            cJSON_AddItemToArray(sub, tmp = cJSON_CreateObject());
            cJSON_AddNumberToObject(tmp, "mua", cfg->polprop[i].mua);
            cJSON_AddNumberToObject(tmp, "radius", cfg->polprop[i].r);
            cJSON_AddNumberToObject(tmp, "rho", cfg->polprop[i].rho);
            cJSON_AddNumberToObject(tmp, "nsph", cfg->polprop[i].nsph);
            cJSON_AddNumberToObject(tmp, "nmed", cfg->polprop[i].nmed);
        }
    }

    if (cfg->nphase > 2 && cfg->invcdf) { /* the reader adds the -1 and 1 end points */
        cJSON_AddItemToObject(obj, "InverseCDF", cJSON_CreateFloatArray(cfg->invcdf + 1, cfg->nphase - 2));
    }

    cJSON_AddItemToObject(obj, "Dim", cJSON_CreateIntArray((int*) & (cfg->dim.x), 3));
    cJSON_AddNumberToObject(obj, "OriginType", 1);

    /* the "Optode" section */
    cJSON_AddItemToObject(root, "Optode", obj = cJSON_CreateObject());
    cJSON_AddItemToObject(obj, "Source", sub = cJSON_CreateObject());

    if (cfg->srctype >= 0) {
        cJSON_AddStringToObject(sub, "Type", srctypeid[(int)cfg->srctype]);
    }

    {
        /* additional sources are written as Nsrc x 4 Pos/Dir/Param1/Param2 arrays, with the source
           weight in Pos(:,4); detectors that mcx_preprocess appended as adjoint sources are left
           out, as they are appended again when the file is loaded */
        int nextra = cfg->extrasrclen;

#ifndef MCX_CONTAINER

        if ((MCX_IS_ADJOINT_TYPE(cfg->outputtype) || cfg->srcid == -2) && cfg->seed != SEED_FROM_FILE && cfg->detnum > 0 && cfg->detdir != NULL) {
            nextra -= cfg->detnum;
        }

#endif

        if (nextra > 0 && cfg->srcdata) {
            cJSON* srcfield[4];
            const char* fieldname[4] = {"Pos", "Dir", "Param1", "Param2"};

            const float* first[4] = {&(cfg->srcpos.x), &(cfg->srcdir.x), &(cfg->srcparam1.x), &(cfg->srcparam2.x)};

            for (int j = 0; j < 4; j++) {
                srcfield[j] = cJSON_CreateArray();
                cJSON_AddItemToArray(srcfield[j], (j == 1) ? mcx_dirtojson(first[j]) : cJSON_CreateFloatArray(first[j], 4));
            }

            for (int i = 0; i < nextra; i++) {
                cJSON_AddItemToArray(srcfield[0], cJSON_CreateFloatArray(&(cfg->srcdata[i].srcpos.x), 4));
                cJSON_AddItemToArray(srcfield[1], mcx_dirtojson(&(cfg->srcdata[i].srcdir.x)));
                cJSON_AddItemToArray(srcfield[2], cJSON_CreateFloatArray(&(cfg->srcdata[i].srcparam1.x), 4));
                cJSON_AddItemToArray(srcfield[3], cJSON_CreateFloatArray(&(cfg->srcdata[i].srcparam2.x), 4));
            }

            for (int j = 0; j < 4; j++) {
                cJSON_AddItemToObject(sub, fieldname[j], srcfield[j]);
            }
        } else {
            cJSON_AddItemToObject(sub, "Pos", cJSON_CreateFloatArray(&(cfg->srcpos.x), 3));

            if (cfg->srcpos.w != 1.f) {
                cJSON_AddNumberToObject(sub, "Weight", cfg->srcpos.w);
            }
            cJSON_AddItemToObject(sub, "Dir", mcx_dirtojson(&(cfg->srcdir.x)));
            cJSON_AddItemToObject(sub, "Param1", cJSON_CreateFloatArray(&(cfg->srcparam1.x), 4));
            cJSON_AddItemToObject(sub, "Param2", cJSON_CreateFloatArray(&(cfg->srcparam2.x), 4));
        }
    }
    cJSON_AddNumberToObject(sub, "SrcNum", cfg->srcnum);

    if (cfg->lambda > 0.f) {
        cJSON_AddNumberToObject(sub, "WaveLength", cfg->lambda);
    }

    if (cfg->polmedianum > 0) {
        cJSON_AddItemToObject(sub, "IQUV", cJSON_CreateFloatArray(&(cfg->srciquv.x), 4));
    }

    if (cfg->omega > 0.f) { /* the file stores the frequency in Hz, cfg->omega is 2*pi*f */
        cJSON_AddNumberToObject(sub, "Frequency", cfg->omega / TWO_PI);
    }

    if (cfg->nangle > 0 && cfg->angleinvcdf) {
        cJSON_AddItemToObject(sub, "AngleInverseCDF", cJSON_CreateFloatArray(cfg->angleinvcdf, cfg->nangle));
    }

    if (cfg->srcid) {
        cJSON_AddNumberToObject(sub, "ID", cfg->srcid);
    }

    if (cfg->srcpattern) {
        uint dims[3] = {cfg->srcnum, (uint)cfg->srcparam1.w, (uint)cfg->srcparam2.w}, ndim = 2 + (cfg->srcnum > 1);
        uint* pdims = dims + (cfg->srcnum == 1);

        if (cfg->srctype == MCX_SRC_PATTERN3D) { /* a volumetric pattern of srcparam1.{x,y,z} voxels */
            dims[0] = (uint)cfg->srcparam1.x;
            dims[1] = (uint)cfg->srcparam1.y;
            dims[2] = (uint)cfg->srcparam1.z;
            ndim = 3;
            pdims = dims;
        }

        uint nx = pdims[0], ny = (ndim > 1) ? pdims[1] : 1, nz = (ndim > 2) ? pdims[2] : 1;
        float* rowpat = (float*)malloc(sizeof(float) * nx * ny * nz);

        /* the JData array is row-major, the reader converts it back with mcx_convertrow2col */
        for (uint x = 0; x < nx; x++)
            for (uint y = 0; y < ny; y++)
                for (uint z = 0; z < nz; z++) {
                    rowpat[(x * ny + y) * nz + z] = cfg->srcpattern[(z * ny + y) * nx + x];
                }

        cJSON_AddItemToObject(sub, "Pattern", tmp = cJSON_CreateObject());

        int ret = mcx_jdataencode(rowpat, ndim, pdims, "single", sizeof(float), cfg->zipid, tmp, 0, 0, cfg);
        free(rowpat);

        if (ret) {
            MCX_ERROR(ret, "data compression or base64 encoding failed");
        }
    }

    cJSON_AddItemToObject(obj, "Detector", sub = cJSON_CreateArray());

    for (int i = 0; i < cfg->detnum; i++) {
        cJSON_AddItemToArray(sub, tmp = cJSON_CreateObject());
        cJSON_AddItemToObject(tmp, "Pos", cJSON_CreateFloatArray(&(cfg->detpos[i].x), 3));
        cJSON_AddNumberToObject(tmp, "R", cfg->detpos[i].w);
    }

    /* save the shape constructs when they define the whole domain (with a Grid object, such as the
       built-in benchmarks); a -P/--shapes overlay applied over the volume or shapes of the input file
       does not, and saving it alone loses the base domain, so the rasterized volume is saved instead */
    if (cfg->shapedata && (cfg->vol == NULL || strstr(cfg->shapedata, "\"Grid\"") != NULL)) {
        cJSON* shape = mcx_parsejson(cfg->shapedata), *sp;

        if (shape == NULL) {
            MCX_ERROR(-1, "the input shape construct is not a valid JSON object");
        }

        sp = FIND_JSON_OBJ("Shapes", "Shapes", shape);

        if (sp == NULL) {
            sp = shape;
        }

        cJSON_AddItemToObject(root, "Shapes", sp);
    } else if (cfg->mediabyte == MEDIA_MUA_FLOAT || cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_ASGN_F2H || cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
        /* continuous and split-voxel media: undo the conversions in mcx_loadvolume/mcx_preprocess
           and save the file representation (floats in 1/mm, or the 8-byte svmc records as
           uint64), row-major with the per-voxel components in the last dimension */
        uint dimxy = cfg->dim.x * cfg->dim.y, dimyz = cfg->dim.y * cfg->dim.z;
        uint ncomp = (cfg->mediabyte == MEDIA_MUA_FLOAT || cfg->mediabyte == MEDIA_2LABEL_SPLIT) ? 1 : ((cfg->mediabyte == MEDIA_AS_F2H) ? 2 : 4);
        uint datalen = dimxy * cfg->dim.z, dims4[4] = {cfg->dim.x, cfg->dim.y, cfg->dim.z, ncomp};
        int elembytes = (cfg->mediabyte == MEDIA_2LABEL_SPLIT) ? 8 : sizeof(float);
        uchar* buf = (uchar*)calloc((size_t)datalen * ncomp, elembytes);
        float* fbuf = (float*)buf;
        int ret = 0;

        for (uint z = 0; z < cfg->dim.z; z++)
            for (uint y = 0; y < cfg->dim.y; y++)
                for (uint x = 0; x < cfg->dim.x; x++) {
                    size_t row = (size_t)x * dimyz + y * cfg->dim.z + z;
                    size_t col = cfg->isrowmajor ? row : (size_t)z * dimxy + y * cfg->dim.x + x;  /* before mcx_preprocess, a JData volume is still row-major */
                    unsigned int v = cfg->vol[col];

                    if (cfg->mediabyte == MEDIA_2LABEL_SPLIT && cfg->isdumpjson == 2) {
                        /* --dumpjson 2 dumps before mcx_preprocess, the records are still in file order */
                        memcpy(buf + row * 8, (uchar*)cfg->vol + col * 8, 8);
                    } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
                        /* mcx_preprocess split each 8-byte svmc record into two 32-bit planes
                           with the byte order reversed; rebuild the record in file order */
                        union {
                            unsigned char c[8];
                            unsigned int  i[2];
                        } b2u;
                        b2u.i[1] = v;
                        b2u.i[0] = cfg->vol[col + datalen];

                        for (int k = 0; k < 8; k++) {
                            buf[row * 8 + k] = b2u.c[7 - k];
                        }
                    } else if (cfg->mediabyte == MEDIA_MUA_FLOAT) {
                        union {
                            unsigned int i;
                            float f;
                        } u;
                        u.i = v;
                        fbuf[row] = (v == 0) ? NAN : u.f / cfg->unitinmm;
                    } else {
                        float* f = fbuf + row * ncomp;
                        f[0] = (v == 0) ? NAN : mcx_half2float(v & 0xFFFF) / cfg->unitinmm;
                        f[1] = (v == 0) ? NAN : mcx_half2float(v >> 16) / cfg->unitinmm;

                        if (cfg->mediabyte == MEDIA_ASGN_F2H) {
                            unsigned int gn = cfg->vol[col + datalen];
                            f[2] = mcx_half2float(gn & 0xFFFF);
                            f[3] = mcx_half2float(gn >> 16);
                        }
                    }
                }

        obj = cJSON_CreateObject();
        ret = mcx_jdataencode(buf, (ncomp > 1 ? 4 : 3), dims4, (cfg->mediabyte == MEDIA_2LABEL_SPLIT ? "uint64" : "single"), elembytes, cfg->zipid, obj, 0, 0, cfg);
        free(buf);

        if (ret) {
            MCX_ERROR(ret, "data compression or base64 encoding failed");
        } else {
            cJSON_AddItemToObject(root, "Shapes", obj);
        }
    } else { /* if shape info is not available, save volume to JData constructs */
        uint datalen = cfg->dim.x * cfg->dim.y * cfg->dim.z;
        size_t outputbytes = datalen * sizeof(int);
        uchar* buf = (uchar*)calloc(datalen, sizeof(int));
        uint*  vol;
        int ret = 0;

        /*converting volume to the minimal size*/
        memcpy(buf, cfg->vol, datalen * sizeof(int));

        if (!cfg->isrowmajor) { /* before mcx_preprocess (--dumpjson 2), a JData volume is still row-major */
            mcx_convertcol2row((uint**)(&buf), &cfg->dim);
        }

        vol = (uint*)buf;

        if (cfg->mediabyte == 1) {
            outputbytes = datalen;

            for (int i = 0; i < datalen; i++) {
                buf[i] = vol[i] & 0xFF;
            }
        } else if (cfg->mediabyte == 2) {
            unsigned short* p = (unsigned short*)buf;
            outputbytes = datalen * sizeof(short);

            for (int i = 0; i < datalen; i++) {
                p[i] = vol[i] & 0xFFFF;
            }
        } else if (cfg->mediabyte <= 4) { /* packed 32-bit formats, such as labelplus, are saved as is */
            for (int i = 0; i < datalen; i++) {
                vol[i] = vol[i] & MED_MASK;
            }
        }

        obj = cJSON_CreateObject();
        ret = mcx_jdataencode(buf, 3, &cfg->dim.x, (cfg->mediabyte == 1 ? "uint8" : (cfg->mediabyte == 2 ? "uint16" : "uint32")),
                              outputbytes / datalen, cfg->zipid, obj, 0, 0, cfg);

        if (buf) {
            free(buf);
        }

        if (ret) {
            MCX_ERROR(ret, "data compression or base64 encoding failed");
        } else {
            cJSON_AddItemToObject(root, "Shapes", obj);
        }
    }

    /* now save JSON to file */
    jsonstr = cJSON_Print(root);

    if (jsonstr == NULL) {
        MCX_ERROR(-1, "error when converting to JSON");
    }

    if (!strcmp(filename, "-")) {
        fprintf(cfg->flog, "%s\n", jsonstr);
    } else {
        FILE* fp;

        if (cfg->rootpath[0]) {
            char name[MAX_FULL_PATH];
            sprintf(name, "%s%c%s", cfg->rootpath, pathsep, filename);
            fp = fopen(name, "wt");
        } else {
            fp = fopen(filename, "wt");
        }

        if (fp == NULL) {
            MCX_ERROR(-1, "error opening file to write");
        }

        fprintf(fp, "%s\n", jsonstr);
        fclose(fp);
    }

    if (jsonstr) {
        free(jsonstr);
    }

    if (root) {
        cJSON_Delete(root);
    }
}


/**
 * Bytes in the array the last mcx_jdatadecode call produced, so that a caller
 * about to read from it can check it holds what the caller expects: a volume
 * that Dim and the media format say is larger than the array is otherwise read
 * past its end.
 */
static size_t mcx_jdata_bytes = 0;

/**
 * @brief Load media index data volume (.bin or .vol) to the memory
 *
 * @param[in] filename: file name to the binary volume data (support 1,2 and 4 bytes per voxel)
 * @param[in] cfg: simulation configuration
 */

void mcx_loadvolume(char* filename, Config* cfg, int isbuf) {
    unsigned int i, datalen, res;
    unsigned char* inputvol = NULL;
    FILE* fp;

    if (!isbuf) {
        if (strstr(filename, ".json") != NULL) {
            int status;
            Grid3D grid = {&(cfg->vol), &(cfg->dim), {1.f, 1.f, 1.f}, cfg->isrowmajor};

            if (cfg->issrcfrom0) {
                memset(&(grid.orig.x), 0, sizeof(float3));
            }

            status = mcx_load_jsonshapes(&grid, filename);

            if (status) {
                MCX_ERROR(status, mcx_last_shapeerror());
            }

            return;
        }

        fp = fopen(filename, "rb");

        if (fp == NULL) {
            MCX_ERROR(-5, "the specified binary volume file does not exist");
        }
    }

    if (cfg->vol) {
        free(cfg->vol);
        cfg->vol = NULL;
    }

    /* the product is 32-bit, as is every index into the volume after it; a
       Dim it does not fit in wraps around to a small allocation that the
       loops below then overrun */
    if (cfg->dim.x == 0 || cfg->dim.y == 0 || cfg->dim.z == 0 ||
            (uint64_t)cfg->dim.x * cfg->dim.y * cfg->dim.z > 0x7FFFFFFFu) {
        MCX_ERROR(-1, "Domain.Dim must be three positive numbers whose product is below 2^31");
    }

    datalen = cfg->dim.x * cfg->dim.y * cfg->dim.z;

    /* a volume handed over in memory -- a JData array from Shapes -- must hold
       as many bytes as the file read below would: Dim voxels of the media
       format's size */
    if (isbuf && (uint64_t)datalen * ((cfg->mediabyte == MEDIA_ASGN_F2H) ? 16 : ((cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_2LABEL_SPLIT) ? 8 : MIN(cfg->mediabyte, 4))) > mcx_jdata_bytes) {
        MCX_ERROR(-6, "the volume array in Shapes is smaller than Domain.Dim and Domain.MediaFormat require");
    }

    cfg->vol = (unsigned int*)malloc(sizeof(unsigned int) * datalen * (1 + (cfg->mediabyte == MEDIA_2LABEL_SPLIT || cfg->mediabyte == MEDIA_ASGN_F2H)));

    if (!isbuf) {
        if (cfg->mediabyte == MEDIA_AS_F2H) {
            inputvol = (unsigned char*)malloc(sizeof(unsigned char) * (datalen << 3));
        } else if (cfg->mediabyte == MEDIA_ASGN_F2H) {
            inputvol = (unsigned char*)malloc(sizeof(unsigned char) * (datalen << 4));
        } else if (cfg->mediabyte >= 4) {
            inputvol = (unsigned char*)(cfg->vol);
        } else {
            inputvol = (unsigned char*)malloc(sizeof(unsigned char) * cfg->mediabyte * datalen);
        }

        res = fread(inputvol, sizeof(unsigned char) * ((cfg->mediabyte == MEDIA_ASGN_F2H) ? 16 : ((cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_2LABEL_SPLIT) ? 8 : MIN(cfg->mediabyte, 4))), datalen, fp);
        fclose(fp);

        if (res != datalen) {
            MCX_ERROR(-6, "file size does not match specified dimensions");
        }
    } else {
        inputvol = (unsigned char*)filename;
    }

    if (cfg->mediabyte == 1) { /*convert all format into 4-byte int index*/
        unsigned char* val = inputvol;

        for (i = 0; i < datalen; i++) {
            cfg->vol[i] = val[i];
        }
    } else if (cfg->mediabyte == 2) {
        unsigned short* val = (unsigned short*)inputvol;

        for (i = 0; i < datalen; i++) {
            cfg->vol[i] = val[i];
        }
    } else if (cfg->mediabyte == 4) {
        memcpy(cfg->vol, inputvol, (datalen << 2));
    } else if (cfg->mediabyte == MEDIA_MUA_FLOAT) {
        union {
            float f;
            uint  i;
        } f2i;
        float* val = (float*)inputvol;

        for (i = 0; i < datalen; i++) {
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
        float* val = (float*)inputvol;
        float f2h[2];
        int offset = (cfg->mediabyte == MEDIA_ASGN_F2H);

        for (i = 0; i < datalen; i++) {
            f2h[0] = val[i << (1 + offset)] * cfg->unitinmm;       // mua
            f2h[1] = val[(i << (1 + offset)) + 1] * cfg->unitinmm; // mus
            cfg->vol[i] = mcx_float2half2(f2h);

            if (f2h[0] != f2h[0] || f2h[1] != f2h[1]) { /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
                cfg->vol[i] = 0;

                if (cfg->mediabyte == MEDIA_AS_F2H) {
                    continue;
                }
            }

            if (cfg->mediabyte == MEDIA_ASGN_F2H) {
                f2h[0] = val[(i << 2) + 2];   // g
                f2h[1] = val[(i << 2) + 3];   // n
                cfg->vol[i + datalen] = mcx_float2half2(f2h);
            }
        }
    } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
        memcpy(cfg->vol, inputvol, (datalen << 3));
    } else if (isbuf) { /* other 4-byte packed formats, such as labelplus; a file is read into cfg->vol directly */
        memcpy(cfg->vol, inputvol, (datalen << 2));
    }

    int medianum = MAX(cfg->medianum, cfg->polmedianum + 1);

    if (cfg->mediabyte <= 4)
        for (i = 0; i < datalen; i++) {
            if (cfg->vol[i] >= medianum) {
                MCX_ERROR(-6, "medium index exceeds the specified medium types");
            }
        }

    if (!isbuf && (cfg->mediabyte < 4 || cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_ASGN_F2H)) {
        free(inputvol);
    }
}

#endif

/**
 * @brief Initialize the replay data structure from detected photon data - in embedded mode (MATLAB/Python)
 *
 * @param[in,out] cfg: simulation configuration
 * @param[in] detps: detected photon data
 * @param[in] dimdetps: the dimension vector of the detected photon data
 * @param[in] seedbyte: the number of bytes per RNG seed
 */

void mcx_replayinit(Config* cfg, float* detps, int dimdetps[2], int seedbyte) {
    int i, j, hasdetid = 0, offset;
    float plen;

    if (cfg->seed == SEED_FROM_FILE && detps == NULL) {
        MCX_ERROR(-6, "you give cfg.seed for replay, but did not specify cfg.detphotons.\nPlease define it as the detphoton output from the baseline simulation\n");
    }

    if (detps == NULL || cfg->seed != SEED_FROM_FILE) {
        return;
    }

    if (cfg->nphoton != dimdetps[1]) {
        MCX_ERROR(-6, "the column numbers of detphotons and seed do not match\n");
    }

    if (seedbyte == 0) {
        MCX_ERROR(-6, "the seed input is empty");
    }

    hasdetid = SAVE_DETID(cfg->savedetflag);
    offset = SAVE_NSCAT(cfg->savedetflag) * (cfg->medianum - 1);

    if (((!hasdetid) && cfg->detnum > 1) || !SAVE_PPATH(cfg->savedetflag)) {
        MCX_ERROR(-6, "please rerun the baseline simulation and save detector ID (D) and partial-path (P) using cfg.savedetflag='dp' ");
    }

    cfg->replay.weight = (float*) malloc(cfg->nphoton * sizeof(float));
    cfg->replay.tof = (float*) calloc(cfg->nphoton, sizeof(float));
    cfg->replay.detid = (int*) calloc(cfg->nphoton, sizeof(int));

    cfg->nphoton = 0;

    for (i = 0; i < dimdetps[1]; i++) {
        if (cfg->replaydet <= 0 || cfg->replaydet == ((int) (detps[i * dimdetps[0]]) & 0xFFFF)) {
            if (cfg->srcid <= 0 || (((int) (detps[i * dimdetps[0]]) & 0xFFFF0000) && cfg->srcid == (((int) (detps[i * dimdetps[0]]) & 0xFFFF0000) >> 16))) {
                if (i != cfg->nphoton) {
                    memcpy((char*) (cfg->replay.seed) + cfg->nphoton * seedbyte,
                           (char*) (cfg->replay.seed) + i * seedbyte,
                           seedbyte);
                }

                cfg->replay.weight[cfg->nphoton] = 1.f;
                cfg->replay.tof[cfg->nphoton] = 0.f;
                cfg->replay.detid[cfg->nphoton] = (hasdetid) ? (int) (detps[i * dimdetps[0]]) : 1;

                for (j = hasdetid; j < cfg->medianum - 1 + hasdetid; j++) {
                    plen = detps[i * dimdetps[0] + offset + j];
                    cfg->replay.weight[cfg->nphoton] *= expf(-cfg->prop[j - hasdetid + 1].mua * plen);
                    plen *= cfg->unitinmm;
                    cfg->replay.tof[cfg->nphoton] += plen * R_C0 * cfg->prop[j - hasdetid + 1].n;
                }

                if (cfg->replay.tof[cfg->nphoton] < cfg->tstart
                        || cfg->replay.tof[cfg->nphoton] > cfg->tend) { /*need to consider -g*/
                    continue;
                }

                cfg->nphoton++;
            }
        }
    }

    cfg->replay.weight = (float*) realloc(cfg->replay.weight, cfg->nphoton * sizeof(float));
    cfg->replay.tof = (float*) realloc(cfg->replay.tof, cfg->nphoton * sizeof(float));
    cfg->replay.detid = (int*) realloc(cfg->replay.detid, cfg->nphoton * sizeof(int));
}

/**
 * @brief Initialize the replay data structure from detected photon data - in standalone mode
 *
 * @param[in,out] cfg: simulation configuration
 * @param[in] detps: detected photon data
 * @param[in] dimdetps: the dimension vector of the detected photon data
 * @param[in] seedbyte: the number of bytes per RNG seed
 */

void mcx_replayprep(int* detid, float* ppath, History* his, Config* cfg) {
    int i, j;
    float plen;
    cfg->nphoton = 0;

    for (i = 0; i < his->savedphoton; i++) {
        if (cfg->replaydet <= 0 || (detid && cfg->replaydet == (detid[i] & 0xFFFF))) {
            if (i != cfg->nphoton) {
                memcpy((char*)(cfg->replay.seed) + cfg->nphoton * his->seedbyte, (char*)(cfg->replay.seed) + i * his->seedbyte, his->seedbyte);
            }

            cfg->replay.weight[cfg->nphoton] = 1.f;
            cfg->replay.tof[cfg->nphoton] = 0.f;
            cfg->replay.detid[cfg->nphoton] = (detid != NULL) ? detid[i] : 1;

            for (j = 0; j < his->maxmedia; j++) {
                plen = ppath[i * his->maxmedia + j] * his->unitinmm;
                cfg->replay.weight[cfg->nphoton] *= expf(-cfg->prop[j + 1].mua * plen);
                cfg->replay.tof[cfg->nphoton] += plen * R_C0 * cfg->prop[j + 1].n;
            }

            if (cfg->replay.tof[cfg->nphoton] < cfg->tstart || cfg->replay.tof[cfg->nphoton] > cfg->tend) { /*need to consider -g*/
                continue;
            }

            cfg->nphoton++;
        }
    }

    cfg->replay.seed = realloc(cfg->replay.seed, cfg->nphoton * his->seedbyte);
    cfg->replay.weight = (float*)realloc(cfg->replay.weight, cfg->nphoton * sizeof(float));
    cfg->replay.tof = (float*)realloc(cfg->replay.tof, cfg->nphoton * sizeof(float));
    cfg->replay.detid = (int*)realloc(cfg->replay.detid, cfg->nphoton * sizeof(int));
    cfg->minenergy = 0.f;
}

/**
 * @brief Validate all input fields, and warn incompatible inputs
 *
 * Perform self-checking and raise exceptions or warnings when input error is detected
 *
 * @param[in,out] cfg: simulation configuration
 * @param[in] detps: detected photon data
 * @param[in] dimdetps: the dimension vector of the detected photon data
 * @param[in] seedbyte: the number of bytes per RNG seed
 */

void mcx_validatecfg(Config* cfg, float* detps, int dimdetps[2], int seedbyte) {
    int i, gates;
    unsigned int partialdata =
        (cfg->medianum - 1) * (SAVE_NSCAT(cfg->savedetflag) + SAVE_PPATH(cfg->savedetflag) + SAVE_MOM(cfg->savedetflag));
    unsigned int hostdetreclen =
        partialdata + SAVE_DETID(cfg->savedetflag) + 3 * (SAVE_PEXIT(cfg->savedetflag) + SAVE_VEXIT(cfg->savedetflag))
        + SAVE_W0(cfg->savedetflag);
    hostdetreclen += cfg->polmedianum ? (4 * SAVE_IQUV(cfg->savedetflag)) : 0; // for polarized photon simulation

    if (!(cfg->mediabyte == 1 || cfg->mediabyte == 2 || cfg->mediabyte == 4
            || (cfg->mediabyte >= MEDIA_ASGN_F2H && cfg->mediabyte <= MEDIA_AS_SHORT))) {
        MCX_ERROR(-1, "Unsupported media format");
    }

    /* the source count sizes the normalization and output arrays; nought or a
       negative count (an int read into unsigned arithmetic) allocated nothing,
       or a wrapped-around size, and both were written past */
    if (cfg->srcnum < 1 || cfg->srcnum > 1000000) {
        MCX_ERROR(-1, "Optode.Source.SrcNum (the number of patterns) must be between 1 and 1000000");
    }

    if (!cfg->issave2pt && cfg->issaveref) {
        cfg->issaveref = 0;
    }

    if (!cfg->issrcfrom0) {
        cfg->srcpos.x--;
        cfg->srcpos.y--;
        cfg->srcpos.z--; /*convert to C index, grid center*/

        if (cfg->extrasrclen) {
            for (int i = 0; i < cfg->extrasrclen; i++) {
                cfg->srcdata[i].srcpos.x--;
                cfg->srcdata[i].srcpos.y--;
                cfg->srcdata[i].srcpos.z--;
            }
        }
    }

    if (cfg->tstep == 0.f) {
        cfg->tstep = cfg->tend;
    }

    if (cfg->tstart >= cfg->tend || cfg->tstep == 0.f) {
        MCX_ERROR(-6, "incorrect time gate settings");
    }

    if (ABS(cfg->srcdir.x * cfg->srcdir.x + cfg->srcdir.y * cfg->srcdir.y + cfg->srcdir.z * cfg->srcdir.z - 1.f) > 1e-5) {
        MCX_ERROR(-6, "field 'srcdir' must be a unitary vector");
    }

    if (cfg->steps.x == 0.f || cfg->steps.y == 0.f || cfg->steps.z == 0.f) {
        MCX_ERROR(-6, "field 'steps' can not have zero elements");
    }

    gates = (int) ((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);

    if (cfg->maxgate > gates) {
        cfg->maxgate = gates;
    }

    if (cfg->sradius > 0.f) {
        cfg->crop0.x = MAX((int) (cfg->srcpos.x - cfg->sradius), 0);
        cfg->crop0.y = MAX((int) (cfg->srcpos.y - cfg->sradius), 0);
        cfg->crop0.z = MAX((int) (cfg->srcpos.z - cfg->sradius), 0);
        cfg->crop1.x = MIN((int) (cfg->srcpos.x + cfg->sradius), cfg->dim.x - 1);
        cfg->crop1.y = MIN((int) (cfg->srcpos.y + cfg->sradius), cfg->dim.y - 1);
        cfg->crop1.z = MIN((int) (cfg->srcpos.z + cfg->sradius), cfg->dim.z - 1);
    } else if (cfg->sradius == 0.f) {
        memset(&(cfg->crop0), 0, sizeof(uint3));
        memset(&(cfg->crop1), 0, sizeof(uint3));
    } else {
        /*
            if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
        */
        if (!cfg->issrcfrom0) {
            cfg->crop0.x--;
            cfg->crop0.y--;
            cfg->crop0.z--;  /*convert to C index*/
            cfg->crop1.x--;
            cfg->crop1.y--;
            cfg->crop1.z--;
        }
    }

    if (cfg->seed < 0 && cfg->seed != SEED_FROM_FILE) {
        cfg->seed = time(NULL);
    }

    for (i = 0; i < cfg->detnum; i++) {
        if (!cfg->issrcfrom0) {
            cfg->detpos[i].x--;
            cfg->detpos[i].y--;
            cfg->detpos[i].z--;  /*convert to C index*/
        }
    }

    if (cfg->shapedata && strstr(cfg->shapedata, ":") != NULL) {
        if (cfg->mediabyte > 4) {
            MCX_ERROR(-6, "rasterization of shapes must be used with label-based mediatype");
        }

        Grid3D grid = {&(cfg->vol), &(cfg->dim), {1.f, 1.f, 1.f}, 0};

        if (cfg->issrcfrom0) {
            memset(&(grid.orig.x), 0, sizeof(float3));
        }

        int status = mcx_parse_shapestring(&grid, cfg->shapedata);

        if (status) {
            MCX_ERROR(-6, mcx_last_shapeerror());
        }
    }

    mcx_preprocess(cfg);

    cfg->his.maxmedia = cfg->medianum - 1; /*skip medium 0*/
    cfg->his.detnum = cfg->detnum;
    cfg->his.srcnum = cfg->srcnum;
    cfg->his.colcount = hostdetreclen; /*column count=maxmedia+2*/
    cfg->his.savedetflag = cfg->savedetflag;
    mcx_replayinit(cfg, detps, dimdetps, seedbyte);
}

#ifndef MCX_CONTAINER

/**
 * @brief Load previously saved photon seeds from a .jdt (or legacy .jdat) file for replay
 *
 * @param[in] filename: the name/path of the .jdt file produced from the baseline simulation
 * @param[in,out] cfg: simulation configuration
 */

void mcx_loadseedjdat(char* filename, Config* cfg) {
    char* jbuf;
    int len;

    FILE* fp = fopen(filename, "rb");

    if (fp == NULL) {
        MCX_ERROR(-6, "fail to open the specified seed jdata file");
    }

    fseek (fp, 0, SEEK_END);
    len = ftell(fp) + 1;
    jbuf = (char*)malloc(len);
    rewind(fp);

    if (fread(jbuf, len - 1, 1, fp) != 1) {
        MCX_ERROR(-2, "reading input file is terminated");
    }

    jbuf[len - 1] = '\0';
    fclose(fp);

    cJSON* root = mcx_parsejson(jbuf);
    free(jbuf);

    if (root) {
        cJSON* photondata = NULL, *detid = NULL, *info = NULL, *ppathdata = NULL, *seed = NULL, *tmp = NULL;
        cJSON* obj = cJSON_GetObjectItem(root, "MCXData");

        if (obj) {
            photondata = cJSON_GetObjectItem(obj, "PhotonData");
            info = cJSON_GetObjectItem(obj, "Info");

            if (photondata) {
                ppathdata = cJSON_GetObjectItem(photondata, "ppath");
                seed = cJSON_GetObjectItem(photondata, "seed");
                detid = cJSON_GetObjectItem(photondata, "detid");
            }
        }

        if (!seed || !ppathdata || !detid || !info) {
            MCX_ERROR(-1, "invalid jdat file, expect MCXData.PhotonData.seed, .detid and .ppath");
        }

        if (cJSON_IsObject(seed) && cJSON_IsObject(detid) && cJSON_IsObject(ppathdata) && cJSON_GetObjectItem(seed, "_ArraySize_")) {
            int ndim;
            uint dims[3] = {1, 1, 1};
            float* ppath = NULL;
            char* type;
            History his = {{0}};

            cJSON* vsize = cJSON_GetObjectItem(seed, "_ArraySize_");

            his.savedphoton = 0;
            his.seedbyte = 0;

            if (vsize) {
                cJSON* tmp = vsize->child;
                his.savedphoton = tmp->valueint;
                tmp = tmp->next;
                his.seedbyte = tmp->valueint;
            }

            if (info) {
                his.unitinmm = FIND_JSON_KEY("LengthUnit", "LengthUnit", info, 1.f, valuedouble);
            } else {
                his.unitinmm = 1.f;
            }

            cfg->nphoton = his.savedphoton;
            cfg->seed = SEED_FROM_FILE;

            mcx_jdatadecode((void**)&ppath, &ndim, dims, 2, &type, ppathdata, cfg);
            his.maxmedia = dims[1];

            mcx_jdatadecode((void**)&cfg->replay.seed, &ndim, dims, 2, &type, seed, cfg);
            mcx_jdatadecode((void**)&cfg->replay.detid, &ndim, dims, 2, &type, detid, cfg);

            cfg->replay.weight = (float*)malloc(his.savedphoton * sizeof(float));
            cfg->replay.tof = (float*)calloc(his.savedphoton, sizeof(float));
            mcx_replayprep(cfg->replay.detid, ppath, &his, cfg);

            if (ppath) {
                free(ppath);
            }
        }

        cJSON_Delete(root);
    } else {
        MCX_ERROR(-1, "invalid jdat file");
    }
}


/**
 * @brief Load previously saved photon seeds from an .mch file for replay
 *
 * @param[in,out] cfg: simulation configuration
 */

void mcx_loadseedfile(Config* cfg) {
    History his;
    FILE* fp = fopen(cfg->seedfile, "rb");

    if (fp == NULL) {
        MCX_ERROR(-7, "can not open the specified history file");
    }

    if (fread(&his, sizeof(History), 1, fp) != 1) {
        MCX_ERROR(-7, "error when reading the history file");
    }

    if (his.savedphoton == 0 || his.seedbyte == 0) {
        MCX_ERROR(-7, "history file does not contain seed data, please re-run your simulation with '-q 1'");
    }

    if (his.maxmedia != cfg->medianum - 1) {
        MCX_ERROR(-7, "the history file was generated with a different media setting");
    }

    if (fseek(fp, his.savedphoton * his.colcount * sizeof(float), SEEK_CUR)) {
        MCX_ERROR(-7, "illegal history file");
    }

    cfg->replay.seed = malloc(his.savedphoton * his.seedbyte);

    if (cfg->replay.seed == NULL) {
        MCX_ERROR(-7, "can not allocate memory");
    }

    if (fread(cfg->replay.seed, his.seedbyte, his.savedphoton, fp) != his.savedphoton) {
        MCX_ERROR(-7, "error when reading the seed data");
    }

    cfg->seed = SEED_FROM_FILE;
    cfg->nphoton = his.savedphoton;

    if (cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS  || cfg->outputtype == otRF || cfg->outputtype == otRFmus || cfg->outputtype == otWLTOF || cfg->outputtype == otWPTOF) { //cfg->replaydet>0
        int i, j, hasdetid = 0, offset;
        float plen, *ppath;
        hasdetid = SAVE_DETID(his.savedetflag);
        offset = SAVE_NSCAT(his.savedetflag) * his.maxmedia;

        if (((!hasdetid) && cfg->detnum > 1) || !SAVE_PPATH(his.savedetflag)) {
            MCX_ERROR(-7, "please rerun the baseline simulation and save detector ID (D) and partial-path (P) using '-w DP'");
        }

        ppath = (float*)malloc(his.savedphoton * his.colcount * sizeof(float));
        cfg->replay.weight = (float*)malloc(his.savedphoton * sizeof(float));
        cfg->replay.tof = (float*)calloc(his.savedphoton, sizeof(float));
        cfg->replay.detid = (int*)calloc(his.savedphoton, sizeof(int));
        fseek(fp, sizeof(his), SEEK_SET);

        if (fread(ppath, his.colcount * sizeof(float), his.savedphoton, fp) != his.savedphoton) {
            MCX_ERROR(-7, "error when reading the seed data");
        }

        cfg->nphoton = 0;

        for (i = 0; i < his.savedphoton; i++)
            if (cfg->replaydet <= 0 || cfg->replaydet == (int)(ppath[i * his.colcount])) {
                if (i != cfg->nphoton) {
                    memcpy((char*)(cfg->replay.seed) + cfg->nphoton * his.seedbyte, (char*)(cfg->replay.seed) + i * his.seedbyte, his.seedbyte);
                }

                cfg->replay.weight[cfg->nphoton] = 1.f;
                cfg->replay.tof[cfg->nphoton] = 0.f;
                cfg->replay.detid[cfg->nphoton] = (hasdetid) ? (int)(ppath[i * his.colcount]) : 1;

                for (j = hasdetid; j < his.maxmedia + hasdetid; j++) {
                    plen = ppath[i * his.colcount + offset + j] * his.unitinmm;
                    cfg->replay.weight[cfg->nphoton] *= expf(-cfg->prop[j - hasdetid + 1].mua * plen);
                    cfg->replay.tof[cfg->nphoton] += plen * R_C0 * cfg->prop[j - hasdetid + 1].n;
                }

                if (cfg->replay.tof[cfg->nphoton] < cfg->tstart || cfg->replay.tof[cfg->nphoton] > cfg->tend) { /*need to consider -g*/
                    continue;
                }

                cfg->nphoton++;
            }

        free(ppath);
        cfg->replay.seed = realloc(cfg->replay.seed, cfg->nphoton * his.seedbyte);
        cfg->replay.weight = (float*)realloc(cfg->replay.weight, cfg->nphoton * sizeof(float));
        cfg->replay.tof = (float*)realloc(cfg->replay.tof, cfg->nphoton * sizeof(float));
        cfg->replay.detid = (int*)realloc(cfg->replay.detid, cfg->nphoton * sizeof(int));
        cfg->minenergy = 0.f;
    }

    fclose(fp);
}

#endif

/**
 * @brief Convert a row-major (C/C++) array to a column-major (MATLAB/FORTRAN) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertrow2col(unsigned int* vol, uint3* dim) {
    uint x, y, z;
    unsigned int dimxy, dimyz;
    unsigned int* newvol = NULL;

    if (vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0) {
        return;
    }

    newvol = (unsigned int*)malloc(sizeof(unsigned int) * dim->x * dim->y * dim->z);
    dimxy = dim->x * dim->y;
    dimyz = dim->y * dim->z;

    for (x = 0; x < dim->x; x++)
        for (y = 0; y < dim->y; y++)
            for (z = 0; z < dim->z; z++) {
                newvol[z * dimxy + y * dim->x + x] = vol[x * dimyz + y * dim->z + z];
            }

    memcpy(vol, newvol, sizeof(unsigned int) * dim->x * dim->y * dim->z);
    free(newvol);
}

/**
 * @brief Convert a row-major (C/C++) array to a column-major (MATLAB/FORTRAN) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertrow2col64(size_t* vol, uint3* dim) {
    uint x, y, z;
    size_t dimxy, dimyz;
    size_t* newvol = NULL;

    if (vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0) {
        return;
    }

    newvol = (size_t*)malloc(sizeof(size_t) * dim->x * dim->y * dim->z);
    dimxy = dim->x * dim->y;
    dimyz = dim->y * dim->z;

    for (x = 0; x < dim->x; x++)
        for (y = 0; y < dim->y; y++)
            for (z = 0; z < dim->z; z++) {
                newvol[z * dimxy + y * dim->x + x] = (vol)[x * dimyz + y * dim->z + z];
            }

    memcpy(vol, newvol, sizeof(size_t) * dim->x * dim->y * dim->z);
    free(newvol);
}

/**
 * @brief Convert a column-major (MATLAB/FORTRAN) array to a row-major (C/C++) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertcol2row(unsigned int** vol, uint3* dim) {
    uint x, y, z;
    unsigned int dimxy, dimyz;
    unsigned int* newvol = NULL;

    if (*vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0) {
        return;
    }

    newvol = (unsigned int*)malloc(sizeof(unsigned int) * dim->x * dim->y * dim->z);
    dimxy = dim->x * dim->y;
    dimyz = dim->y * dim->z;

    for (z = 0; z < dim->z; z++)
        for (y = 0; y < dim->y; y++)
            for (x = 0; x < dim->x; x++) {
                newvol[x * dimyz + y * dim->z + z] = (*vol)[z * dimxy + y * dim->x + x];
            }

    memcpy(*vol, newvol, sizeof(unsigned int) * dim->x * dim->y * dim->z);
    free(newvol);
}

/**
 * @brief Convert a column-major (MATLAB/FORTRAN) array to a row-major (C/C++) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertcol2row4d(unsigned int** vol, uint4* dim) {
    uint x, y, z, w;
    unsigned int dimxyz, dimyzw, dimxy, dimzw;
    unsigned int* newvol = NULL;

    if (*vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0 || dim->w == 0) {
        return;
    }

    newvol = (unsigned int*)malloc(sizeof(unsigned int) * dim->x * dim->y * dim->z * dim->w);
    dimxyz = dim->x * dim->y * dim->z;
    dimyzw = dim->y * dim->z * dim->w;
    dimxy = dim->x * dim->y;
    dimzw = dim->z * dim->w;

    for (w = 0; w < dim->w; w++)
        for (z = 0; z < dim->z; z++)
            for (y = 0; y < dim->y; y++)
                for (x = 0; x < dim->x; x++) {
                    newvol[x * dimyzw + y * dimzw + z * dim->w + w] = (*vol)[w * dimxyz + z * dimxy + y * dim->x + x];
                }

    free(*vol);
    *vol = newvol;
}

/**
 * @brief Check if a voxel contains background medium(0)
 *
 * This function check if a voxel contains background medium,
 * under SVMC mode
 *
 * @param[in] vol: high 32 bit of the input volume under SVMC mdoe
 */

int mcx_svmc_bgvoxel(int vol) {
    unsigned int lower;
    lower = (unsigned int)(vol & LOWER_MASK) >> 24;
    return !lower;
}

/**
 * @brief Pre-label the voxel near a detector for easy photon detection
 *
 * This function preprocess the volume and detector data and add the detector ID to the
 * upper 16bits of the voxel that the detector encompasses. If two detectors cover the same
 * voxel, the later one will overwrite the ID of the 1st one. In MCX kernel, the detector
 * coverage is tested for all detectors despite the label written (only used as a binary mask)
 *
 * @param[in] cfg: simulation configuration
 */

void  mcx_maskdet(Config* cfg) {
    uint d, dx, dy, dz, idx1d, zi, yi, c, count, isonecube;
    float x, y, z, ix, iy, iz, rx, ry, rz, d2, mind2, d2max, radius;
    unsigned int* padvol;
    const float corners[8][3] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
        {1.f, 1.f, 0.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f}, {1.f, 1.f, 1.f}
    };

    dx = cfg->dim.x + 2;
    dy = cfg->dim.y + 2;
    dz = cfg->dim.z + 2;

    isonecube = (cfg->dim.x == 1) && (cfg->dim.y == 1) && (cfg->dim.z == 1);

    /*handling boundaries in a volume search is tedious, I first pad vol by a layer of zeros,
      then I don't need to worry about boundaries any more*/

    padvol = (unsigned int*)calloc(dx * dy * sizeof(unsigned int), dz);

    for (zi = 1; zi <= cfg->dim.z; zi++)
        for (yi = 1; yi <= cfg->dim.y; yi++) {
            memcpy(padvol + zi * dy * dx + yi * dx + 1, cfg->vol + (zi - 1)*cfg->dim.y * cfg->dim.x + (yi - 1)*cfg->dim.x, cfg->dim.x * sizeof(int));
        }

    /**
     * The goal here is to find a set of voxels for each
     * detector so that the intersection between a sphere
     * of R=cfg->detradius,c0=cfg->detpos[d] and the object
     * surface (or bounding box) is fully covered.
     */
    for (d = 0; d < cfg->detnum; d++) {                     /*loop over each detector*/
        float zlo, zhi, ylo, yhi, xlo, xhi, maxdim;

        count = 0;
        radius = ABS(cfg->detpos[d].w);
        d2max = (radius + 1.7321f) * (radius + 1.7321f);

        if (!isfinite(radius) || !isfinite(cfg->detpos[d].x) || !isfinite(cfg->detpos[d].y) || !isfinite(cfg->detpos[d].z)) {
            MCX_ERROR(-1, "a detector position or radius is not a finite number");
        }

        /* search in a cube with edge length 2*R+3, in steps of half a voxel.  A
           detector far larger than the volume only reaches the volume, and its
           cube is searched only there: stepped through the whole of it, a
           radius of 1e12 never ends -- adding 0.5 to a float that large leaves
           it unchanged.  Every other detector keeps exactly the original walk. */
        zlo = ylo = xlo = -radius - 1.f;
        zhi = yhi = xhi = radius + 1.f;
        maxdim = MAX(MAX(cfg->dim.x, cfg->dim.y), cfg->dim.z);

        if (radius > 4.f * maxdim + 4.f) {
            zlo = -cfg->detpos[d].z;
            zhi = zlo + cfg->dim.z;
            ylo = -cfg->detpos[d].y;
            yhi = ylo + cfg->dim.y;
            xlo = -cfg->detpos[d].x;
            xhi = xlo + cfg->dim.x;

            if (fabsf(cfg->detpos[d].x) + fabsf(cfg->detpos[d].y) + fabsf(cfg->detpos[d].z) > 1e6f) {
                MCX_ERROR(-1, "a detector is placed too far from the volume");
            }
        }

        for (z = zlo; z <= zhi; z += 0.5f) {
            iz = z + cfg->detpos[d].z;

            for (y = ylo; y <= yhi; y += 0.5f) {
                iy = y + cfg->detpos[d].y;

                for (x = xlo; x <= xhi; x += 0.5f) {
                    ix = x + cfg->detpos[d].x;

                    if (iz < 0 || ix < 0 || iy < 0 || ix >= cfg->dim.x || iy >= cfg->dim.y || iz >= cfg->dim.z ||
                            x * x + y * y + z * z > (radius + 1.f) * (radius + 1.f)) {
                        continue;
                    }

                    mind2 = VERY_BIG;

                    for (c = 0; c < 8; c++) { /*test each corner of a voxel*/
                        rx = (int)ix - cfg->detpos[d].x + corners[c][0];
                        ry = (int)iy - cfg->detpos[d].y + corners[c][1];
                        rz = (int)iz - cfg->detpos[d].z + corners[c][2];
                        d2 = rx * rx + ry * ry + rz * rz;

                        if (d2 > d2max) { /*R+sqrt(3) to make sure the circle is fully corvered*/
                            mind2 = VERY_BIG;
                            break;
                        }

                        if (d2 < mind2) {
                            mind2 = d2;
                        }
                    }

                    if (mind2 == VERY_BIG || mind2 >= (radius + 0.5f * (1.f + isonecube)) * (radius + 0.5f * (1.f + isonecube))) {
                        continue;
                    }

                    idx1d = ((int)(iz + 1.f) * dy * dx + (int)(iy + 1.f) * dx + (int)(ix + 1.f)); /*1.f comes from the padded layer*/

                    if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
                        unsigned int lower, upper;
                        lower = (unsigned int)(padvol[idx1d] & LOWER_MASK) >> 24;
                        upper = (unsigned int)(padvol[idx1d] & UPPER_MASK) >> 16;

                        if (lower || upper) { /*background voxel if both are 0, do nothing*/
                            if (!lower && upper) { /*a split voxel that contains background*/
                                cfg->vol[((int)iz * cfg->dim.y * cfg->dim.x + (int)iy * cfg->dim.x + (int)ix)] |= DET_MASK; /*set the highest bit to 1*/
                                count++;
                            } else if ((lower && !upper) || (lower && upper)) /*a split voxel that contains no background, check surrounding voxels*/
                                if (mcx_svmc_bgvoxel(padvol[idx1d + 1]) || mcx_svmc_bgvoxel(padvol[idx1d - 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dx]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d - dx]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d + dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dx - 1]) || mcx_svmc_bgvoxel(padvol[idx1d - dx + 1]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d - dx - 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx - 1]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d - dy * dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx - 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx + dx]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d + dy * dx - dx]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx + dx]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx - dx]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d + dy * dx + dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx + dx - 1]) || mcx_svmc_bgvoxel(padvol[idx1d + dy * dx - dx + 1]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d + dy * dx - dx - 1]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx + dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx + dx - 1]) ||
                                        mcx_svmc_bgvoxel(padvol[idx1d - dy * dx - dx + 1]) || mcx_svmc_bgvoxel(padvol[idx1d - dy * dx - dx - 1])) {
                                    cfg->vol[((int)iz * cfg->dim.y * cfg->dim.x + (int)iy * cfg->dim.x + (int)ix)] |= DET_MASK; /*set the highest bit to 1*/
                                    count++;
                                }
                        }
                    } else {
                        if (padvol[idx1d]) /*looking for a voxel on the interface or bounding box*/
                            if (!(padvol[idx1d + 1] && padvol[idx1d - 1] && padvol[idx1d + dx] && padvol[idx1d - dx] && padvol[idx1d + dy * dx] && padvol[idx1d - dy * dx] &&
                                    padvol[idx1d + dx + 1] && padvol[idx1d + dx - 1] && padvol[idx1d - dx + 1] && padvol[idx1d - dx - 1] &&
                                    padvol[idx1d + dy * dx + 1] && padvol[idx1d + dy * dx - 1] && padvol[idx1d - dy * dx + 1] && padvol[idx1d - dy * dx - 1] &&
                                    padvol[idx1d + dy * dx + dx] && padvol[idx1d + dy * dx - dx] && padvol[idx1d - dy * dx + dx] && padvol[idx1d - dy * dx - dx] &&
                                    padvol[idx1d + dy * dx + dx + 1] && padvol[idx1d + dy * dx + dx - 1] && padvol[idx1d + dy * dx - dx + 1] && padvol[idx1d + dy * dx - dx - 1] &&
                                    padvol[idx1d - dy * dx + dx + 1] && padvol[idx1d - dy * dx + dx - 1] && padvol[idx1d - dy * dx - dx + 1] && padvol[idx1d - dy * dx - dx - 1])) {
                                cfg->vol[((int)iz * cfg->dim.y * cfg->dim.x + (int)iy * cfg->dim.x + (int)ix)] |= DET_MASK; /*set the highest bit to 1*/
                                count++;
                            }
                    }
                }
            }
        }

        if (cfg->issavedet && count == 0) {
            MCX_FPRINTF(stderr, S_RED "WARNING: detector %d is not located on an interface, please check coordinates.\n" S_RESET, d + 1);
        }
    }

    free(padvol);

#ifndef MCX_CONTAINER

    if (cfg->isdumpmask) {
        mcx_dumpmask(cfg);
    }

#endif
}

/**
 * @brief Save the pre-masked volume (with detector ID) to an nii file
 *
 * To test the results, you should use -M to dump the det-mask, load
 * it in matlab, and plot the interface containing the detector with
 * pcolor() (has the matching index), and then draw a circle with the
 * radius and center set in the input file. the pixels should completely
 * cover the circle.
 *
 * @param[in] cfg: simulation configuration
 */

#ifndef MCX_CONTAINER

void mcx_dumpmask(Config* cfg) {
    char fname[MAX_FULL_PATH];

    if (cfg->rootpath[0]) {
        sprintf(fname, "%s%c%s_vol", cfg->rootpath, pathsep, cfg->session);
    } else {
        sprintf(fname, "%s_vol", cfg->session);
    }

    if (cfg->outputformat == ofJNifti || cfg->outputformat == ofBJNifti) {
        uint dims[] = {cfg->dim.x, cfg->dim.y, cfg->dim.z};
        size_t datalen = sizeof(uint) * cfg->dim.x * cfg->dim.y * cfg->dim.z;
        float voxelsize[] = {cfg->steps.x, cfg->steps.y, cfg->steps.z};
        uint* buf = malloc(datalen);
        memcpy(buf, cfg->vol, datalen);
        mcx_convertcol2row((uint**)(&buf), (uint3*)dims);

        if (cfg->outputformat == ofJNifti) {
            mcx_savejnii((float*)buf, 3, dims, voxelsize, fname, 0, 0, cfg);
        } else {
            mcx_savebnii((float*)buf, 3, dims, voxelsize, fname, 0, 0, cfg);
        }

        free(buf);
    } else {
        mcx_savenii((float*)cfg->vol, cfg->dim.x * cfg->dim.y * cfg->dim.z, fname, NIFTI_TYPE_UINT32, ofNifti, cfg);
    }

    if (cfg->isdumpmask == 1 && cfg->isdumpjson == 0) { /*if dumpmask>1, simulation will also run*/
        MCX_FPRINTF(cfg->flog, "volume mask is saved in %s\n", fname);
        mcx_clearcfg(cfg);
        exit(0);
    }
}

/**
 * Bytes per element of a JData _ArrayType_, or 0 for a name that is not one of
 * the numeric types.
 */
static int mcx_jdata_typesize(const char* type) {
    if (type == NULL) {
        return 0;
    }

    if (!strcmp(type, "uint8") || !strcmp(type, "int8") || !strcmp(type, "char") || !strcmp(type, "uchar") || !strcmp(type, "logical")) {
        return 1;
    }

    if (!strcmp(type, "uint16") || !strcmp(type, "int16")) {
        return 2;
    }

    if (!strcmp(type, "uint32") || !strcmp(type, "int32") || !strcmp(type, "single") || !strcmp(type, "float32")) {
        return 4;
    }

    if (!strcmp(type, "uint64") || !strcmp(type, "int64") || !strcmp(type, "double") || !strcmp(type, "float64")) {
        return 8;
    }

    return 0;
}

/**
 * @brief Decode an ND array from JSON/JData construct and output to a volumetric array
 *
 * The JData specification defines a portable way to encode and share volumetric
 * ND arrays and other complex data structures, such as trees, graphs and tables.
 * This function is capable of importing any ND numerical arrays in the JData
 * construct in to a generic array, permitting data decompression and base64 decoding.
 *
 * @param[in] vol: a pointer that points to the ND array buffer
 * @param[in] ndim: the number of dimensions
 * @param[in] dims: an integer pointer that points to the dimensional vector
 * @param[in] type: a string of JData data types, such as "uint8" "float32", "int32" etc
 * @param[in] byte: number of byte per voxel
 * @param[in] zipid: zip method: 0:zlib,1:gzip,2:base64,3:lzma,4:lzip,5:lz4,6:lz4hc
 * @param[in] obj: a pre-created cJSON object to store the output JData fields
 */

int  mcx_jdatadecode(void** vol, int* ndim, uint* dims, int maxdim, char** type, cJSON* obj, Config* cfg) {
    int ret = 0;
    size_t elemnum = 0;
    cJSON* ztype = NULL;
    cJSON* vsize = cJSON_GetObjectItem(obj, "_ArraySize_");
    cJSON* vtype = cJSON_GetObjectItem(obj, "_ArrayType_");
    cJSON* vdata = cJSON_GetObjectItem(obj, "_ArrayData_");

    if (!vdata) {
        ztype = cJSON_GetObjectItem(obj, "_ArrayZipType_");
        vdata = cJSON_GetObjectItem(obj, "_ArrayZipData_");
    }

    /* every annotation is checked before any of it is used: a field of the
       wrong JSON type has a NULL valuestring, and a size that does not match
       the data has every caller read past the end of the array */
    mcx_jdata_bytes = 0;

    if (!vtype || !cJSON_IsString(vtype) || mcx_jdata_typesize(vtype->valuestring) == 0) {
        MCX_ERROR(-1, "a JData array needs an _ArrayType_ naming a numeric type, such as \"uint8\" or \"single\"");
    }

    if (!vsize || !(cJSON_IsArray(vsize) || cJSON_IsNumber(vsize))) {
        MCX_ERROR(-1, "a JData array needs an _ArraySize_ of positive integers");
    }

    if (ztype && (!cJSON_IsString(ztype) || !cJSON_IsString(vdata))) {
        MCX_ERROR(-1, "_ArrayZipType_ and _ArrayZipData_ must both be strings");
    }

    if (vtype) {
        *type = vtype->valuestring;
    }

    /* only infer the media format from the array type if the input did not set it; the
       array type is returned regardless, as the callers check it (e.g. the source pattern) */
    if (!flagset['K'] && vtype) {
        if (strstr(*type, "int8")) {
            cfg->mediabyte = 1;
        } else if (strstr(*type, "int16")) {
            cfg->mediabyte = 2;
        } else if (strstr(*type, "int64") && cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
            cfg->mediabyte = MEDIA_2LABEL_SPLIT;
        } else if (strstr(*type, "double") || (strstr(*type, "int64") && cfg->mediabyte != MEDIA_2LABEL_SPLIT)) {
            MCX_ERROR(-1, "8-byte volume array is not supported");
        } else {
            cfg->mediabyte = 4;
        }
    }

    if (vdata) {

        {
            /* a 1-D size may be written as a plain number */
            cJSON* tmp = cJSON_IsArray(vsize) ? vsize->child : vsize;
            *ndim = cJSON_IsArray(vsize) ? cJSON_GetArraySize(vsize) : 1;
            elemnum = 1;

            if (*ndim < 1) {
                MCX_ERROR(-1, "a JData array needs an _ArraySize_ of positive integers");
            }

            for (int i = 0; i < *ndim; i++, tmp = tmp->next) {
                if (!cJSON_IsNumber(tmp) || tmp->valuedouble < 1 || tmp->valuedouble > 2147483647.0) {
                    MCX_ERROR(-1, "a JData array needs an _ArraySize_ of positive integers");
                }

                if (i < maxdim) {
                    dims[i] = (uint)tmp->valuedouble;
                }

                elemnum *= (size_t)tmp->valuedouble;

                if (elemnum > ((size_t)1 << 40)) {
                    MCX_ERROR(-1, "the JData array is too large");
                }
            }
        }

        if (ztype) {
            size_t len, newlen;
            int status = 0;
            char* buf = NULL;
            int zipid = mcx_keylookup((char*)(ztype->valuestring), zipformat);

            if (zipid < 0) {
                MCX_ERROR(-1, "the JData array uses an unsupported _ArrayZipType_");
            }

            ret = zmat_run(strlen(vdata->valuestring), (uchar*)vdata->valuestring, &len, (uchar**)&buf, zmBase64, &status,
            ((TZMatFlags) {
                .param = {.nthread = 2}
            }).iscompress);

            /* a string that is not base64 decodes to nothing */
            if (ret || buf == NULL || len == 0) {
                ret = -1;
            } else {
                if (*vol) {
                    free(*vol);
                    *vol = NULL;
                }

                ret = zmat_run(len, (uchar*)buf, &newlen, (uchar**)(vol), zipid, &status,
                ((TZMatFlags) {
                    .param = {.nthread = 2}
                }).iscompress);
            }

            if (buf) {
                free(buf);
            }

            if (ret || *vol == NULL) {
                MCX_ERROR(-1, "the JData array data could not be decoded");
            }

            mcx_jdata_bytes = newlen;
            cfg->isrowmajor = 1;
        } else if (cJSON_IsString(vdata)) {
            /* uncompressed _ArrayData_: base64-decode only, no further zmat pass */
            size_t len;
            int status = 0;

            if (*vol) {
                free(*vol);
                *vol = NULL;
            }

            ret = zmat_run(strlen(vdata->valuestring), (uchar*)vdata->valuestring, &len, (uchar**)(vol), zmBase64, &status,
            ((TZMatFlags) {
                .param = {.nthread = 2}
            }).iscompress);

            if (ret || *vol == NULL) {
                MCX_ERROR(-1, "the JData array data could not be decoded");
            }

            mcx_jdata_bytes = len;
            cfg->isrowmajor = 1;
        } else if (cJSON_IsArray(vdata)) {
            size_t arraylen = cJSON_GetArraySize(vdata);
            int i;
            cJSON* tmp = vdata->child;

            if (!cJSON_IsNumber(tmp)) {
                MCX_ERROR(-1, "JData array constructs are not supported");
            }

            /* the list is copied into an array of _ArraySize_ elements */
            if (arraylen > elemnum) {
                MCX_ERROR(-1, "the JData array holds more values than its _ArraySize_ says");
            }

            if (*vol) {
                free(*vol);
            }

            *vol = calloc(cfg->mediabyte == MEDIA_2LABEL_SPLIT ? 8 : cfg->mediabyte, elemnum);
            mcx_jdata_bytes = elemnum * (cfg->mediabyte == MEDIA_2LABEL_SPLIT ? 8 : cfg->mediabyte);

            if (cfg->mediabyte == 1) {
                for (i = 0; i < arraylen; i++) {
                    ((char*)(*vol))[i] = tmp->valueint;
                    tmp = tmp->next;
                }
            } else if (cfg->mediabyte == 2) {
                for (i = 0; i < arraylen; i++) {
                    ((short*)(*vol))[i] = tmp->valueint;
                    tmp = tmp->next;
                }
            } else if (cfg->mediabyte == 4) {
                for (i = 0; i < arraylen; i++) {
                    ((int*)(*vol))[i] = tmp->valueint;
                    tmp = tmp->next;
                }
            } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
                for (i = 0; i < arraylen; i++) {
                    ((int64_t*)(*vol))[i] = *((int64_t*)&tmp->valuedouble);
                    tmp = tmp->next;
                }
            }

            cfg->isrowmajor = 1;
        } else {
            MCX_ERROR(-1, "JData array constructs are not supported");
        }
    } else {
        MCX_ERROR(-1, "No _ArrayZipData_ or _ArrayData_ field is found");
    }

    /* the decoded bytes have to cover the declared size; a list is copied
       into a zero-filled array of that size, so it always does */
    if (!cJSON_IsArray(vdata) && mcx_jdata_bytes < elemnum * (size_t)mcx_jdata_typesize(*type)) {
        MCX_ERROR(-1, "the JData array holds fewer values than its _ArraySize_ says");
    }

    return ret;
}

/**
 * @brief Export an ND volumetric image to JSON/JData encoded construct
 *
 * The JData specification defines a portable way to encode and share volumetric
 * ND arrays and other complex data structures, such as trees, graphs and tables.
 * This function is capable of exporting any ND numerical arrays into a JData
 * construct, permitting data compression and base64 encoding.
 *
 * @param[in] vol: a pointer that points to the ND array buffer
 * @param[in] ndim: the number of dimensions
 * @param[in] dims: an integer pointer that points to the dimensional vector
 * @param[in] type: a string of JData data types, such as "uint8" "float32", "int32" etc
 * @param[in] byte: number of byte per voxel
 * @param[in] zipid: zip method: 0:zlib,1:gzip,2:base64,3:lzma,4:lzip,5:lz4,6:lz4hc
 * @param[in] obj: a pre-created cJSON or UBJ object to store the output JData fields
 * @param[in] isubj: 1 if obj is a binary JSON (UBJ) object, 0 if obj is a cJSON object
 * @param[in] cfg: mcx config struct
 */

int  mcx_jdataencode(void* vol, int ndim, uint* dims, char* type, int byte, int zipid, void* obj, int isubj, int iscol, Config* cfg) {
    uint datalen = 1;
    size_t compressedbytes, totalbytes;
    uchar* compressed = NULL, *buf = NULL;
    int ret = 0, status = 0;

    for (int i = 0; i < ndim; i++) {
        datalen *= dims[i];
    }

    totalbytes = datalen * byte;

    if (!cfg->isdumpjson) {
        MCX_FPRINTF(cfg->flog, "%s [%s] ...", T_("compressing data"), zipformat[zipid]);
    }

    /*compress data using zlib*/
    if (zipid != zmBase64) {
        ret = zmat_run(totalbytes, (uchar*)vol, &compressedbytes, (uchar**)&compressed, zipid, &status,
        ((TZMatFlags) {
            .param = {.clevel = 1, .nthread = 2}
        }).iscompress);
    } else {
        compressed = (uchar*)vol;
        compressedbytes = totalbytes;
    }

    if (!ret) {
        if (!cfg->isdumpjson) {
            MCX_FPRINTF(cfg->flog, "%s: %.1f%%\t", T_("compression ratio"), compressedbytes * 100.f / totalbytes);
        }

        if (isubj) {
            ubjw_context_t* item = (ubjw_context_t*)obj;
            UBJ_WRITE_KEY(item, "_ArrayType_", string, type);
            ubjw_write_key(item, "_ArraySize_");
            UBJ_WRITE_ARRAY(item, uint32, ndim, dims);

            if (iscol) {
                UBJ_WRITE_KEY(item, "_ArrayOrder_", string, "c");
            }

            if (!cfg->isdumpjson) {
                MCX_FPRINTF(cfg->flog, "\n");
            }

            if (zipid == zmBase64) {
                /* binary JSON: raw typed _ArrayData_ buffer (no base64) */
                UBJ_TYPE marker = UBJ_UINT8;

                if (!strcmp(type, "int8")) {
                    marker = UBJ_INT8;
                } else if (!strcmp(type, "uint8")) {
                    marker = UBJ_UINT8;
                } else if (!strcmp(type, "int16")) {
                    marker = UBJ_INT16;
                } else if (!strcmp(type, "uint16")) {
                    marker = UBJ_UINT16;
                } else if (!strcmp(type, "int32")) {
                    marker = UBJ_INT32;
                } else if (!strcmp(type, "uint32")) {
                    marker = UBJ_UINT32;
                } else if (!strcmp(type, "int64")) {
                    marker = UBJ_INT64;
                } else if (!strcmp(type, "uint64")) {
                    marker = UBJ_UINT64;
                } else if (!strcmp(type, "single") || !strcmp(type, "float32")) {
                    marker = UBJ_FLOAT32;
                } else if (!strcmp(type, "double") || !strcmp(type, "float64")) {
                    marker = UBJ_FLOAT64;
                }

                ubjw_write_key(item, "_ArrayData_");
                ubjw_write_buffer(item, (uint8_t*)vol, marker, datalen);
            } else {
                UBJ_WRITE_KEY(item, "_ArrayZipType_", string, zipformat[zipid]);
                UBJ_WRITE_KEY(item, "_ArrayZipSize_", uint32, datalen);
                ubjw_write_key(item, "_ArrayZipData_");
                ubjw_write_buffer(item, compressed, UBJ_UINT8, compressedbytes);
            }
        } else {
            totalbytes = 0;
            /*encode data using base64*/
            ret = zmat_run(compressedbytes, compressed, &totalbytes, (uchar**)&buf, zmBase64, &status,
            ((TZMatFlags) {
                .param = {.clevel = 1, .nthread = 2}
            }).iscompress);

            if (!cfg->isdumpjson) {
                MCX_FPRINTF(cfg->flog, "%s: %.1f%%\n", T_("after encoding"), totalbytes * 100.f / (datalen * byte));
            }

            if (!ret) {
                cJSON_AddStringToObject((cJSON*)obj, "_ArrayType_", type);
                cJSON_AddItemToObject((cJSON*)obj,   "_ArraySize_", cJSON_CreateIntArray((int*)dims, ndim));

                if (iscol) {
                    cJSON_AddStringToObject((cJSON*)obj, "_ArrayOrder_", "c");
                }

                if (zipid == zmBase64) {
                    /* text JSON: base64-encoded raw bytes, no compression keys */
                    cJSON_AddStringToObject((cJSON*)obj, "_ArrayData_", (char*)buf);
                } else {
                    cJSON_AddStringToObject((cJSON*)obj, "_ArrayZipType_", zipformat[zipid]);
                    cJSON_AddNumberToObject((cJSON*)obj, "_ArrayZipSize_", datalen);
                    cJSON_AddStringToObject((cJSON*)obj, "_ArrayZipData_", (char*)buf);
                }
            }
        }
    }

    if (compressed && zipid != zmBase64) {
        free(compressed);
    }

    if (buf) {
        free(buf);
    }

    return ret;
}

#endif

/**
 * @brief Print a progress bar
 *
 * When -D P is specified, this function prints and update a progress bar.
 *
 * @param[in] percent: the percentage value from 1 to 100
 * @param[in] cfg: simulation configuration
 */

void mcx_progressbar(float percent, Config* cfg) {
    unsigned int percentage, j, colwidth = 79;
    static unsigned int oldmarker = 0xFFFFFFFF;

#ifndef MCX_CONTAINER
#ifdef TIOCGWINSZ
    struct winsize ttys = {0, 0, 0, 0};
    ioctl(0, TIOCGWINSZ, &ttys);
    colwidth = ttys.ws_col;

    if (colwidth == 0) {
        colwidth = 79;
    }

#endif
#endif
    percent = MIN(percent, 1.f);

    percentage = percent * (colwidth - 18);

    if (percentage != oldmarker) {
        if (percent != -0.f)
            for (j = 0; j < colwidth; j++) {
                MCX_FPRINTF(stdout, "\b");
            }

        oldmarker = percentage;
        MCX_FPRINTF(stdout, S_YELLOW"Progress: [");

        for (j = 0; j < percentage; j++) {
            MCX_FPRINTF(stdout, "=");
        }

        MCX_FPRINTF(stdout, (percentage < colwidth - 18) ? ">" : "=");

        for (j = percentage; j < colwidth - 18; j++) {
            MCX_FPRINTF(stdout, " ");
        }

        MCX_FPRINTF(stdout, "] %3d%%" S_RESET, (int)(percent * 100));
#if defined(MCX_CONTAINER) && (defined(MATLAB_MEX_FILE) || defined(OCTAVE_API_VERSION_NUMBER))
        mcx_matlab_flush();
#else
        fflush(stdout);
#endif
    }
}

/**
 * @brief Function to read a single parameter value followed by a command line option
 *
 * This function reads different types of parameter values following a command line option.
 *
 * @param[in] argc: the number of total command line parameters
 * @param[in] argv: the pointer to all command line options
 * @param[in] id: which parameter to be parsed
 * @param[out] output: the pointer to which the parsed value to be written
 * @param[in] type: the type of data support char, int, float, string, bytenumlist, floatlist
 */

int mcx_readarg(int argc, char* argv[], int id, void* output, const char* type) {
    /*
        when a binary option is given without a following number (0~1),
        we assume it is 1
    */
    if (strcmp(type, "char") == 0 && (id >= argc - 1 || (argv[id + 1][0] < '0' || argv[id + 1][0] > '9'))) {
        *((char*)output) = 1;
        return id;
    }

    if (id < argc - 1) {
        if (strcmp(type, "char") == 0) {
            *((char*)output) = atoi(argv[id + 1]);
        } else if (strcmp(type, "int") == 0) {
            *((int*)output) = atoi(argv[id + 1]);
        } else if (strcmp(type, "float") == 0) {
            *((float*)output) = atof(argv[id + 1]);
        } else if (strcmp(type, "string") == 0) {
            strcpy((char*)output, argv[id + 1]);
        } else if (strcmp(type, "bytenumlist") == 0) {
            char* nexttok, *numlist = (char*)output;
            int len = 0, i;
            nexttok = strtok(argv[id + 1], " ,;");

            while (nexttok) {
                if (len >= MAX_DEVICE - 1) { /* room for the terminating zero */
                    break;
                }

                numlist[len++] = (char)(atoi(nexttok)); /*device id<256*/

                for (i = 0; i < len - 1; i++) /* remove duplicaetd ids */
                    if (numlist[i] == numlist[len - 1]) {
                        numlist[--len] = '\0';
                        break;
                    }

                nexttok = strtok(NULL, " ,;");
                /*if(len>=MAX_DEVICE) break;*/
            }
        } else if (strcmp(type, "floatlist") == 0) {
            char* nexttok;
            float* numlist = (float*)output;
            int len = 0;
            nexttok = strtok(argv[id + 1], " ,;");

            while (nexttok) {
                if (len >= MAX_DEVICE) { /* the lists read this way are MAX_DEVICE long */
                    break;
                }

                numlist[len++] = atof(nexttok); /*device id<256*/
                nexttok = strtok(NULL, " ,;");
            }
        }
    } else {
        MCX_ERROR(-1, "incomplete input");
    }

    return id + 1;
}

/**
 * @brief Reads the string value that follows a command line option into a
 * buffer of len bytes.
 *
 * mcx_readarg's "string" type has no size to go by and copies whatever it is
 * given -- a long -s overran the 64-byte session name into the fields after
 * it.  This one refuses a value that does not fit instead.
 */
static int mcx_readstrarg(int argc, char* argv[], int id, char* output, size_t len) {
    if (id >= argc - 1) {
        MCX_ERROR(-1, "incomplete input");
    }

    if (strlen(argv[id + 1]) >= len) {
        char msg[MAX_PATH_LENGTH];
        snprintf(msg, MAX_PATH_LENGTH, "the value given to %s is too long (at most %u characters)", argv[id], (unsigned int)(len - 1));
        MCX_ERROR(-1, msg);
    }

    strcpy(output, argv[id + 1]);
    return id + 1;
}

/**
 * @brief Test if a long command line option is supported
 *
 * This function returns 1 if a long option is found, and 0 otherwise
 *
 * @param[in] opt: the long command line option string
 */

int mcx_remap(char* opt) {
    int i = 0;

    while (shortopt[i] != '\0') {
        if (strcmp(opt, fullopt[i]) == 0) {
            opt[1] = shortopt[i];

            if (shortopt[i] != '-') {
                opt[2] = '\0';
            }

            return 0;
        }

        i++;
    }

    return 1;
}

#ifndef MCX_CONTAINER

/**
 * @brief Main function to read user command line options
 *
 * This function process user command line inputs and parse all short and long options.
 *
 * @param[in] argc: the number of total command line parameters
 * @param[in] argv: the pointer to all command line options
 * @param[in] cfg: simulation configuration
 */

void mcx_parsecmd(int argc, char* argv[], Config* cfg) {
    int i = 1, isinteractive = 1, len;
    char filename[MAX_PATH_LENGTH] = {0}, *jsoninput = NULL, issavelog = -1;
    char logfile[MAX_PATH_LENGTH] = {0};
    float np = 0.f;

#if defined(_WIN32) && defined(USE_OS_TIMER) && !defined(MCX_CONTAINER)
    EnableVTMode();
#endif

    if (argc <= 1) {
        mcx_usage(cfg, argv[0]);
        exit(0);
    }

    while (i < argc) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == '-') {
                if (mcx_remap(argv[i])) {
                    MCX_FPRINTF(cfg->flog, "%s: %s", T_("Command option"), argv[i]);
                    MCX_ERROR(-2, T_("unknown verbose option"));
                }
            } else if (strlen(argv[i]) > 2) {
                MCX_FPRINTF(cfg->flog, "%s: %s", T_("Command option"), argv[i]);
                MCX_ERROR(-2, T_("unknown short option"));
            }

            if (argv[i][1] <= 'z' && argv[i][1] >= 'A') {
                flagset[(int)(argv[i][1])] = 1;
            }

            switch (argv[i][1]) {
                case 'h':
                    mcx_usage(cfg, argv[0]);
                    exit(0);

                case 'i':
                    if (filename[0]) {
                        MCX_ERROR(-2, T_("you can not specify both interactive mode and config file"));
                    }

                    isinteractive = 1;
                    break;

                case 'f':
                    isinteractive = 0;

                    if (i < argc - 1 && argv[i + 1][0] == '{') {
                        jsoninput = argv[i + 1];
                        i++;
                    } else if (i == argc - 1 || argv[i + 1][0] == '-') {
                        runcommand("", "", &jsoninput);
                        isinteractive = 2;
                        i += (i < argc - 1);
                    } else {
                        i = mcx_readstrarg(argc, argv, i, filename, sizeof(filename));
                    }

                    break;

                case 'n':
                    i = mcx_readarg(argc, argv, i, &(np), "float");
                    cfg->nphoton = (size_t)np;
                    break;

                case 't':
                    i = mcx_readarg(argc, argv, i, &(cfg->nthread), "int");
                    break;

                case 'T':
                    i = mcx_readarg(argc, argv, i, &(cfg->nblocksize), "int");
                    break;

                case 's':
                    i = mcx_readstrarg(argc, argv, i, cfg->session, sizeof(cfg->session));
                    break;

                case 'a':
                    i = mcx_readarg(argc, argv, i, &(cfg->isrowmajor), "char");
                    break;

                case 'q':
                    i = mcx_readarg(argc, argv, i, &(cfg->issaveseed), "char");
                    break;

                case 'g':
                    i = mcx_readarg(argc, argv, i, &(cfg->maxgate), "int");
                    break;

                case 'b':
                    i = mcx_readarg(argc, argv, i, &(cfg->isreflect), "char");
                    cfg->isref3 = cfg->isreflect;
                    break;

                case 'B':
                    if (i < argc + 1) {
                        strncpy(cfg->bc, argv[i + 1], 12);
                    }

                    i++;
                    break;

                case 'd':
                    i = mcx_readarg(argc, argv, i, &(cfg->issavedet), "char");
                    break;

                case 'm':
                    i = mcx_readarg(argc, argv, i, &(cfg->ismomentum), "char");

                    if (cfg->ismomentum) {
                        cfg->issavedet = 1;
                    }

                    break;

                case 'r':
                    i = mcx_readarg(argc, argv, i, &(cfg->respin), "int");
                    break;

                case 'S':
                    i = mcx_readarg(argc, argv, i, &(cfg->issave2pt), "char");
                    break;

                case 'p':
                    i = mcx_readarg(argc, argv, i, &(cfg->printnum), "int");
                    break;

                case 'e':
                    i = mcx_readarg(argc, argv, i, &(cfg->minenergy), "float");
                    break;

                case 'U':
                    i = mcx_readarg(argc, argv, i, &(cfg->isnormalized), "char");
                    break;

                case 'R':
                    i = mcx_readarg(argc, argv, i, &(cfg->sradius), "float");
                    break;

                case 'u':
                    i = mcx_readarg(argc, argv, i, &(cfg->unitinmm), "float");
                    break;

                case 'l':
                    i = mcx_readarg(argc, argv, i, &issavelog, "char");
                    break;

                case 'L':
                    cfg->isgpuinfo = 2;
                    break;

                case 'I':
                    cfg->isgpuinfo = 1;
                    break;

                case 'G':
                    if (mcx_isbinstr(argv[i + 1])) {
                        i = mcx_readstrarg(argc, argv, i, cfg->deviceid, sizeof(cfg->deviceid));
                        break;
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->gpuid), "int");
                        memset(cfg->deviceid, '0', MAX_DEVICE);

                        if (cfg->gpuid > 0 && cfg->gpuid < MAX_DEVICE) {
                            cfg->deviceid[cfg->gpuid - 1] = '1';
                        } else {
                            MCX_ERROR(-2, T_("GPU ID can not be more than 256"));
                        }

                        break;
                    }

                case 'W':
                    i = mcx_readarg(argc, argv, i, cfg->workload, "floatlist");
                    break;

                case 'z':
                    i = mcx_readarg(argc, argv, i, &(cfg->issrcfrom0), "char");
                    break;

                case 'M':
                    i = mcx_readarg(argc, argv, i, &(cfg->isdumpmask), "char");
                    break;

                case 'Y':
                    i = mcx_readarg(argc, argv, i, &(cfg->replaydet), "int");
                    break;

                case 'H':
                    i = mcx_readarg(argc, argv, i, &(cfg->maxdetphoton), "int");
                    break;

                case 'P':
                    if (i + 1 < argc) {
                        len = strlen(argv[i + 1]);

                        if (cfg->shapedata) {
                            free(cfg->shapedata);
                        }

                        cfg->shapedata = (char*)calloc(1, len + 1); /* keep the terminating NUL */
                        memcpy(cfg->shapedata, argv[++i], len);
                    } else {
                        MCX_ERROR(-1, T_("json shape constructs are expected after -P"));
                    }

                    break;

                case 'j':
                    if (i + 1 < argc) {
                        len = strlen(argv[i + 1]);

                        if (cfg->extrajson) {
                            free(cfg->extrajson);
                        }

                        cfg->extrajson = (char*)calloc(1, len + 1);
                        memcpy(cfg->extrajson, argv[++i], len);
                    } else {
                        MCX_ERROR(-1, T_("json fragment is expected after --json"));
                    }

                    break;

                case 'A':
                    i = mcx_readarg(argc, argv, i, &(cfg->autopilot), "char");
                    break;

                case 'E':
                    if (i < argc - 1 && (strstr(argv[i + 1], ".mch") != NULL || strstr(argv[i + 1], ".jdt") != NULL || strstr(argv[i + 1], ".jdat") != NULL) ) { /*give an mch file to initialize the seed*/
#if defined(USE_LL5_RAND)
                        MCX_ERROR(-1, T_("seeding file is not supported in this binary"));
#else
                        i = mcx_readstrarg(argc, argv, i, cfg->seedfile, sizeof(cfg->seedfile));
                        cfg->seed = SEED_FROM_FILE;
#endif
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->seed), "int");
                    }

                    break;

                case 'O':
                    {
                        /* outputtype is a single char: the letter typed, then
                           mcx_lookupindex turns it into an index in place */
                        char otype[MAX_SESSION_LENGTH];
                        i = mcx_readstrarg(argc, argv, i, otype, sizeof(otype));
                        cfg->outputtype = otype[0];
                    }

                    if (mcx_lookupindex(&(cfg->outputtype), outputtype)) {
                        MCX_ERROR(-2, T_("the specified output data type is not recognized"));
                    }

                    break;

                case 'k':
                    i = mcx_readarg(argc, argv, i, &(cfg->voidtime), "int");
                    break;

                case 'V':
                    i = mcx_readarg(argc, argv, i, &(cfg->isspecular), "char");
                    break;

                case 'v':
                    mcx_version(cfg);
                    break;

                case 'D':
                    if (i + 1 < argc && isalpha(argv[i + 1][0]) ) {
                        cfg->debuglevel = mcx_parsedebugopt(argv[++i], debugflag);
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->debuglevel), "int");
                    }

                    break;

                case 'K':
                    if (i + 1 < argc && isalpha(argv[i + 1][0]) ) {
                        cfg->mediabyte = mcx_keylookup(argv[++i], mediaformat);

                        if (cfg->mediabyte == -1) {
                            MCX_ERROR(-1, T_("Unsupported media format"));
                        }

                        cfg->mediabyte = mediaformatid[cfg->mediabyte];
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->mediabyte), "int");
                    }

                    break;

                case 'F':
                    if (i + 1 >= argc) {
                        MCX_ERROR(-1, T_("incomplete input"));
                    }

                    if ((cfg->outputformat = mcx_keylookup(argv[++i], outputformat)) < 0) {
                        MCX_ERROR(-2, T_("the specified output data type is not recognized"));
                    }

                    break;

                case 'x':
                    i = mcx_readarg(argc, argv, i, &(cfg->issaveexit), "char");

                    if (cfg->issaveexit) {
                        cfg->issavedet = 1;
                    }

                    break;

                case 'X':
                    i = mcx_readarg(argc, argv, i, &(cfg->issaveref), "char");

                    if (cfg->issaveref) {
                        cfg->issaveref = 1;
                    }

                    break;

                case 'Z':
                    if (i + 1 < argc && isalpha(argv[i + 1][0]) ) {
                        cfg->zipid = mcx_keylookup(argv[++i], zipformat);
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->zipid), "int");
                    }

                    break;

                case 'w':
                    if (i + 1 < argc && isalpha(argv[i + 1][0]) ) {
                        cfg->savedetflag = mcx_parsedebugopt(argv[++i], saveflag);
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->savedetflag), "int");
                    }

                    break;

                case 'y':
                    if (i + 1 < argc && isalpha((int)argv[i + 1][0]) ) {
                        int idx = mcx_keystartwith(argv[++i], languagename);

                        if (idx == -1) {
                            MCX_FPRINTF(cfg->flog, "Unsupported language ID (%s), fallback to default\n", argv[i]);
                        } else {
                            mcx_lang = mcx_parsejson(translations[idx]);
                        }

                    } else {
                        MCX_FPRINTF(cfg->flog, "%s: \n", T_("Built-in languages"));

                        for (int i = 0; i < sizeof(benchname) / sizeof(char*) - 1; i++) {
                            cJSON* tmp = NULL, *root = NULL;
                            mcx_lang = mcx_parsejson(translations[i]);
                            MCX_FPRINTF(cfg->flog, "\t%s\t%s\n", languagename[i], FIND_JSON_KEY("_LANG_", NULL, mcx_lang, "(Invalid JSON, see error above)", valuestring));
                            cJSON_Delete(mcx_lang);
                        }

                        exit(0);
                    }

                    break;

                case 'Q':
                    if (i + 1 < argc && isalpha((int)argv[i + 1][0]) ) {
                        int idx = mcx_keylookup(argv[++i], benchname);

                        if (idx == -1) {
                            MCX_ERROR(-1, T_("Unsupported bechmark."));
                        }

                        isinteractive = 0;
                        jsoninput = (char*)benchjson[idx];
                    } else {
                        MCX_FPRINTF(cfg->flog, "%s: \n", T_("Built-in benchmarks"));

                        for (int i = 0; i < sizeof(benchname) / sizeof(char*) - 1; i++) {
                            MCX_FPRINTF(cfg->flog, "\t%s\n", benchname[i]);
                        }

                        exit(0);
                    }

                    break;

                case 'N':
                    if (i == argc - 1 || argv[i + 1][0] == '-' || argv[i + 1][0] == '{' || strchr(argv[i + 1], '=')) {
                        int j, doclen;
                        char* jbuf = NULL;

                        if (i < argc - 1 && argv[i + 1][0] == '{') {
                            char cmd[MAX_PATH_LENGTH] = "curl -s -X POST -H 'Content-Type: application/json' -d ";
                            sprintf(cmd + strlen(cmd), "'%s' \"https://neurojson.io:7777/mcx/_find\"", argv[i + 1]);
                            runcommand(cmd, "", &jbuf);
                        } else if (i < argc - 1 && strchr(argv[i + 1], '=')) {
                            char cmd[MAX_PATH_LENGTH] = "curl -s -X GET 'https://neurojson.io:7777/mcx/_all_docs?";
                            sprintf(cmd + strlen(cmd), "%s'", argv[i + 1]);
                            runcommand(cmd, "", &jbuf);
                        } else {
                            runcommand("curl -s -X POST -H 'Content-Type: application/json' -d '{\"selector\": {\"Session\": {\"$gt\": null}},\"fields\": [\"_id\"],\"limit\":50}' \"https://neurojson.io:7777/mcx/_find\"", "", &jbuf);
                        }

                        cJSON* root = mcx_parsejson(jbuf), *docs = cJSON_GetObjectItem(root, "docs"), *subitem, *tmp;

                        if (!docs) {
                            docs = cJSON_GetObjectItem(root, "rows");

                            if (!docs || !root) {
                                MCX_ERROR(-1, jbuf);
                            }
                        }

                        doclen = cJSON_GetArraySize(docs);
                        subitem = docs->child;
                        MCX_FPRINTF(cfg->flog, "%s (%d)\n", T_("Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx)"), doclen - 1);

                        for (j = 0; j < doclen; j++) {
                            char* docid = (cJSON_GetObjectItem(subitem, "_id") ? FIND_JSON_KEY("_id", "id", subitem, "", valuestring) : FIND_JSON_KEY("id", "id", subitem, "", valuestring));

                            if (docid && docid[0] != '_' && docid[0] != '.') {
                                printf("\t%s\n", docid);
                            }

                            subitem = subitem->next;
                        }

                        cJSON_Delete(root);
                        free(jbuf);
                        exit(0);
                    } else {
                        if (strstr(argv[i + 1], "http") == argv[i + 1]) {
                            runcommand("curl -s -X GET ", argv[i + 1], &jsoninput);
                        } else if (strchr(argv[i + 1], '/')) {
                            runcommand("curl -s -X GET https://neurojson.io:7777/", argv[i + 1], &jsoninput);
                        } else {
                            runcommand("curl -s -X GET https://neurojson.io:7777/mcx/", argv[i + 1], &jsoninput);
                        }

                        isinteractive = 2;
                    }

                    break;

                case '-':  /*additional verbose parameters*/
                    if (strcmp(argv[i] + 2, "maxvoidstep") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->maxvoidstep), "int");
                    } else if (strcmp(argv[i] + 2, "maxjumpdebug") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->maxjumpdebug), "int");
                    } else if (strcmp(argv[i] + 2, "gscatter") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->gscatter), "int");
                    } else if (strcmp(argv[i] + 2, "faststep") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->faststep), "char");
                    } else if (strcmp(argv[i] + 2, "root") == 0) {
                        i = mcx_readstrarg(argc, argv, i, cfg->rootpath, sizeof(cfg->rootpath));
                    } else if (strcmp(argv[i] + 2, "dumpjson") == 0) {
                        cfg->jsonfile[0] = '-';

                        /* --dumpjson followed by nothing or another option prints to stdout, by a number
                           selects the dump mode, and by anything else (including an absolute or ./ path,
                           which used to be read as a number and silently ignored) names the output file */
                        if (i + 1 >= argc || (argv[i + 1][0] == '-' && argv[i + 1][1] != '\0')) {
                            cfg->isdumpjson = 1;
                        } else if (!isdigit((int)(argv[i + 1][0]))) {
                            cfg->isdumpjson = 1;
                            memcpy(cfg->jsonfile, argv[i + 1], MIN(strlen(argv[i + 1]), MAX_PATH_LENGTH));
                            i++;
                        } else {
                            i = mcx_readarg(argc, argv, i, &(cfg->isdumpjson), "int");
                        }
                    } else if (strcmp(argv[i] + 2, "reflectin") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->isrefint), "char");
                    } else if (strcmp(argv[i] + 2, "atomic") == 0) {
                        int isatomic = 1;
                        i = mcx_readarg(argc, argv, i, &(isatomic), "char");

                        if (!isatomic) {
                            MCX_FPRINTF(cfg->flog, S_RED "WARNING: --atomic 0 is deprecated and ignored; non-atomic accumulation raced and lost up to 60%% of the deposited energy on modern GPUs at no speed benefit\n" S_RESET);
                        }
                    } else if (strcmp(argv[i] + 2, "srcid") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->srcid), "int");
                    } else if (strcmp(argv[i] + 2, "trajstokes") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->istrajstokes), "int");
                    } else if (strcmp(argv[i] + 2, "internalsrc") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->internalsrc), "int");
                    } else {
                        MCX_FPRINTF(cfg->flog, "%s: --%s\n", T_("unknown verbose option"), argv[i] + 2);
                    }

                    break;

                default:
                    MCX_FPRINTF(cfg->flog, "%s: %s", T_("Command option"), argv[i]);
                    MCX_ERROR(-2, T_("unknown short option"));
                    break;
            }
        }

        i++;
    }

    if (issavelog == 0) {
        cfg->printnum = -1;
    }

    if (issavelog > 0) {
        sprintf(logfile, "%s.log", (strlen(cfg->session) ? cfg->session : T_("unnamed")));
        cfg->flog = fopen(logfile, "wt");

        if (cfg->flog == NULL) {
            cfg->flog = stdout;
            MCX_FPRINTF(cfg->flog, "%s\n", T_("unable to save to log file, will print from stdout"));
        }
    }

    /* the replay-only output types are checked by mcx_preprocess, once the input file and --json have set the seed and omega */
    if (cfg->isgpuinfo != 2) { /*print gpu info only*/
        if (isinteractive == 2 && jsoninput) {
            mcx_readconfig(jsoninput, cfg);
            free(jsoninput);
        } else if (isinteractive) {
            mcx_readconfig((char*)"", cfg);
        } else if (jsoninput) {
            mcx_readconfig(jsoninput, cfg);
        } else {
            mcx_readconfig(filename, cfg);
        }

        if (cfg->extrajson) {
            cJSON* jroot = mcx_parsejson(cfg->extrajson);

            if (jroot) {
                cfg->extrajson[0] = '_';
                mcx_loadjson(jroot, cfg);
                cJSON_Delete(jroot);
            } else {
                MCX_ERROR(-1, T_("invalid json fragment following --json"));
            }
        }

        mcx_preprocess(cfg);
    }

    if (cfg->isdumpjson == 1) {
        mcx_savejdata(cfg->jsonfile, cfg);
        exit(0);
    }
}

#endif

/**
 * @brief Parse the debug flag in the letter format
 *
 * The debug flag following the -D can be either a string format, or numerical format.
 * This function converts the string debug flags into number format
 *
 * @param[in] debugopt: string following the -D parameter
 * @param[out] debugflag: the numerical format of the debug flag
 */

int mcx_parsedebugopt(char* debugopt, const char* debugflag) {
    char* c = debugopt, *p;
    int debuglevel = 0;

    while (*c) {
        p = strchr(debugflag, ((*c <= 'z' && *c >= 'a') ? *c - 'a' + 'A' : *c) );

        if (p != NULL) {
            debuglevel |= (1 << (p - debugflag));
        }

        c++;
    }

    return debuglevel;
}

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int mcx_keylookup(char* origkey, const char* table[]) {
    int i = 0;
    char* key = malloc(strlen(origkey) + 1);
    memcpy(key, origkey, strlen(origkey) + 1);

    while (key[i]) {
        key[i] = tolower(key[i]);
        i++;
    }

    i = 0;

    while (table[i] && table[i][0] != '\0') {
        if (strcmp(key, table[i]) == 0) {
            free(key);
            return i;
        }

        i++;
    }

    free(key);
    return -1;
}

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int mcx_keystartwith(char* origkey, const char* table[]) {
    int i = 0;
    char* key = malloc(strlen(origkey) + 1);
    memcpy(key, origkey, strlen(origkey) + 1);

    if (strlen(origkey) > 2 && key[2] == '-') {
        key[2] = '_';
    }

    while (key[i]) {
        key[i] = tolower(key[i]);
        i++;
    }

    i = 0;

    while (table[i] && table[i][0] != '\0') {
        if (strstr(table[i], key)) {
            free(key);
            return i;
        }

        i++;
    }

    free(key);
    return -1;
}

/**
 * @brief Look up a single character in a string
 *
 * @param[in] key: character to be looked up
 * @param[out] index: the dictionary string where the char is searched
 * @return if found, return 0; otherwise, return 1
 */

int mcx_lookupindex(char* key, const char* index) {
    int i = 0;

    while (index[i] != '\0') {
        if (tolower(*key) == index[i]) {
            *key = i;
            return 0;
        }

        i++;
    }

    return 1;
}

/**
 * @brief Print MCX software version
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_version(Config* cfg) {
    const char ver[] = "$Rev::      $ " MCX_VERSION;
    uint v = 0;
    sscanf(ver, "$Rev::%x", &v);
    MCX_FPRINTF(cfg->flog, "%s:\t%x\nVersion:\t%s\nMajor:\t\t%d\nMinor:\t\t%d\n", T_("MCX Revision"), v, MCX_VERSION, MCX_VERSION_MAJOR, MCX_VERSION_MINOR);
    exit(0);
}

/**
 * @brief Test if a string contains only '0' and '1'
 *
 * @param[in] str: string to be tested
 */

int mcx_isbinstr(const char* str) {
    int i, len = strlen(str);

    if (len == 0) {
        return 0;
    }

    for (i = 0; i < len; i++)
        if (str[i] != '0' && str[i] != '1') {
            return 0;
        }

    return 1;
}

/**
 * @brief Convert a float to an IEEE 754 half-precision number
 *
 * The mantissa is truncated to 10 bits; values below 2^-14 become denormals,
 * values beyond the half-precision range are clamped to +/-65504.
 *
 * @param[in] input: the float value
 */

static unsigned short mcx_float2half(float input) {
    union {
        float f;
        unsigned int i;
    } f2i;
    unsigned int sign, expo, mant;

    f2i.f = input;
    sign = (f2i.i >> 16) & 0x8000;
    expo = (f2i.i >> 23) & 0xff;
    mant = f2i.i & 0x7fffff;

    if (expo == 0xff) { /*inf or nan*/
        return sign | 0x7c00 | (mant ? 0x200 : 0);
    }

    if (expo <= 112) { /*below the smallest normal half 2^-14: a denormal (between 2^-24 and 2^-14) or 0*/
        if (expo < 101) {
            return sign;
        }

        mant = (mant | 0x800000) >> 12; /*12 bits, including the implicit leading 1*/
        return sign | ((mant >> (114 - expo)) + ((mant >> (113 - expo)) & 1));
    }

    if (expo >= 143) { /*above the largest half, clamp to 65504*/
        return sign | 0x7bff;
    }

    return sign | ((expo - 112) << 10) | (mant >> 13); /*truncate the mantissa to 10 bits*/
}

/**
 * @brief Pack two floats, such as mua and mus, into the two halves of a 32-bit word
 *
 * @param[in] input: the two float values, stored in the lower and upper 16 bits
 */

int mcx_float2half2(float input[2]) {
    /**
     * float to half conversion, see
     * https://stackoverflow.com/questions/3026441/float32-to-float16/5587983#5587983
     * https://gamedev.stackexchange.com/a/17410  (for denorms)
     */
    unsigned int packed = mcx_float2half(input[0]) | ((unsigned int)mcx_float2half(input[1]) << 16);

    if (packed == 0) { /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16, only happens for _AS, not ASGN as n will never be 0*/
        packed = 0x00010000;
    }

    if (packed == SIGN_BIT) { /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16, , only happens for _AS, not ASGN as n will never be 0*/
        packed = 0;
    }

    return packed;
}

/**
 * @brief Convert an IEEE 754 half-precision number to a float, the inverse of mcx_float2half2
 *
 * @param[in] h: the 16-bit half-precision number
 */

float mcx_half2float(unsigned short h) {
    union {
        unsigned int i;
        float f;
    } u;
    unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16, expo = (h >> 10) & 0x1f, mant = h & 0x3ff;

    if (expo == 0) {
        if (mant == 0) {
            u.i = sign;
            return u.f;
        }

        while (!(mant & 0x400)) { /* normalize a denormal */
            mant <<= 1;
            expo--;
        }

        expo++;
        mant &= 0x3ff;
    } else if (expo == 31) {
        u.i = sign | 0x7f800000u | (mant << 13);
        return u.f;
    }

    u.i = sign | ((expo + 112) << 23) | (mant << 13);
    return u.f;
}

/**
 * @brief Lookup and return translations for specified language
 *
 * @param[in] cfg: simulation configuration
 */

char* T_(const char* key) {
    cJSON* tmp = NULL, *root = NULL;

    if (mcx_lang) {
        if (cJSON_IsObject(mcx_lang->child)) {
            mcx_lang = mcx_lang->child;
        }

        return (char*)FIND_JSON_KEY(key, NULL, mcx_lang, key, valuestring);
    }

    return (char*)key;
}


#ifndef MCX_CONTAINER

/**
 * @brief Run MCX simulations from a JSON input in a persistent session
 *
 * @param[in] jsonstr: a string in the JSON format, the content of the .json input file
 */

int mcx_run_from_json(char* jsonstr) {
    Config  mcxconfig;            /** mcxconfig: structure to store all simulation parameters */
    GPUInfo* gpuinfo = NULL;      /** gpuinfo: structure to store GPU information */
    unsigned int activedev = 0;   /** activedev: count of total active GPUs to be used */

    mcx_initcfg(&mcxconfig);
    mcx_readconfig(jsonstr, &mcxconfig);

    if (!(activedev = mcx_list_gpu(&mcxconfig, &gpuinfo))) {
        MCX_ERROR(-1, T_("No GPU device found"));
    }

#ifdef _OPENMP
    omp_set_num_threads(activedev);
    #pragma omp parallel
    {
#endif
        mcx_run_simulation(&mcxconfig, gpuinfo);
#ifdef _OPENMP
    }
#endif

    mcx_cleargpuinfo(&gpuinfo);
    mcx_clearcfg(&mcxconfig);
    return 0;
}

#endif

/**
 * @brief Print MCX output header
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_printheader(Config* cfg) {
    if (cfg->printnum >= 0 ) {
        if (strcmp(T_("_MCX_BANNER_"), "_MCX_BANNER_")) {
            MCX_FPRINTF(cfg->flog, "%s", T_("_MCX_BANNER_"));
        } else {
            MCX_FPRINTF(cfg->flog, S_MAGENTA"\
###############################################################################\n\
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #\n\
#          Copyright (c) 2009-2026 Qianqian Fang <q.fang at neu.edu>          #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab- " S_BLUE "http://fanglab.org " S_MAGENTA "#\n\
#   Department of Bioengineering, Northeastern University, Boston, MA, USA    #\n\
###############################################################################\n\
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #\n\
###############################################################################\n\
#  Open-source codes and reusable scientific data are essential for research, #\n\
# MCX proudly developed human-readable JSON-based data formats for easy reuse.#\n\
#                                                                             #\n\
#Please visit our free scientific data sharing portal at " S_BLUE "https://neurojson.io " S_MAGENTA "#\n\
# and consider sharing your public datasets in standardized JSON/JData format #\n" S_RESET);
        }

        MCX_FPRINTF(cfg->flog, S_MAGENTA"\
###############################################################################\n\
$Rev::      $" S_GREEN MCX_VERSION S_MAGENTA  "$Date::                       $ by $Author::             $\n\
###############################################################################\n" S_RESET);
    }
}

/**
 * @brief Print MCX help information
 *
 * @param[in] cfg: simulation configuration structure
 * @param[in] exename: path and name of the mcx executable
 */

void mcx_usage(Config* cfg, char* exename) {
    mcx_printheader(cfg);
    printf("\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first value in [*|*] is the default)\n\
\n"S_BOLD S_CYAN"\
== Required option ==\n" S_RESET"\
 -f config     (--input)       read an input file in the .json format,if config\n\
                               string starts with '{',it is parsed as an inline\n\
                               JSON input file; if -f is followed by nothing or\n\
                               a single '-', it reads input from stdin via pipe\n\
      or\n\
 -Q/--bench [cube60, skinvessel,...] run a built-in benchmark specified by name\n\
                               run -Q without parameter to get a list\n\
 -N benchmark  (--net)         get benchmark from NeuroJSON.io, -N only to list\n\
                               benchmark can be dataset URL,or dbname/benchname\n\
                               requires 'curl', install from https://curl.se/\n\
\n"S_BOLD S_CYAN"\
== MC options ==\n" S_RESET"\
 -n [0|int]    (--photon)      total photon number (exponential form accepted)\n\
                               max accepted value:9.2234e+18 on 64bit systems\n\
 -r [1|+/-int] (--repeat)      if positive, repeat by r times,total= #photon*r\n\
                               if negative, divide #photon into r subsets\n\
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit\n\
 -B '______'   (--bc)          per-face boundary condition (BC), 6 letters for\n\
    /case insensitive/         bounding box faces at -x,-y,-z,+x,+y,+z axes;\n\
                               overwrite -b if given. \n\
                               each letter can be one of the following:\n\
                               '_': undefined, fallback to -b\n\
                               'r': like -b 1, Fresnel reflection BC\n\
                               'a': like -b 0, total absorption BC\n\
                               'm': mirror or total reflection BC\n\
                               'c': cyclic BC, enter from opposite face\n\
\n\
                               if input contains additional 6 letters,\n\
                               the 7th-12th letters can be:\n\
                               '0': do not use this face to detect photon, or\n\
                               '1': use this face for photon detection (-d 1)\n\
                               the order of the faces for letters 7-12 is \n\
                               the same as the first 6 letters\n\
                               eg: --bc ______010 saves photons exiting at y=0\n\
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge\n\
 -U [1|0|2]    (--normalize)   1 to normalize flux to unitary; 0 save raw;\n\
                               2 behaves as 1, except in replay mode, where\n\
                               the normalization by the total detected weight\n\
                               is skipped and only the unit conversion applies\n\
 -E [1648335518|int|mch](--seed) set rand-number-generator seed, -1 to generate\n\
                               if an mch file is followed, MCX \"replays\" \n\
                               the detected photon; the replay mode can be used\n\
                               to calculate the mua/mus Jacobian matrices\n\
 -z [0|1]      (--srcfrom0)    1 volume origin is [0 0 0]; 0: origin at [1 1 1]\n\
 -k [1|0]      (--voidtime)    when src is outside, 1 enables timer inside void\n\
 -Y [0|int]    (--replaydet)   replay only the detected photons from a given \n\
                               detector (det ID starts from 1), used with -E \n\
                               if 0, replay all detectors and sum all Jacobians\n\
                               if -1, replay all detectors and save separately\n\
 -V [0|1]      (--specular)    1 source located in the background,0 inside mesh\n\
 -e [0.|float] (--minenergy)   minimum energy level to trigger Russian roulette\n\
 -g [1|int]    (--gategroup)   number of maximum time gates per run\n\
\n"S_BOLD S_CYAN"\
== GPU options ==\n" S_RESET"\
 -L            (--listgpu)     print GPU information only\n\
 -t [16384|int](--thread)      total thread number\n\
 -T [64|int]   (--blocksize)   thread number per block\n\
 -A [1|int]    (--autopilot)   1 let mcx decide thread/block size, 0 use -T/-t\n\
                               autopilot also dispatches photons dynamically,\n\
                               so runs with the same seed agree statistically\n\
                               but not bit-wise; use -A 0 to reproduce a run\n\
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto\n\
      or\n\
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)\n\
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum\n\
 -I            (--printgpu)    print GPU information and run program\n\
 --atomic [1|0]                deprecated and ignored; atomic accumulation is\n\
                               always used, non-atomic mode raced and lost data\n\
\n"S_BOLD S_CYAN"\
== Input options ==\n" S_RESET"\
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid.\n\
                               only the root object named 'Shapes' is parsed \n\
                               and added to the existing domain defined via -f \n\
                               or --bench\n\
 -j '{...}'    (--json)        a JSON string for modifying all input settings.\n\
                               this input can be used to modify all existing \n\
                               settings defined by -f or --bench\n\
 -K [1|int|str](--mediabyte)   volume data format, use either a number or a str\n\
       voxel binary data layouts are shown in {...}, where [] for byte,[i:]\n\
       for 4-byte integer, [s:] for 2-byte short, [h:] for 2-byte half float,\n\
       [f:] for 4-byte float; on Little-Endian systems, least-sig. bit on left\n\
                               1 or byte: 0-128 tissue labels\n\
                               2 or short: 0-65535 (max to 4000) tissue labels\n\
                               4 or integer: integer tissue labels \n\
                              96 or asgn_float: mua/mus/g/n 4xfloat format\n\
                                {[f:mua][f:mus][f:g][f:n]}\n\
                              97 or svmc: split-voxel MC 8-byte format\n\
                                {[n.z][n.y][n.x][p.z][p.y][p.x][upper][lower]}\n\
                              98 or mixlabel: label1+label2+label1_percentage\n\
                                {[label1][label2][s:0-32767 label1 percentage]}\n\
                              99 or labelplus: 32bit composite voxel format\n\
                                {[h:mua/mus/g/n][s:(B15-16:0/1/2/3)(label)]}\n\
                             100 or muamus_float: 2x 32bit floats for mua/mus\n\
                                {[f:mua][f:mus]}; g/n from medium type 1\n\
                             101 or mua_float: 1 float per voxel for mua\n\
                                {[f:mua]}; mus/g/n from medium type 1\n\
                             102 or muamus_half: 2x 16bit float for mua/mus\n\
                                {[h:mua][h:mus]}; g/n from medium type 1\n\
                             103 or asgn_byte: 4-byte gray-levels for mua/s/g/n\n\
                                {[mua][mus][g][n]}; 0-255 mixing prop types 1&2\n\
                             104 or muamus_short: 2-short gray-levels for mua/s\n\
                                {[s:mua][s:mus]}; 0-65535 mixing prop types 1&2\n\
       when formats 99 or 102 is used, the mua/mus values in the input volume\n\
       binary data must be pre-scaled by voxel size (unitinmm) if it is not 1.\n\
       pre-scaling is not needed when using these 2 formats in mcxlab/pmcx\n\
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array\n\
\n"S_BOLD S_CYAN"\
== Output options ==\n" S_RESET"\
 -s sessionid  (--session)     a string to label all output file names\n\
 -O [X|XFEJPMRLSTBADCUVWQ](--outputtype) X - output flux, F - fluence, E - energy\n\
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, \n\
                               event counts at each voxel (replay mode only)\n\
                               M - momentum transfer; L - total pathlength\n\
                               R - RF/FD mua Jacobian (replay, needs -E)\n\
                                   or RF forward complex fluence (forward,\n\
                                   without -E, needs omega); output is complex\n\
                               S - RF/FD mus Jacobian (replay mode only)\n\
                               T - time-of-flight*path;B - time-of-flight*nscat\n\
                               A - adjoint mua Jacobian (needs detpos+detdir)\n\
                               D - adjoint D-coeff Jacobian (needs detpos+dir)\n\
                               U - adjoint mus Jacobian (grad*grad*3*D^2*(1-g))\n\
                               V - adjoint mus' Jacobian (grad*grad*3*D^2)\n\
                               W - dual adjoint [J_mua,J_D] pair in 1 session\n\
                               Q - dual adjoint [J_mua,J_mus'] pair in 1 session\n\
 -d [1|0-3]    (--savedet)     1 to save photon info at detectors; 0 not save\n\
                               2 reserved, 3 terminate simulation when detected\n\
                               photon buffer is filled\n\
 -w [DP|DSPMXVW](--savedetflag)a string controlling detected photon data fields\n\
    /case insensitive/         1 D  output detector ID (1)\n\
                               2 S  output partial scat. even counts (#media)\n\
                               4 P  output partial path-lengths (#media)\n\
                               8 M  output momentum transfer (#media)\n\
                              16 X  output exit position (3)\n\
                              32 V  output exit direction (3)\n\
                              64 W  output initial weight (1)\n\
      combine multiple items by using a string,or add selected numbers together\n\
      by default, mcx only saves detector ID and partial-path data\n\
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions\n\
                               setting -x to 1 also implies setting '-d' to 1.\n\
                               same as adding 'XV' to -w.\n\
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels\n\
                               right outside of the domain; if non-zero voxels\n\
                               appear at the boundary, pad 0s before using -X\n\
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save\n\
                               same as adding 'M' to the -w flag\n\
 -q [0|1]      (--saveseed)    1 to save photon RNG seed for replay; 0 not save\n\
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save\n\
 -H [1000000] (--maxdetphoton) max number of detected photons\n\
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save\n\
 -F [jnii|...](--outputformat) fluence data output format:\n\
                               mc2 - MCX mc2 format (binary 32bit float)\n\
                               jnii - JNIfTI format (https://neurojson.org)\n\
                               bnii - Binary JNIfTI (https://neurojson.org)\n\
                               nii - NIfTI format\n\
                               hdr - Analyze 7.5 hdr/img format\n\
                               tx3 - GL texture data for rendering (GL_RGBA32F)\n\
    the bnii/jnii formats support compression (-Z) and generate small files\n\
    load jnii (JSON) and bnii (BJData) files using below lightweight libs:\n\
      MATLAB/Octave: JNIfTI toolbox   https://neurojson.org/download/jnifty\n\
      MATLAB/Octave: JSONLab toolbox  https://neurojson.org/download/jsonlab\n\
      Python:        PyJData:         https://neurojson.org/download/pyjdata\n\
      JavaScript:    JSData:          https://neurojson.org/download/jsdata\n\
 -Z [zlib|...] (--zip)         set compression method if -F jnii or --dumpjson\n\
                               is used (when saving data to JSON/JNIfTI format)\n\
                               0 zlib: zip format (moderate compression,fast) \n\
                               1 gzip: gzip format (compatible with *.gz)\n\
                               2 base64: base64 encoding with no compression\n\
                               3 lzip: lzip format (high compression,very slow)\n\
                               4 lzma: lzma format (high compression,very slow)\n\
                               5 lz4: LZ4 format (low compression,extrem. fast)\n\
                               6 lz4hc: LZ4HC format(moderate compression,fast)\n\
 --dumpjson [-,0,1,'file.json'] export all settings,including volume data using\n\
                               JSON/JData (https://neurojson.org) format for\n\
                               easy sharing; can be reused using -f\n\
                               if followed by nothing or '-', mcx will print\n\
                               the JSON to the console; write to a file if file\n\
                               name is specified; by default, prints settings\n\
                               after pre-processing; '--dumpjson 2' prints \n\
                               raw inputs before pre-processing\n\
\n"S_BOLD S_CYAN"\
== User IO options ==\n" S_RESET"\
 -h            (--help)        print this message\n\
 -y [zh_CN,..] (--lang)        select language, followed by nothing to print\n\
 -v            (--version)     print MCX revision number\n\
 -l            (--log)         print messages to a log file instead\n\
 -i            (--interactive) interactive mode\n\
\n"S_BOLD S_CYAN"\
== Debug options ==\n" S_RESET"\
 -D [0|int]    (--debug)       print debug information (you can use an integer\n\
  or                           or a string by combining the following flags)\n\
 -D [''|RMPT]                  1 R debug RNG\n\
    /case insensitive/         2 M store photon trajectory info\n\
                               4 P print progress bar\n\
                               8 T save trajectory data only, disable flux/detp\n\
      combine multiple items by using a string,or add selected numbers together\n\
\n"S_BOLD S_CYAN"\
== Additional options ==\n" S_RESET"\
 --root         [''|string]    full path to the folder storing the input files\n\
 --gscatter     [1e9|int]      after a photon completes the specified number of\n\
                               scattering events, mcx then ignores anisotropy g\n\
                               and only performs isotropic scattering for speed\n\
 --srcid  [0|-1,0,1,2,..]     -1 simulate multi-source separately;0 all sources\n\
                               together;a positive integer runs a single source\n\
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch\n\
 --trajstokes   [0|1]          set to 1 to save Stokes IQUV in trajectory data\n\
 --maxvoidstep  [1000|int]     max distance (in voxel unit) of a photon that\n\
                               can travel before entering the domain, if \n\
                               launched outside (i.e. a widefield source)\n\
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),\n\
                               use this parameter to set the maximum positions\n\
                               stored (default: 1e7)\n\
\n"S_BOLD S_CYAN"\
== Example ==\n" S_RESET"\
example: (list built-in benchmarks: -Q/--bench)\n"S_MAGENTA"\
       %s -Q\n" S_RESET"\
or (list supported GPUs on the system: -L/--listgpu)\n"S_MAGENTA"\
       %s -L\n" S_RESET"\
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)\n"S_MAGENTA"\
       %s -Q cube60b -n 1e7 -G 1101 -W 10,10,10\n" S_RESET"\
or (use inline domain definition)\n"S_MAGENTA"\
       %s -f input.json -P '{\"Shapes\":[{\"ZLayers\":[[1,10,1],[11,30,2],[31,60,3]]}]}'\n" S_RESET"\
or (use inline json setting modifier)\n"S_MAGENTA"\
       %s -f input.json -j '{\"Optode\":{\"Source\":{\"Type\":\"isotropic\"}}}'\n" S_RESET"\
or (dump simulation in a single json file)\n"S_MAGENTA"\
       %s -Q cube60planar --dumpjson\n" S_RESET"\
or (use -N/--net to browse community-contributed mcx simulations at https://neurojson.io)\n"S_MAGENTA"\
       %s -N\n" S_RESET"\
or (run user-shared mcx simulations, see full list at https://neurojson.org/db/mcx)\n"S_MAGENTA"\
       %s -N aircube60\n" S_RESET"\
or (print in simplified Chinese using -y/--lang)\n"S_MAGENTA"\
       %s -y zh_CN -Q cube60\n" S_RESET"\
or (use -f - to read piped input file modified by shell text processing utilities)\n"S_MAGENTA"\
       %s -Q cube60 --dumpjson | sed -e 's/pencil/cone/g' | %s -f -\n" S_RESET"\
or (download/modify simulations from NeuroJSON.io and run with mcx -f)\n"S_MAGENTA"\
       curl -s -X GET https://neurojson.io:7777/mcx/aircube60 | jq '.Forward.Dt = 1e-9' | %s -f" S_RESET"\n",
           exename, exename, exename, exename, exename, exename, exename, exename, exename, exename, exename, exename, exename);
}
