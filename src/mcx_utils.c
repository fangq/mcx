/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2023
**
**  \section sref Reference:
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

#ifndef WIN32
    #include <sys/ioctl.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include "mcx_utils.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include "mcx_core.h"
#include "mcx_bench.h"
#include "mcx_mie.h"

#ifndef MCX_CONTAINER
    #include "zmat/zmatlib.h"
    #include "ubj/ubj.h"
#endif

/**
 * Macro to load JSON keys
 */
#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
     ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? fallback : tmp->val) \
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
#define MCX_ASSERT(a)  (!(a) && (mcx_error((a),"input error",__FILE__,__LINE__),1) );

#define MIN_HEADER_SIZE 348    /**< Analyze header size */
#define NII_HEADER_SIZE 352    /**< NIFTI header size */
#define GL_RGBA32F 0x8814

/**
 * Short command line options
 * If a short command line option is '-' that means it only has long/verbose option.
 * Array terminates with '\0'.
 * Currently un-used options: cCJNoQy0-9
 */

const char shortopt[] = {'h', 'i', 'f', 'n', 't', 'T', 's', 'a', 'g', 'b', '-', 'z', 'u', 'H', 'P',
                         'd', 'r', 'S', 'p', 'e', 'U', 'R', 'l', 'L', '-', 'I', '-', 'G', 'M', 'A', 'E', 'v', 'D',
                         'k', 'q', 'Y', 'O', 'F', '-', '-', 'x', 'X', '-', 'K', 'm', 'V', 'B', 'W', 'w', '-',
                         '-', '-', 'Z', 'j', '-', '\0'
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
                         "--internalsrc", "--bench", "--dumpjson", "--zip", "--json", "--atomic", ""
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
 */

const char outputtype[] = {'x', 'f', 'e', 'j', 'p', 'm', 'r', 'l', '\0'};

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
 * c: cylic BC
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

const unsigned int mediaformatid[] = {1, 2, 4, 97, 98, 99, 100, 101, 102, 103, 104, 0};
const char* mediaformat[] = {"byte", "short", "integer", "svmc", "mixlabel", "labelplus",
                             "muamus_float", "mua_float", "muamus_half", "asgn_byte", "muamus_short", ""
                            };

/**
 * Flag to decide if parameter has been initialized over command line
 */

char flagset[256] = {'\0'};

const char* zipformat[] = {"zlib", "gzip", "base64", "lzip", "lzma", "lz4", "lz4hc", ""};

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
    memset(cfg->jsonfile, 0, MAX_PATH_LENGTH);
    memset(cfg->bc, 0, 12);
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
        short* mask = (short*)logval;

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

void mcx_savebnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, Config* cfg) {
    FILE* fp;
    char fname[MAX_FULL_PATH] = {'\0'};
    int affine[] = {0, 0, 1, 0, 0, 0};
    size_t datalen = sizeof(int), outputlen = 0;

    ubjw_context_t* root = NULL;
    uchar* jsonstr = NULL;

    for (int i = 0; i < ndim; i++) {
        datalen *= dims[i];
    }

    jsonstr = malloc(datalen << 1);
    root = ubjw_open_memory(jsonstr, jsonstr + (datalen << 1));

    ubjw_begin_object(root, UBJ_MIXED, 0);
    /* the "_DataInfo_" section */
    ubjw_write_key(root, "_DataInfo_");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    UBJ_WRITE_KEY(root, "JNIFTIVersion", string, "0.5");
    UBJ_WRITE_KEY(root, "Comment", string, "Created by MCX (http://mcx.space)");
    UBJ_WRITE_KEY(root, "AnnotationFormat", string, "https://neurojson.org/jnifti/draft1");
    UBJ_WRITE_KEY(root, "SerialFormat", string, "https://neurojson.org/bjdata/draft2");
    ubjw_write_key(root, "Parser");
    ubjw_begin_object(root, UBJ_MIXED, 0);
    ubjw_write_key(root, "Python");
    ubjw_begin_array(root, UBJ_STRING, 2);
    ubjw_write_string(root, "https://pypi.org/project/jdata");
    ubjw_write_string(root, "https://pypi.org/project/bjdata");
    ubjw_end(root);
    ubjw_write_key(root, "MATLAB");
    ubjw_begin_array(root, UBJ_STRING, 2);
    ubjw_write_string(root, "https://github.com/NeuroJSON/jnifty");
    ubjw_write_string(root, "https://github.com/NeuroJSON/jsonlab");
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

    if (cfg->outputtype >= 0) {
        const char* typestr[] = {"MCX volumetric output: Fluence rate (W/mm^2)", "MCX volumetric output: Fluence (J/mm^2)",
                                 "MCX volumetric output: Voxel-wise energy deposit (J)", "MCX volumetric output: Jacobian for mua (J/mm)", "MCX volumetric output: Scattering count",
                                 "MCX volumetric output: Partial momentum transfer"
                                };
        UBJ_WRITE_KEY(root, "Description", string, typestr[(int)cfg->outputtype]);
    } else {
        UBJ_WRITE_KEY(root, "Description", string, "MCX volumetric output");
    }

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

    if (mcx_jdataencode(vol, ndim, dims, (isfloat ? "single" : "uint32"), 4, cfg->zipid, root, 1, cfg)) {
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

void mcx_savejnii(float* vol, int ndim, uint* dims, float* voxelsize, char* name, int isfloat, Config* cfg) {
    FILE* fp;
    char fname[MAX_FULL_PATH] = {'\0'};
    int affine[] = {0, 0, 1, 0, 0, 0};
    const char* libpy[] = {"https://pypi.org/project/jdata", "https://pypi.org/project/bjdata"};
    const char* libmat[] = {"https://github.com/NeuroJSON/jnifty", "https://github.com/NeuroJSON/jsonlab"};
    const char* libjs[] = {"https://www.npmjs.com/package/jda", "https://www.npmjs.com/package/bjd"};
    const char* libc[]  = {"https://github.com/DaveGamble/cJSON", "https://github.com/NeuroJSON/ubj"};

    cJSON* root = NULL, *hdr = NULL, *dat = NULL, *sub = NULL, *info = NULL, *parser = NULL;
    char* jsonstr = NULL;
    root = cJSON_CreateObject();

    /* the "_DataInfo_" section */
    cJSON_AddItemToObject(root, "_DataInfo_", info = cJSON_CreateObject());
    cJSON_AddStringToObject(info, "JNIFTIVersion", "0.5");
    cJSON_AddStringToObject(info, "Comment", "Created by MCX (http://mcx.space)");
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

    if (cfg->outputtype >= 0) {
        const char* typestr[] = {"MCX volumetric output: Fluence rate (W/mm^2)", "MCX volumetric output: Fluence (J/mm^2)",
                                 "MCX volumetric output: Voxel-wise energy deposit (J)", "MCX volumetric output: Jacobian for mua (J/mm)", "MCX volumetric output: Scattering count",
                                 "MCX volumetric output: Partial momentum transfer"
                                };
        cJSON_AddStringToObject(hdr, "Description", typestr[(int)cfg->outputtype]);
    } else {
        cJSON_AddStringToObject(hdr, "Description", "MCX volumetric output");
    }

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

    if (mcx_jdataencode(vol, ndim, dims, (isfloat ? "single" : "uint32"), 4, cfg->zipid, dat, 0, cfg)) {
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
        mcx_savenii(dat, len * (1 + (cfg->outputtype == otRF)), name, NIFTI_TYPE_FLOAT32, cfg->outputformat, cfg);
        return;
    } else if (cfg->outputformat == ofJNifti || cfg->outputformat == ofBJNifti) {
        int d1 = (cfg->maxgate == 1);

        if (cfg->seed == SEED_FROM_FILE && ((cfg->replaydet == -1 && cfg->detnum > 1) || cfg->srcnum > 1 || cfg->outputtype == otRF)) {
            if (cfg->outputtype == otRF) {
                uint dims[6] = {2, cfg->detnum* cfg->srcnum, cfg->maxgate, cfg->dim.z, cfg->dim.y, cfg->dim.x};
                float voxelsize[] = {1, 1, cfg->tstep, cfg->steps.z, cfg->steps.y, cfg->steps.x};

                if (cfg->outputformat == ofJNifti) {
                    mcx_savejnii(dat, 6, dims, voxelsize, name, 1, cfg);
                } else {
                    mcx_savebnii(dat, 6, dims, voxelsize, name, 1, cfg);
                }
            } else {
                uint dims[5] = {cfg->detnum* cfg->srcnum, cfg->maxgate, cfg->dim.z, cfg->dim.y, cfg->dim.x};
                float voxelsize[] = {1, cfg->tstep, cfg->steps.z, cfg->steps.y, cfg->steps.x};

                if (cfg->outputformat == ofJNifti) {
                    mcx_savejnii(dat, 5, dims, voxelsize, name, 1, cfg);
                } else {
                    mcx_savebnii(dat, 5, dims, voxelsize, name, 1, cfg);
                }
            }
        } else {
            uint dims[] = {cfg->dim.x, cfg->dim.y, cfg->dim.z, cfg->maxgate};
            float voxelsize[] = {cfg->steps.x, cfg->steps.y, cfg->steps.z, cfg->tstep};
            size_t datalen = cfg->dim.x * cfg->dim.y * cfg->dim.z * cfg->maxgate;
            uint* buf = (uint*)malloc(datalen * sizeof(float));
            memcpy(buf, dat, datalen * sizeof(float));

            if (d1) {
                mcx_convertcol2row(&buf, (uint3*)dims);
            } else {
                mcx_convertcol2row4d(&buf, (uint4*)dims);
            }

            if (cfg->outputformat == ofJNifti) {
                mcx_savejnii((float*)buf, 4 - d1, dims, voxelsize, name, 1, cfg);
            } else {
                mcx_savebnii((float*)buf, 4 - d1, dims, voxelsize, name, 1, cfg);
            }

            free(buf);
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

    fwrite(dat, sizeof(float), len * (1 + (cfg->outputtype == otRF)), fp);
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

    if (cfg->outputformat == ofJNifti || cfg->outputformat == ofBJNifti) {
        mcx_savejdet(ppath, seeds, count, doappend, cfg);
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
    cJSON_AddItemToObject(hdr, "Media", sub = cJSON_CreateArray());

    for (int i = 0; i < cfg->medianum; i++) {
        cJSON_AddItemToArray(sub, dat = cJSON_CreateObject());
        cJSON_AddNumberToObject(dat, "mua", cfg->prop[i].mua / cfg->unitinmm);
        cJSON_AddNumberToObject(dat, "mus", cfg->prop[i].mus / cfg->unitinmm);
        cJSON_AddNumberToObject(dat, "g",   cfg->prop[i].g);
        cJSON_AddNumberToObject(dat, "n",   cfg->prop[i].n);
    }

    if (cfg->his.detected == 0  && cfg->his.savedphoton) {
        char colnum[] = {1, 3, 1};
        char* dtype[] = {"uint32", "single", "single"};
        char* dname[] = {"photonid", "p", "w0"};
        cJSON_AddItemToObject(obj, "Trajectory", dat = cJSON_CreateObject());

        for (int id = 0; id < sizeof(colnum); id++) {
            uint dims[2] = {count, colnum[id]};
            float* buf = (float*)calloc(dims[0] * dims[1], sizeof(float));

            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++) {
                    buf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
                }

            cJSON_AddItemToObject(dat, dname[id], sub = cJSON_CreateObject());

            if (mcx_jdataencode(buf, 2, dims, dtype[id], 4, cfg->zipid, sub, 0, cfg)) {
                MCX_ERROR(-1, "error when converting to JSON");
            }

            free(buf);
            col += dims[1];
        }
    } else {
        char colnum[] = {1, cfg->his.maxmedia, cfg->his.maxmedia, cfg->his.maxmedia, 3, 3, 1};
        char* dtype[] = {"uint32", "uint32", "single", "single", "single", "single", "single"};
        char* dname[] = {"detid", "nscat", "ppath", "mom", "p", "v", "w0"};
        cJSON_AddItemToObject(obj, "PhotonData", dat = cJSON_CreateObject());

        for (int id = 0; id < sizeof(colnum); id++) {
            if ((cfg->savedetflag >> id) & 0x1) {
                uint dims[2] = {count, colnum[id]};
                void* val = NULL;
                float* fbuf = NULL;
                uint*  ibuf = NULL;

                if (!strcmp(dtype[id], "uint32")) {
                    ibuf = (uint*)calloc(dims[0] * dims[1], sizeof(uint));

                    for (int i = 0; i < dims[0]; i++)
                        for (int j = 0; j < dims[1]; j++) {
                            ibuf[i * dims[1] + j] = ppath[i * cfg->his.colcount + col + j];
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

                cJSON_AddItemToObject(dat, dname[id], sub = cJSON_CreateObject());

                if (mcx_jdataencode(val, 2, dims, dtype[id], 4, cfg->zipid, sub, 0, cfg)) {
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

        if (mcx_jdataencode(seeds, 2, dims, "uint8", 1, cfg->zipid, sub, 0, cfg)) {
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
        sprintf(fhistory, "%s%c%s_%s.jdat", cfg->rootpath, pathsep, cfg->session, (filetag == 't' ? "traj" : "detp"));
    } else {
        sprintf(fhistory, "%s_%s.jdat", cfg->session, (filetag == 't' ? "traj" : "detp"));
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

#endif

/**
 * @brief Print a message to the console or a log file
 *
 * @param[in] cfg: simulation configuration
 * @param[in] str: a string to be printed
 */

void mcx_printlog(Config* cfg, char* str) {
    if (cfg->flog > 0) { /*stdout is 1*/
        MCX_FPRINTF(cfg->flog, "%s\n", str);
    }
}

/**
 * @brief Normalize the solution by multiplying a scaling factor
 *
 * @param[in,out] field: volumetric data before normalization
 * @param[in] scale: the scaling factor (or normalization factor) to be applied
 * @param[in] fieldlen: the length (floating point) of elements in the volume
 * @param[in] option: if set to 2, only normalize positive values (negative values for diffuse reflectance calculations)
 */

void mcx_normalize(float field[], float scale, int fieldlen, int option, int pidx, int srcnum) {
    int i;

    for (i = 0; i < fieldlen; i++) {
        if (option == 2 && field[i * srcnum + pidx] < 0.f) {
            continue;
        }

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
URL: http://mcx.space/wiki/index.cgi?Doc/FAQ\n" S_RESET);
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

    if (dir_path[strlen(p) - 1] == pathsep) {
        dir_path[strlen(p) - 1] = '\0';
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
        MCX_ERROR(ret, "assert error");
    }
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

            jroot = cJSON_Parse(jbuf);

            if (jroot) {
                mcx_loadjson(jroot, cfg);
                cJSON_Delete(jroot);
            } else {
                char* ptrold, *ptr = (char*)cJSON_GetErrorPtr();

                if (ptr) {
                    ptrold = strstr(jbuf, ptr);
                }

                if (fp != NULL) {
                    fclose(fp);
                }

                if (ptr && ptrold) {
                    char* offs = (ptrold - jbuf >= 50) ? ptrold - 50 : jbuf;

                    while (offs < ptrold) {
                        MCX_FPRINTF(stderr, "%c", *offs);
                        offs++;
                    }

                    MCX_FPRINTF(stderr, "<error>%.50s\n", ptrold);
                }

                if (fp != NULL) {
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
            strncpy(cfg->session, fname, MAX_SESSION_LENGTH);
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
            mcx_convertrow2col64((size_t**) & (cfg->vol), &(cfg->dim));
        } else {
            mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
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

    if (cfg->mediabyte >= 100 && cfg->savedetflag) {
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
    if (cfg->isdumpjson == 2) {
        mcx_savejdata(cfg->jsonfile, cfg);
        exit(0);
    }

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
    } else {
        MCX_ERROR(-4, "one must specify a binary volume file in order to run the simulation");
    }

    if (cfg->seed == SEED_FROM_FILE && cfg->seedfile[0]) {
        if (strstr(cfg->seedfile, ".jdat") != NULL) {
            mcx_loadseedjdat(cfg->seedfile, cfg);
        } else {
            mcx_loadseedfile(cfg);
        }
    }

    mcx_preprocess(cfg);

    if (cfg->isdumpjson == 3) {
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
        MCX_ASSERT(fscanf(in, "%d", &itmp ) == 1);
    }

    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%d\nPlease specify the position of the source (in grid unit): [10 10 5]\n\t", cfg->seed);
    }

    MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->srcpos.x), &(cfg->srcpos.y), &(cfg->srcpos.z) ) == 3);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (cfg->issrcfrom0 == 0 && comm != NULL && sscanf(comm, "%d", &itmp) == 1) {
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

    if (cfg->tstart > cfg->tend || cfg->tstep == 0.f) {
        MCX_ERROR(-9, "incorrect time gate settings");
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
        strncpy(filename, comment, MAX_FULL_PATH);
    }

    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t", filename);
    }

    MCX_ASSERT(fscanf(in, "%f %d %d %d", &(cfg->steps.x), &(cfg->dim.x), &(cfg->crop0.x), &(cfg->crop1.x)) == 4);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin)
        fprintf(stdout, "%f %d %d %d\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                cfg->steps.x, cfg->dim.x, cfg->crop0.x, cfg->crop1.x);

    MCX_ASSERT(fscanf(in, "%f %d %d %d", &(cfg->steps.y), &(cfg->dim.y), &(cfg->crop0.y), &(cfg->crop1.y)) == 4);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin)
        fprintf(stdout, "%f %d %d %d\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                cfg->steps.y, cfg->dim.y, cfg->crop0.y, cfg->crop1.y);

    MCX_ASSERT(fscanf(in, "%f %d %d %d", &(cfg->steps.z), &(cfg->dim.z), &(cfg->crop0.z), &(cfg->crop1.z)) == 4);
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

    MCX_ASSERT(fscanf(in, "%d", &(cfg->medianum)) == 1);
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

    MCX_ASSERT(fscanf(in, "%d %f", &(cfg->detnum), &(cfg->detradius)) == 2);
    comm = fgets(comment, MAX_PATH_LENGTH, in);

    if (in == stdin) {
        fprintf(stdout, "%d %f\n", cfg->detnum, cfg->detradius);
    }

    if (cfg->medianum + cfg->detnum > MAX_PROP_AND_DETECTORS) {
        MCX_ERROR(-4, "input media types plus detector number exceeds the maximum total (4000)");
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
    char filename[MAX_FULL_PATH] = {'\0'};
    Domain  = cJSON_GetObjectItem(root, "Domain");
    Optode  = cJSON_GetObjectItem(root, "Optode");
    Session = cJSON_GetObjectItem(root, "Session");
    Forward = cJSON_GetObjectItem(root, "Forward");
    Shapes  = cJSON_GetObjectItem(root, "Shapes");

    if (Domain) {
        char volfile[MAX_PATH_LENGTH];
        cJSON* meds, *val, *vv;
        val = FIND_JSON_OBJ("VolumeFile", "Domain.VolumeFile", Domain);

        if (val) {
            strncpy(volfile, val->valuestring, MAX_PATH_LENGTH);

            if (cfg->rootpath[0]) {
#ifdef WIN32
                sprintf(filename, "%s\\%s", cfg->rootpath, volfile);
#else
                sprintf(filename, "%s/%s", cfg->rootpath, volfile);
#endif
            } else {
                strncpy(filename, volfile, MAX_PATH_LENGTH);
            }
        }

        val = FIND_JSON_OBJ("MediaFormat", "Domain.MediaFormat", Domain);

        if (val) {
            cfg->mediabyte = mcx_keylookup((char*)FIND_JSON_KEY("MediaFormat", "Domain.MediaFormat", Domain, "byte", valuestring), mediaformat);

            if (cfg->mediabyte == -1) {
                MCX_ERROR(-1, "Unsupported media format.");
            }

            cfg->mediabyte = mediaformatid[cfg->mediabyte];
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
            cfg->nphase += (cfg->nphase & 0x1); /* make cfg.nphase even number */

            if (cfg->invcdf) {
                free(cfg->invcdf);
            }

            cfg->invcdf = (float*)calloc(cfg->nphase, sizeof(float));
            cfg->invcdf[0] = -1.f; /*left end is always -1.f,right-end is always 1.f*/
            vv = val->child;

            for (i = 1; i <= nphase; i++) {
                cfg->invcdf[i] = vv->valuedouble;
                vv = vv->next;

                if (cfg->invcdf[i] < cfg->invcdf[i - 1] || (cfg->invcdf[i] > 1.f || cfg->invcdf[i] < -1.f)) {
                    MCX_ERROR(-1, "Domain.InverseCDF contains invalid data; it must be a monotonically increasing vector with all values between -1 and 1");
                }
            }

            cfg->invcdf[nphase + 1] = 1.f; /*left end is always -1.f,right-end is always 1.f*/
            cfg->invcdf[cfg->nphase - 1] = 1.f;
        }

        val = FIND_JSON_OBJ("VoxelSize", "Domain.VoxelSize", Domain);

        if (val) {
            val = FIND_JSON_OBJ("Dx", "Domain.VoxelSize.Dx", Domain);

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

            val = FIND_JSON_OBJ("Dy", "Domain.VoxelSize.Dy", Domain);

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

            val = FIND_JSON_OBJ("Dz", "Domain.VoxelSize.Dz", Domain);

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
    }

    if (Optode) {
        cJSON* dets, *src = FIND_JSON_OBJ("Source", "Optode.Source", Optode);

        if (src) {
            subitem = FIND_JSON_OBJ("Pos", "Optode.Source.Pos", src);

            if (subitem) {
                cfg->srcpos.x = subitem->child->valuedouble;
                cfg->srcpos.y = subitem->child->next->valuedouble;
                cfg->srcpos.z = subitem->child->next->next->valuedouble;
            }

            cfg->srcpos.w = FIND_JSON_KEY("Weight", "Optode.Source.Weight", src, 1.f, valuedouble);

            subitem = FIND_JSON_OBJ("Dir", "Optode.Source.Dir", src);

            if (subitem) {
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
            }

            subitem = FIND_JSON_OBJ("IQUV", "Optode.Source.IQUV", src);

            if (subitem) {
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

            cfg->srctype = mcx_keylookup((char*)FIND_JSON_KEY("Type", "Optode.Source.Type", src, "pencil", valuestring), srctypeid);
            subitem = FIND_JSON_OBJ("Param1", "Optode.Source.Param1", src);

            if (subitem) {
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
            }

            subitem = FIND_JSON_OBJ("Param2", "Optode.Source.Param2", src);

            if (subitem) {
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
            }

            cfg->omega = FIND_JSON_KEY("Frequency", "Optode.Source.Frequency", src, 0.f, valuedouble);
            cfg->omega *= TWO_PI;
            cfg->lambda = FIND_JSON_KEY("WaveLength", "Optode.Source.WaveLength", src, 0.f, valuedouble);
            cfg->srcnum = FIND_JSON_KEY("SrcNum", "Optode.Source.SrcNum", src, cfg->srcnum, valueint);
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

                    if (strcmp(type, "single")) {
                        if (cfg->srcpattern) {
                            free(cfg->srcpattern);
                        }

                        MCX_ERROR(-1, "Optode.Source.Pattern JData-annotated array must be in the 'single' format");
                    }

                    if (ndim == 3 && dims[2] > 1 && dims[0] > 1 && cfg->srctype == MCX_SRC_PATTERN) {
                        cfg->srcnum = dims[0];
                    }
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
        }

        dets = FIND_JSON_OBJ("Detector", "Optode.Detector", Optode);

        if (dets) {
            cJSON* det = dets->child;

            if (det) {
                cfg->detnum = cJSON_GetArraySize(dets);
                cfg->detpos = (float4*)malloc(sizeof(float4) * cfg->detnum);

                for (i = 0; i < cfg->detnum; i++) {
                    cJSON* pos = dets, *rad = NULL;
                    rad = FIND_JSON_OBJ("R", "Optode.Detector.R", det);

                    if (cJSON_GetArraySize(det) == 2) {
                        pos = FIND_JSON_OBJ("Pos", "Optode.Detector.Pos", det);
                    }

                    if (pos) {
                        cfg->detpos[i].x = pos->child->valuedouble;
                        cfg->detpos[i].y = pos->child->next->valuedouble;
                        cfg->detpos[i].z = pos->child->next->next->valuedouble;
                    }

                    if (rad) {
                        cfg->detpos[i].w = rad->valuedouble;
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

    if (cfg->medianum + cfg->detnum > MAX_PROP_AND_DETECTORS) {
        MCX_ERROR(-4, "input media types plus detector number exceeds the maximum total (4000)");
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
            strncpy(cfg->session, FIND_JSON_KEY("ID", "Session.ID", Session, "default", valuestring), MAX_SESSION_LENGTH);
        }

        if (cfg->rootpath[0] == '\0') {
            strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath", "Session.RootPath", Session, "", valuestring), MAX_PATH_LENGTH);
        }

        if (!flagset['B']) {
            strncpy(cfg->bc, FIND_JSON_KEY("BCFlags", "Session.BCFlags", Session, cfg->bc, valuestring), 12);
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

        if (cfg->outputformat == ofJNifti) {
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
        cfg->tstart = FIND_JSON_KEY("T0", "Forward.T0", Forward, 0.0, valuedouble);
        cfg->tend  = FIND_JSON_KEY("T1", "Forward.T1", Forward, 0.0, valuedouble);
        cfg->tstep = FIND_JSON_KEY("Dt", "Forward.Dt", Forward, 0.0, valuedouble);

        if (cfg->tstart > cfg->tend || cfg->tstep == 0.f) {
            MCX_ERROR(-9, "incorrect time gate settings");
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

            if (FIND_JSON_OBJ("_ArrayZipData_", "Volume._ArrayZipData_", Shapes)) {
                int ndim;
                char* type = NULL, *buf = NULL;

                if (mcx_jdatadecode((void**)&buf, &ndim, &(cfg->dim.x), 3, &type, Shapes, cfg) == 0) {
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
            }
        } else if (!(cfg->extrajson && cfg->extrajson[0] == '_')) {
            MCX_ERROR(-1, "You must either define Domain.VolumeFile, or define a Shapes section");
        }
    } else if (Shapes) {
        MCX_ERROR(-1, "You can not specify both Domain.VolumeFile and Shapes sections");
    }

    mcx_prepdomain(filename, cfg);
    cfg->his.maxmedia = cfg->medianum - 1; /*skip media 0*/
    cfg->his.detnum = cfg->detnum;
    cfg->his.srcnum = cfg->srcnum;
    cfg->his.savedetflag = cfg->savedetflag;
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

    cJSON_AddItemToObject(obj, "Dim", cJSON_CreateIntArray((int*) & (cfg->dim.x), 3));
    cJSON_AddNumberToObject(obj, "OriginType", 1);

    /* the "Optode" section */
    cJSON_AddItemToObject(root, "Optode", obj = cJSON_CreateObject());
    cJSON_AddItemToObject(obj, "Source", sub = cJSON_CreateObject());

    if (cfg->srctype >= 0) {
        cJSON_AddStringToObject(sub, "Type", srctypeid[(int)cfg->srctype]);
    }

    cJSON_AddItemToObject(sub, "Pos", cJSON_CreateFloatArray(&(cfg->srcpos.x), 3));
    cJSON_AddItemToObject(sub, "Dir", cJSON_CreateFloatArray(&(cfg->srcdir.x), 4));
    cJSON_AddItemToObject(sub, "Param1", cJSON_CreateFloatArray(&(cfg->srcparam1.x), 4));
    cJSON_AddItemToObject(sub, "Param2", cJSON_CreateFloatArray(&(cfg->srcparam2.x), 4));
    cJSON_AddNumberToObject(sub, "SrcNum", cfg->srcnum);

    if (cfg->srcpattern) {
        uint dims[3];
        dims[0] = cfg->srcnum;
        dims[1] = cfg->srcparam1.w;
        dims[2] = cfg->srcparam2.w;
        cJSON_AddItemToObject(sub, "Pattern", tmp = cJSON_CreateObject());

        int ret = mcx_jdataencode(cfg->srcpattern, 2 + (cfg->srcnum > 1), dims + (cfg->srcnum == 1), "single", dims[0] * dims[1] * dims[2], cfg->zipid, tmp, 0, cfg);

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

    /* save "Shapes" constructs, prioritize over saving volume for smaller size */
    if (cfg->shapedata) {
        cJSON* shape = cJSON_Parse(cfg->shapedata), *sp;

        if (shape == NULL) {
            MCX_ERROR(-1, "the input shape construct is not a valid JSON object");
        }

        sp = FIND_JSON_OBJ("Shapes", "Shapes", shape);

        if (sp == NULL) {
            sp = shape;
        }

        cJSON_AddItemToObject(root, "Shapes", sp);
    } else { /* if shape info is not available, save volume to JData constructs */
        uint datalen = cfg->dim.x * cfg->dim.y * cfg->dim.z;
        size_t outputbytes = datalen * sizeof(int);
        uchar* buf = (uchar*)calloc(datalen, sizeof(int));
        uint*  vol;
        int ret = 0;

        /*converting volume to the minimal size*/
        memcpy(buf, cfg->vol, datalen * sizeof(int));
        mcx_convertcol2row((uint**)(&buf), &cfg->dim);
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
        } else {
            for (int i = 0; i < datalen; i++) {
                vol[i] = vol[i] & MED_MASK;
            }
        }

        obj = cJSON_CreateObject();
        ret = mcx_jdataencode(buf, 3, &cfg->dim.x, (cfg->mediabyte == 1 ? "uint8" : (cfg->mediabyte == 2 ? "uint16" : "uint32")),
                              outputbytes / datalen, cfg->zipid, obj, 0, cfg);

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

    datalen = cfg->dim.x * cfg->dim.y * cfg->dim.z;
    cfg->vol = (unsigned int*)malloc(sizeof(unsigned int) * datalen * (1 + (cfg->mediabyte == MEDIA_2LABEL_SPLIT)));

    if (!isbuf) {
        if (cfg->mediabyte == MEDIA_AS_F2H) {
            inputvol = (unsigned char*)malloc(sizeof(unsigned char) * (datalen << 3));
        } else if (cfg->mediabyte >= 4) {
            inputvol = (unsigned char*)(cfg->vol);
        } else {
            inputvol = (unsigned char*)malloc(sizeof(unsigned char) * cfg->mediabyte * datalen);
        }

        res = fread(inputvol, sizeof(unsigned char) * ((cfg->mediabyte == MEDIA_AS_F2H || cfg->mediabyte == MEDIA_2LABEL_SPLIT) ? 8 : MIN(cfg->mediabyte, 4)), datalen, fp);
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
    } else if (cfg->mediabyte == MEDIA_AS_F2H) {
        float* val = (float*)inputvol;
        union {
            float f[2];
            unsigned int i[2];
            unsigned short h[2];
        } f2h;
        unsigned short tmp, m;

        for (i = 0; i < datalen; i++) {
            f2h.f[0] = val[i << 1] * cfg->unitinmm;
            f2h.f[1] = val[(i << 1) + 1] * cfg->unitinmm;

            if (f2h.f[0] != f2h.f[0] || f2h.f[1] != f2h.f[1]) { /*if one of mua/mus is nan in continuous medium, convert to 0-voxel*/
                cfg->vol[i] = 0;
                continue;
            }

            /**
            float to half conversion
            https://stackoverflow.com/questions/3026441/float32-to-float16/5587983#5587983
            https://gamedev.stackexchange.com/a/17410  (for denorms)
                */
            m = ((f2h.i[0] >> 13) & 0x03ff);
            tmp = (f2h.i[0] >> 23) & 0xff; /*exponent*/
            tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);

            if (m < 0x10 && tmp == 0) { /*handle denorms - between 2^-24 and 2^-14*/
                unsigned short sign = (f2h.i[0] >> 16) & 0x8000;
                tmp = ((f2h.i[0] >> 23) & 0xff);
                m = (f2h.i[0] >> 12) & 0x07ff;
                m |= 0x0800u;
                f2h.h[0] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
            } else {
                f2h.h[0] = (f2h.i[0] >> 31) << 5;
                f2h.h[0] = (f2h.h[0] | tmp) << 10;
                f2h.h[0] |= (f2h.i[0] >> 13) & 0x3ff;
            }

            m = ((f2h.i[1] >> 13) & 0x03ff);
            tmp = (f2h.i[1] >> 23) & 0xff; /*exponent*/
            tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);

            if (m < 0x10 && tmp == 0) { /*handle denorms - between 2^-24 and 2^-14*/
                unsigned short sign = (f2h.i[1] >> 16) & 0x8000;
                tmp = ((f2h.i[1] >> 23) & 0xff);
                m = (f2h.i[1] >> 12) & 0x07ff;
                m |= 0x0800u;
                f2h.h[1] = sign | ((m >> (114 - tmp)) + ((m >> (113 - tmp)) & 1));
            } else {
                f2h.h[1] = (f2h.i[1] >> 31) << 5;
                f2h.h[1] = (f2h.h[1] | tmp) << 10;
                f2h.h[1] |= (f2h.i[1] >> 13) & 0x3ff;
            }

            if (f2h.i[0] == 0) { /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16*/
                f2h.i[0] = 0x00010000;
            }

            if (f2h.i[0] == SIGN_BIT) { /*avoid being detected as a 0-label voxel, setting mus=EPS_fp16*/
                f2h.i[0] = 0;
            }

            cfg->vol[i] = f2h.i[0];
        }
    } else if (cfg->mediabyte == MEDIA_2LABEL_SPLIT) {
        memcpy(cfg->vol, inputvol, (datalen << 3));
    }

    int medianum = MAX(cfg->medianum, cfg->polmedianum + 1);

    if (cfg->mediabyte <= 4)
        for (i = 0; i < datalen; i++) {
            if (cfg->vol[i] >= medianum) {
                MCX_ERROR(-6, "medium index exceeds the specified medium types");
            }
        }

    if (!isbuf && (cfg->mediabyte < 4 || cfg->mediabyte == MEDIA_AS_F2H)) {
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
        if (cfg->replaydet <= 0 || cfg->replaydet == (int) (detps[i * dimdetps[0]])) {
            if (i != cfg->nphoton)
                memcpy((char*) (cfg->replay.seed) + cfg->nphoton * seedbyte,
                       (char*) (cfg->replay.seed) + i * seedbyte,
                       seedbyte);

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
        if (cfg->replaydet <= 0 || (detid && cfg->replaydet == detid[i])) {
            if (i != cfg->nphoton) {
                memcpy((char*)(cfg->replay.seed) + cfg->nphoton * his->seedbyte, (char*)(cfg->replay.seed) + i * his->seedbyte, his->seedbyte);
            }

            cfg->replay.weight[cfg->nphoton] = 1.f;
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

    if (!cfg->issrcfrom0) {
        cfg->srcpos.x--;
        cfg->srcpos.y--;
        cfg->srcpos.z--; /*convert to C index, grid center*/
    }

    if (cfg->tstart > cfg->tend || cfg->tstep == 0.f) {
        MCX_ERROR(-6, "incorrect time gate settings");
    }

    if (ABS(cfg->srcdir.x * cfg->srcdir.x + cfg->srcdir.y * cfg->srcdir.y + cfg->srcdir.z * cfg->srcdir.z - 1.f) > 1e-5) {
        MCX_ERROR(-6, "field 'srcdir' must be a unitary vector");
    }

    if (cfg->steps.x == 0.f || cfg->steps.y == 0.f || cfg->steps.z == 0.f) {
        MCX_ERROR(-6, "field 'steps' can not have zero elements");
    }

    if (cfg->tend <= cfg->tstart) {
        MCX_ERROR(-6, "field 'tend' must be greater than field 'tstart'");
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

    if ((cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otRF)
            && cfg->seed != SEED_FROM_FILE) {
        MCX_ERROR(-6, "Jacobian output is only valid in the reply mode. Please define cfg.seed");
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
 * @brief Load previously saved photon seeds from an .jdat file for replay
 *
 * @param[in] filename: the name/path of the .jdat file produced from the baseline simulation
 * @param[in,out] cfg: simulation configuration
 */

void mcx_loadseedjdat(char* filename, Config* cfg) {
    char* jbuf;
    int len;

    FILE* fp = fopen(filename, "rt");

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

    cJSON* root = cJSON_Parse(jbuf);
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
            History his = {0};

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

    if (cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS  || cfg->outputtype == otRF) { //cfg->replaydet>0
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

void  mcx_convertrow2col(unsigned int** vol, uint3* dim) {
    uint x, y, z;
    unsigned int dimxy, dimyz;
    unsigned int* newvol = NULL;

    if (*vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0) {
        return;
    }

    newvol = (unsigned int*)malloc(sizeof(unsigned int) * dim->x * dim->y * dim->z);
    dimxy = dim->x * dim->y;
    dimyz = dim->y * dim->z;

    for (x = 0; x < dim->x; x++)
        for (y = 0; y < dim->y; y++)
            for (z = 0; z < dim->z; z++) {
                newvol[z * dimxy + y * dim->x + x] = (*vol)[x * dimyz + y * dim->z + z];
            }

    free(*vol);
    *vol = newvol;
}

/**
 * @brief Convert a row-major (C/C++) array to a column-major (MATLAB/FORTRAN) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertrow2col64(size_t** vol, uint3* dim) {
    uint x, y, z;
    size_t dimxy, dimyz;
    size_t* newvol = NULL;

    if (*vol == NULL || dim->x == 0 || dim->y == 0 || dim->z == 0) {
        return;
    }

    newvol = (size_t*)malloc(sizeof(size_t) * dim->x * dim->y * dim->z);
    dimxy = dim->x * dim->y;
    dimyz = dim->y * dim->z;

    for (x = 0; x < dim->x; x++)
        for (y = 0; y < dim->y; y++)
            for (z = 0; z < dim->z; z++) {
                newvol[z * dimxy + y * dim->x + x] = (*vol)[x * dimyz + y * dim->z + z];
            }

    free(*vol);
    *vol = newvol;
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

    free(*vol);
    *vol = newvol;
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
    uint d, dx, dy, dz, idx1d, zi, yi, c, count;
    float x, y, z, ix, iy, iz, rx, ry, rz, d2, mind2, d2max;
    unsigned int* padvol;
    const float corners[8][3] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
        {1.f, 1.f, 0.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f}, {1.f, 1.f, 1.f}
    };

    dx = cfg->dim.x + 2;
    dy = cfg->dim.y + 2;
    dz = cfg->dim.z + 2;

    /*handling boundaries in a volume search is tedious, I first pad vol by a layer of zeros,
      then I don't need to worry about boundaries any more*/

    padvol = (unsigned int*)calloc(dx * dy * sizeof(unsigned int), dz);

    for (zi = 1; zi <= cfg->dim.z; zi++)
        for (yi = 1; yi <= cfg->dim.y; yi++) {
            memcpy(padvol + zi * dy * dx + yi * dx + 1, cfg->vol + (zi - 1)*cfg->dim.y * cfg->dim.x + (yi - 1)*cfg->dim.x, cfg->dim.x * sizeof(int));
        }

    /**
       The goal here is to find a set of voxels for each
    detector so that the intersection between a sphere
    of R=cfg->detradius,c0=cfg->detpos[d] and the object
    surface (or bounding box) is fully covered.
        */
    for (d = 0; d < cfg->detnum; d++) {                     /*loop over each detector*/
        count = 0;
        d2max = (cfg->detpos[d].w + 1.7321f) * (cfg->detpos[d].w + 1.7321f);

        for (z = -cfg->detpos[d].w - 1.f; z <= cfg->detpos[d].w + 1.f; z += 0.5f) { /*search in a cube with edge length 2*R+3*/
            iz = z + cfg->detpos[d].z;

            for (y = -cfg->detpos[d].w - 1.f; y <= cfg->detpos[d].w + 1.f; y += 0.5f) {
                iy = y + cfg->detpos[d].y;

                for (x = -cfg->detpos[d].w - 1.f; x <= cfg->detpos[d].w + 1.f; x += 0.5f) {
                    ix = x + cfg->detpos[d].x;

                    if (iz < 0 || ix < 0 || iy < 0 || ix >= cfg->dim.x || iy >= cfg->dim.y || iz >= cfg->dim.z ||
                            x * x + y * y + z * z > (cfg->detpos[d].w + 1.f) * (cfg->detpos[d].w + 1.f)) {
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

                    if (mind2 == VERY_BIG || mind2 >= (cfg->detpos[d].w + 0.5f) * (cfg->detpos[d].w + 0.5f)) {
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
            mcx_savejnii((float*)buf, 3, dims, voxelsize, fname, 0, cfg);
        } else {
            mcx_savebnii((float*)buf, 3, dims, voxelsize, fname, 0, cfg);
        }

        free(buf);
    } else {
        mcx_savenii((float*)cfg->vol, cfg->dim.x * cfg->dim.y * cfg->dim.z, fname, NIFTI_TYPE_UINT32, ofNifti, cfg);
    }

    if (cfg->isdumpmask == 1 && cfg->isdumpjson == 0) { /*if dumpmask>1, simulation will also run*/
        MCX_FPRINTF(cfg->flog, "volume mask is saved in %s\n", fname);
        exit(0);
    }
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
    cJSON* ztype = NULL;
    cJSON* vsize = cJSON_GetObjectItem(obj, "_ArraySize_");
    cJSON* vtype = cJSON_GetObjectItem(obj, "_ArrayType_");
    cJSON* vdata = cJSON_GetObjectItem(obj, "_ArrayData_");

    if (!vdata) {
        ztype = cJSON_GetObjectItem(obj, "_ArrayZipType_");
        vdata = cJSON_GetObjectItem(obj, "_ArrayZipData_");
    }

    if (vtype) {
        *type = vtype->valuestring;

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
        if (vsize) {
            cJSON* tmp = vsize->child;
            *ndim = cJSON_GetArraySize(vsize);

            for (int i = 0; i < MIN(maxdim, *ndim); i++) {
                dims[i] = tmp->valueint;
                tmp = tmp->next;
            }
        }

        if (ztype) {
            size_t len, newlen;
            int status = 0;
            char* buf = NULL;
            int zipid = mcx_keylookup((char*)(ztype->valuestring), zipformat);
            ret = zmat_decode(strlen(vdata->valuestring), (uchar*)vdata->valuestring, &len, (uchar**)&buf, zmBase64, &status);

            if (!ret && vsize) {
                if (*vol) {
                    free(*vol);
                }

                ret = zmat_decode(len, (uchar*)buf, &newlen, (uchar**)(vol), zipid, &status);
            }

            if (buf) {
                free(buf);
            }

            cfg->isrowmajor = 1;
        } else {
            MCX_ERROR(-1, "Only compressed JData array constructs are supported");
        }
    } else {
        MCX_ERROR(-1, "No _ArrayZipData_ field is found");
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
 * @param[in] obj: a pre-created cJSON object to store the output JData fields
 */

int  mcx_jdataencode(void* vol, int ndim, uint* dims, char* type, int byte, int zipid, void* obj, int isubj, Config* cfg) {
    uint datalen = 1;
    size_t compressedbytes, totalbytes;
    uchar* compressed = NULL, *buf = NULL;
    int ret = 0, status = 0;

    for (int i = 0; i < ndim; i++) {
        datalen *= dims[i];
    }

    totalbytes = datalen * byte;

    if (!cfg->isdumpjson) {
        MCX_FPRINTF(stdout, "compressing data [%s] ...", zipformat[zipid]);
    }

    /*compress data using zlib*/
    ret = zmat_encode(totalbytes, (uchar*)vol, &compressedbytes, (uchar**)&compressed, zipid, &status);

    if (!ret) {
        if (!cfg->isdumpjson) {
            MCX_FPRINTF(stdout, "compression ratio: %.1f%%\t", compressedbytes * 100.f / totalbytes);
        }

        if (isubj) {
            ubjw_context_t* item = (ubjw_context_t*)obj;
            UBJ_WRITE_KEY(item, "_ArrayType_", string, type);
            ubjw_write_key(item, "_ArraySize_");
            UBJ_WRITE_ARRAY(item, uint32, ndim, dims);
            UBJ_WRITE_KEY(item, "_ArrayZipType_", string, zipformat[zipid]);
            UBJ_WRITE_KEY(item, "_ArrayZipSize_", uint32, datalen);
            ubjw_write_key(item, "_ArrayZipData_");
            ubjw_write_buffer(item, compressed, UBJ_UINT8, compressedbytes);
        } else {
            totalbytes = 0;
            /*encode data using base64*/
            ret = zmat_encode(compressedbytes, compressed, &totalbytes, (uchar**)&buf, zmBase64, &status);

            if (!cfg->isdumpjson) {
                MCX_FPRINTF(stdout, "after encoding: %.1f%%\n", totalbytes * 100.f / (datalen * byte));
            }

            if (!ret) {
                cJSON_AddStringToObject((cJSON*)obj, "_ArrayType_", type);
                cJSON_AddItemToObject((cJSON*)obj,   "_ArraySize_", cJSON_CreateIntArray((int*)dims, ndim));
                cJSON_AddStringToObject((cJSON*)obj, "_ArrayZipType_", zipformat[zipid]);
                cJSON_AddNumberToObject((cJSON*)obj, "_ArrayZipSize_", datalen);
                cJSON_AddStringToObject((cJSON*)obj, "_ArrayZipData_", (char*)buf);
            }
        }
    }

    if (compressed) {
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
    int i = 1, isinteractive = 1, issavelog = 0, len;
    char filename[MAX_PATH_LENGTH] = {0}, *jsoninput = NULL;
    char logfile[MAX_PATH_LENGTH] = {0};
    float np = 0.f;

    if (argc <= 1) {
        mcx_usage(cfg, argv[0]);
        exit(0);
    }

    while (i < argc) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == '-') {
                if (mcx_remap(argv[i])) {
                    MCX_FPRINTF(cfg->flog, "Command option: %s", argv[i]);
                    MCX_ERROR(-2, "unknown verbose option");
                }
            } else if (strlen(argv[i]) > 2) {
                MCX_FPRINTF(cfg->flog, "Command option: %s", argv[i]);
                MCX_ERROR(-2, "unknown short option");
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
                        MCX_ERROR(-2, "you can not specify both interactive mode and config file");
                    }

                    isinteractive = 1;
                    break;

                case 'f':
                    isinteractive = 0;

                    if (i < argc - 1 && argv[i + 1][0] == '{') {
                        jsoninput = argv[i + 1];
                        i++;
                    } else {
                        i = mcx_readarg(argc, argv, i, filename, "string");
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
                    i = mcx_readarg(argc, argv, i, cfg->session, "string");
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
                    issavelog = 1;
                    break;

                case 'L':
                    cfg->isgpuinfo = 2;
                    break;

                case 'I':
                    cfg->isgpuinfo = 1;
                    break;

                case 'G':
                    if (mcx_isbinstr(argv[i + 1])) {
                        i = mcx_readarg(argc, argv, i, cfg->deviceid, "string");
                        break;
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->gpuid), "int");
                        memset(cfg->deviceid, '0', MAX_DEVICE);

                        if (cfg->gpuid > 0 && cfg->gpuid < MAX_DEVICE) {
                            cfg->deviceid[cfg->gpuid - 1] = '1';
                        } else {
                            MCX_ERROR(-2, "GPU id can not be more than 256");
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

                        cfg->shapedata = (char*)malloc(len);
                        memcpy(cfg->shapedata, argv[++i], len);
                    } else {
                        MCX_ERROR(-1, "json shape constructs are expected after -P");
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
                        MCX_ERROR(-1, "json fragment is expected after --json");
                    }

                    break;

                case 'A':
                    i = mcx_readarg(argc, argv, i, &(cfg->autopilot), "char");
                    break;

                case 'E':
                    if (i < argc - 1 && (strstr(argv[i + 1], ".mch") != NULL || strstr(argv[i + 1], ".jdat") != NULL) ) { /*give an mch file to initialize the seed*/
#if defined(USE_LL5_RAND)
                        MCX_ERROR(-1, "seeding file is not supported in this binary");
#else
                        i = mcx_readarg(argc, argv, i, cfg->seedfile, "string");
                        cfg->seed = SEED_FROM_FILE;
#endif
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->seed), "int");
                    }

                    break;

                case 'O':
                    i = mcx_readarg(argc, argv, i, &(cfg->outputtype), "string");

                    if (mcx_lookupindex(&(cfg->outputtype), outputtype)) {
                        MCX_ERROR(-2, "the specified output data type is not recognized");
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
                            MCX_ERROR(-1, "Unsupported media format.");
                        }

                        cfg->mediabyte = mediaformatid[cfg->mediabyte];
                    } else {
                        i = mcx_readarg(argc, argv, i, &(cfg->mediabyte), "int");
                    }

                    break;

                case 'F':
                    if (i >= argc) {
                        MCX_ERROR(-1, "incomplete input");
                    }

                    if ((cfg->outputformat = mcx_keylookup(argv[++i], outputformat)) < 0) {
                        MCX_ERROR(-2, "the specified output data type is not recognized");
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
                        i = mcx_readarg(argc, argv, i, cfg->rootpath, "string");
                    } else if (strcmp(argv[i] + 2, "dumpjson") == 0) {
                        cfg->jsonfile[0] = '-';

                        if (i + 1 >= argc) {
                            cfg->isdumpjson = 1;
                            i++;
                        } else if (i + 1 < argc && (isalpha(argv[i + 1][0]) || argv[i + 1][0] == '-')) {
                            cfg->isdumpjson = 1;
                            memcpy(cfg->jsonfile, argv[i + 1], MIN(strlen(argv[i + 1]), MAX_PATH_LENGTH));
                            i++;
                        } else {
                            i = mcx_readarg(argc, argv, i, &(cfg->isdumpjson), "int");
                        }
                    } else if (strcmp(argv[i] + 2, "bench") == 0) {
                        if (i + 1 < argc && isalpha(argv[i + 1][0]) ) {
                            int idx = mcx_keylookup(argv[++i], benchname);

                            if (idx == -1) {
                                MCX_ERROR(-1, "Unsupported bechmark.");
                            }

                            isinteractive = 0;
                            jsoninput = (char*)benchjson[idx];
                        } else {
                            MCX_FPRINTF(cfg->flog, "Built-in benchmarks:\n");

                            for (int i = 0; i < MAX_MCX_BENCH - 1; i++) {
                                if (benchname[i][0] == '\0') {
                                    break;
                                }

                                MCX_FPRINTF(cfg->flog, "\t%s\n", benchname[i]);
                            }

                            exit(0);
                        }
                    } else if (strcmp(argv[i] + 2, "reflectin") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->isrefint), "char");
                    } else if (strcmp(argv[i] + 2, "atomic") == 0) {
                        int isatomic = 1;
                        i = mcx_readarg(argc, argv, i, &(isatomic), "char");
                        cfg->sradius = (isatomic) ? -2.f : 0.f;
                    } else if (strcmp(argv[i] + 2, "internalsrc") == 0) {
                        i = mcx_readarg(argc, argv, i, &(cfg->internalsrc), "int");
                    } else {
                        MCX_FPRINTF(cfg->flog, "unknown verbose option: --%s\n", argv[i] + 2);
                    }

                    break;

                default:
                    MCX_FPRINTF(cfg->flog, "Command option: %s", argv[i]);
                    MCX_ERROR(-2, "unknown short option");
                    break;
            }
        }

        i++;
    }

    if (issavelog && cfg->session[0]) {
        sprintf(logfile, "%s.log", cfg->session);
        cfg->flog = fopen(logfile, "wt");

        if (cfg->flog == NULL) {
            cfg->flog = stdout;
            MCX_FPRINTF(cfg->flog, "unable to save to log file, will print from stdout\n");
        }
    }

    if ((cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS  || cfg->outputtype == otRF) && cfg->seed != SEED_FROM_FILE) {
        MCX_ERROR(-1, "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.");
    }

    if (cfg->isgpuinfo != 2) { /*print gpu info only*/
        if (isinteractive) {
            mcx_readconfig((char*)"", cfg);
        } else if (jsoninput) {
            mcx_readconfig(jsoninput, cfg);
        } else {
            mcx_readconfig(filename, cfg);
        }

        if (cfg->extrajson) {
            cJSON* jroot = cJSON_Parse(cfg->extrajson);

            if (jroot) {
                cfg->extrajson[0] = '_';
                mcx_loadjson(jroot, cfg);
                cJSON_Delete(jroot);
            } else {
                MCX_ERROR(-1, "invalid json fragment following --json");
            }
        }
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
    int v = 0;
    sscanf(ver, "$Rev::%x", &v);
    MCX_FPRINTF(cfg->flog, "MCX Revision %x\n", v);
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
        MCX_ERROR(-1, "No GPU device found\n");
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
    MCX_FPRINTF(cfg->flog, S_MAGENTA"\
###############################################################################\n\
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #\n\
#          Copyright (c) 2009-2023 Qianqian Fang <q.fang at neu.edu>          #\n\
#                https://mcx.space/  &  https://neurojson.org/                #\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.org #\n\
#   Department of Bioengineering, Northeastern University, Boston, MA, USA    #\n\
###############################################################################\n\
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #\n\
###############################################################################\n\
#  Open-source codes and reusable scientific data are essential for research, #\n\
# MCX proudly developed human-readable JSON-based data formats for easy reuse,#\n\
#  Please consider using JSON (https://neurojson.org/) for your research data #\n\
###############################################################################\n\
$Rev::      $ " MCX_VERSION "  $Date::                       $ by $Author::             $\n\
###############################################################################\n" S_RESET);
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
 -f config     (--input)       read an input file in .json or .inp format\n\
                               if the string starts with '{', it is parsed as\n\
			       an inline JSON input file\n\
      or\n\
 --bench ['cube60','skinvessel',..] run a buint-in benchmark specified by name\n\
                               run --bench without parameter to get a list\n\
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
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw\n\
 -E [0|int|mch](--seed)        set random-number-generator seed, -1 to generate\n\
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
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto\n\
      or\n\
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)\n\
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum\n\
 -I            (--printgpu)    print GPU information and run program\n\
 --atomic [1|0]                1: use atomic operations to avoid thread racing\n\
                               0: do not use atomic operation (not recommended)\n\
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
			      97 or svmc: split-voxel MC 8-byte format\n\
			        {[n.z][n.y][n.x][p.z][p.y][p.x][upper][lower]}\n\
			      98 or mixlabel: label1+label2+label1_percentage\n\
			        {[label1][label2][s:0-65535 label1 percentage]}\n\
			      99 or labelplus: 32bit composite voxel format\n\
			        {[h:mua/mus/g/n][s:(B15-16:0/1/2/3)(label)]}\n\
                             100 or muamus_float: 2x 32bit floats for mua/mus\n\
			        {[f:mua][f:mus]}; g/n from medium type 1\n\
                             101 or mua_float: 1 float per voxel for mua\n\
			        {[f:mua]}; mus/g/n from medium type 1\n\
			     102 or muamus_half: 2x 16bit float for mua/mus\n\
			        {[h:mua][h:mus]}; g/n from medium type 1\n\
			     103 or asgn_byte: 4x byte gray-levels for mua/s/g/n\n\
			        {[mua][mus][g][n]}; 0-255 mixing prop types 1&2\n\
			     104 or muamus_short: 2x short gray-levels for mua/s\n\
			        {[s:mua][s:mus]}; 0-65535 mixing prop types 1&2\n\
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array\n\
\n"S_BOLD S_CYAN"\
== Output options ==\n" S_RESET"\
 -s sessionid  (--session)     a string to label all output file names\n\
 -O [X|XFEJPMRL](--outputtype) X - output flux, F - fluence, E - energy deposit\n\
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, \n\
			       event counts at each voxel (replay mode only)\n\
                               M - momentum transfer; R - RF/FD Jacobian\n\
                               L - total pathlength\n\
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
      combine multiple items by using a string, or add selected numbers together\n\
      by default, mcx only saves detector ID and partial-path data\n\
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions\n\
                               setting -x to 1 also implies setting '-d' to 1.\n\
			       same as adding 'XV' to -w.\n\
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels\n\
                               right outside of the domain; if non-zero voxels\n\
			       appear at the boundary, pad 0s before using -X\n\
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save.\n\
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
	load jnii (JSON) and bnii (UBJSON) files using below lightweight libs:\n\
	  MATLAB/Octave: JNIfTI toolbox   https://github.com/NeuroJSON/jnifti,\n\
	  MATLAB/Octave: JSONLab toolbox  https://github.com/NeuroJSON/jsonlab,\n\
	  Python:        PyJData:         https://pypi.org/project/jdata\n\
	  JavaScript:    JSData:          https://github.com/NeuroJSON/jsdata\n\
 -Z [zlib|...] (--zip)         set compression method if -F jnii or --dumpjson\n\
                               is used (when saving data to JSON/JNIfTI format)\n\
			       0 zlib: zip format (moderate compression,fast) \n\
			       1 gzip: gzip format (compatible with *.gz)\n\
			       2 base64: base64 encoding with no compression\n\
			       3 lzip: lzip format (high compression,very slow)\n\
			       4 lzma: lzma format (high compression,very slow)\n\
			       5 lz4: LZ4 format (low compression,extrem. fast)\n\
			       6 lz4hc: LZ4HC format (moderate compression,fast)\n\
 --dumpjson [-,0,1,'file.json']  export all settings, including volume data using\n\
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
 -v            (--version)     print MCX revision number\n\
 -l            (--log)         print messages to a log file instead\n\
 -i 	       (--interactive) interactive mode\n\
\n"S_BOLD S_CYAN"\
== Debug options ==\n" S_RESET"\
 -D [0|int]    (--debug)       print debug information (you can use an integer\n\
  or                           or a string by combining the following flags)\n\
 -D [''|RMPT]                  1 R  debug RNG\n\
    /case insensitive/         2 M  store photon trajectory info\n\
                               4 P  print progress bar\n\
                               8 T  save trajectory data only, disable flux/detp\n\
      combine multiple items by using a string, or add selected numbers together\n\
\n"S_BOLD S_CYAN"\
== Additional options ==\n" S_RESET"\
 --root         [''|string]    full path to the folder storing the input files\n\
 --gscatter     [1e9|int]      after a photon completes the specified number of\n\
                               scattering events, mcx then ignores anisotropy g\n\
                               and only performs isotropic scattering for speed\n\
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch\n\
 --maxvoidstep  [1000|int]     maximum distance (in voxel unit) of a photon that\n\
                               can travel before entering the domain, if \n\
                               launched outside (i.e. a widefield source)\n\
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),\n\
                               use this parameter to set the maximum positions\n\
                               stored (default: 1e7)\n\
\n"S_BOLD S_CYAN"\
== Example ==\n" S_RESET"\
example: (list built-in benchmarks)\n"S_MAGENTA"\
       %s --bench\n" S_RESET"\
or (list supported GPUs on the system)\n"S_MAGENTA"\
       %s -L\n" S_RESET"\
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)\n"S_MAGENTA"\
       %s --bench cube60b -n 1e7 -G 1101 -W 10,10,10\n" S_RESET"\
or (use inline domain definition)\n"S_MAGENTA"\
       %s -f input.json -P '{\"Shapes\":[{\"ZLayers\":[[1,10,1],[11,30,2],[31,60,3]]}]}'\n" S_RESET"\
or (use inline json setting modifier)\n"S_MAGENTA"\
       %s -f input.json -j '{\"Optode\":{\"Source\":{\"Type\":\"isotropic\"}}}'\n" S_RESET"\
or (dump simulation in a single json file)\n"S_MAGENTA"\
       %s --bench cube60planar --dumpjson" S_RESET"\n",
           exename, exename, exename, exename, exename, exename, exename);
}
