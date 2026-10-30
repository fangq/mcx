/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2026
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    test_utils.c

@brief   Host-side unit tests of the MCX/MCX-CL utility, shape and Mie functions

Each test calls a function of mcx_utils.c, mcx_shapes.c or mcx_mie.cpp directly
on a tiny input, such as a 2x2x2 volume, a handful of detected photons or a
short JSON string, instead of running a simulation. Files are written to and
read back from a temporary folder. The same file is used by mcx and mcx-cl.
*******************************************************************************/

#define _DEFAULT_SOURCE  /* mkdtemp */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#include "unittest.h"
#include "mcx_utils.h"
#include "mcx_shapes.h"
#include "mcx_const.h"
#include "ubj.h"
#include "mcx_neurojson.h"
#include "zmatlib.h"

void Mie(double x, double mx, const double* mu, float4* smatrix, double* qsca, double* g);
void mcx_preprocess(Config* cfg);
void mcx_loadseedjdat(char* filename, Config* cfg);
int mcx_readarg(int argc, char* argv[], int id, void* output, const char* type);
void mcx_loadbenchmark(char* key, Config* cfg);
#ifdef MCX_OPENCL
    void mcx_run_simulation(Config* cfg, float* fluence, float* totalenergy);
#endif

#ifndef MAX_PROP_AND_DETECTORS
    #define MAX_PROP_AND_DETECTORS (MAX_PROP + MAX_DETECTORS)  /* the total limit of mcx-cl */
#endif

#ifdef MCX_OPENCL
    typedef uint4 GridDim;  /* mcx-cl stores the domain size and shape grid dimensions as uint4 */
#else
    typedef uint3 GridDim;
#endif

extern const char* srctypeid[];
extern const char* zipformat[];
extern const char* mediaformat[];
extern char flagset[256];  /* command line options already set, which the JSON input must not override */

static FILE* devnull = NULL;

/** a Config with default settings that prints nothing */
static void init_cfg(Config* cfg) {
    memset(flagset, 0, sizeof(flagset));  /* forget the options of the previous test */
    mcx_initcfg(cfg);
    cfg->flog = devnull;
}

static void clear_cfg(Config* cfg) {
    cfg->flog = stdout;  /* mcx_clearcfg closes flog unless it is stdout */
    mcx_clearcfg(cfg);
}

/** read a whole file into a NUL-terminated buffer, return its size or -1 */
static long read_file(const char* fname, char** buf) {
    FILE* fp = fopen(fname, "rb");
    long len;

    *buf = NULL;

    if (fp == NULL) {
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    len = ftell(fp);
    rewind(fp);
    *buf = (char*)calloc(len + 1, 1);

    if (fread(*buf, 1, len, fp) != (size_t)len) {
        len = -1;
    }

    fclose(fp);
    return len;
}

static void write_file(const char* fname, const void* buf, size_t len) {
    FILE* fp = fopen(fname, "wb");

    if (fp) {
        fwrite(buf, 1, len, fp);
        fclose(fp);
    }
}

/** true if the binary buffer contains the string key */
static int has_key(const char* buf, long len, const char* key) {
    size_t klen = strlen(key);

    for (long i = 0; i + (long)klen <= len; i++) {
        if (memcmp(buf + i, key, klen) == 0) {
            return 1;
        }
    }

    return 0;
}

/*-----------------------------------------------------------------------------
 *  string, key and small numeric helpers
 *---------------------------------------------------------------------------*/

static void test_helpers(void) {
    const char* table[] = {"alpha", "beta", "gamma", ""};

    EXPECT_EQ(mcx_keylookup((char*)"beta", table), 1, "mcx_keylookup finds a key");
    EXPECT_EQ(mcx_keylookup((char*)"delta", table), -1, "mcx_keylookup misses a key");
    EXPECT_EQ(mcx_keystartwith((char*)"gam", table), 2, "mcx_keystartwith finds the entry containing the key");
    EXPECT_EQ(mcx_keystartwith((char*)"zeta", table), -1, "mcx_keystartwith misses");
    char bc[2] = "r";
    EXPECT_EQ(mcx_lookupindex(bc, "_ram"), 0, "mcx_lookupindex finds a char");
    EXPECT_EQ(bc[0], 1, "mcx_lookupindex replaces the char by its index");
    EXPECT_EQ(mcx_parsedebugopt((char*)"RP", "RMP"), 1 | 4, "mcx_parsedebugopt sets bits by letter");
    EXPECT_TRUE(mcx_isbinstr("0101"), "mcx_isbinstr accepts a binary string");
    EXPECT_TRUE(!mcx_isbinstr("0121"), "mcx_isbinstr rejects a non-binary string");
    EXPECT_EQ(mcx_keylookup((char*)"zgaussian", srctypeid), MCX_SRC_ZGAUSSIAN, "srctypeid lists zgaussian");

    float sum = 0.f, c = 0.f;

    for (int i = 0; i < 1000; i++) {
        mcx_kahanSum(&sum, &c, 0.1f);
    }

    EXPECT_NEAR(sum, 100.f, 1e-4, "mcx_kahanSum compensates the rounding error");

    float field[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    mcx_normalize(field, 0.5f, 8, 0, 1);
    EXPECT_NEAR(field[7], 4.f, 1e-6, "mcx_normalize scales the field");

#ifndef MCX_OPENCL
    /* an svmc voxel without a lower label (bits 24-31) is a background voxel */
    EXPECT_EQ(mcx_svmc_bgvoxel(0x00FFFFFF), 1, "mcx_svmc_bgvoxel, no lower label");
    EXPECT_EQ(mcx_svmc_bgvoxel(0x02000000), 0, "mcx_svmc_bgvoxel, with a lower label");
#endif
}

/*-----------------------------------------------------------------------------
 *  half precision conversion used by the continuous media formats
 *---------------------------------------------------------------------------*/

static void test_half(void) {
    /* normal values, the largest half, and denormals between 2^-24 and 2^-14 */
    const float vals[] = {1.f, -2.5f, 0.0004f, 65504.f, 3e-6f, -7e-5f, 0.03f, 0.04f};

    for (int i = 0; i < (int)(sizeof(vals) / sizeof(float)); i += 2) {
        float in[2] = {vals[i], vals[i + 1]};
        unsigned int packed = (unsigned int)mcx_float2half2(in);
        float lo = mcx_half2float(packed & 0xFFFF), hi = mcx_half2float(packed >> 16);
        /* a truncated 10-bit mantissa, or a denormal with a 2^-24 step */
        EXPECT_NEAR(lo, in[0], fabs(in[0]) * 1e-3 + 6e-8, "half round trip, first value");
        EXPECT_NEAR(hi, in[1], fabs(in[1]) * 1e-3 + 6e-8, "half round trip, second value");
    }

    /* mcx_float2half2 never returns 0 for a zero mua, so the voxel is not a 0-label */
    float zeros[2] = {0.f, 0.f};
    EXPECT_TRUE(mcx_float2half2(zeros) != 0, "a zero mua/mus is not packed as label 0");

    EXPECT_EQ(mcx_half2float(0x0000), 0.f, "half +0");
    EXPECT_TRUE(signbit(mcx_half2float(0x8000)), "half -0 keeps the sign");
    EXPECT_REL(mcx_half2float(0x0001), 5.9604645e-8, 1e-6, "smallest half denormal");
    EXPECT_TRUE(isinf(mcx_half2float(0x7C00)), "half +inf");
    EXPECT_TRUE(isnan(mcx_half2float(0x7E00)), "half NaN");
    EXPECT_EQ(mcx_half2float(0x3C00), 1.f, "half 1.0");

    float big[2] = {1e6f, INFINITY};
    unsigned int packed = (unsigned int)mcx_float2half2(big);
    EXPECT_EQ(packed & 0xFFFF, 0x7BFF, "values beyond the half range are clamped to 65504");
    EXPECT_EQ(packed >> 16, 0x7C00, "inf is kept");
}

/*-----------------------------------------------------------------------------
 *  Mie scattering, compared against an independent Bohren-Huffman solution
 *  (scipy spherical Bessel functions); the angular values are S11 and S12
 *  normalized by S11(mu=1), so only the shape of the phase matrix is compared
 *---------------------------------------------------------------------------*/

typedef struct {
    double x, m, qsca, g, s11[5], s12[5];
} MieRef;

static const MieRef mieref[] = {
    /* x*m < 0.1 uses small_Mie */
    {0.05, 1.2, 2.7253668821e-07, 4.3291420820e-04, {1.00000000e+00, 6.24684703e-01, 4.99469912e-01, 6.23990236e-01, 9.97880592e-01}, {0.00000000e+00, -3.74785089e-01, -4.99469910e-01, -3.74419852e-01, 0.00000000e+00}},
    {0.06, 1.59, 3.9378887744e-06, 7.4546881603e-04, {1.00000000e+00, 6.24527445e-01, 4.99120653e-01, 6.23274646e-01, 9.96485041e-01}, {0.00000000e+00, -3.74592872e-01, -4.99120610e-01, -3.74088192e-01, 0.00000000e+00}},
    {1.0, 1.33, 9.3924001214e-02, 1.8451667398e-01, {1.00000000e+00, 5.14609636e-01, 3.23119273e-01, 3.12709043e-01, 4.02105967e-01}, {0.00000000e+00, -2.91911898e-01, -3.22263924e-01, -1.99223761e-01, 0.00000000e+00}},
    {5.0, 1.5, 3.9278267316e+00, 7.0729478402e-01, {1.00000000e+00, 2.47106849e-02, 6.30611822e-03, 4.22447575e-03, 2.25774038e-02}, {0.00000000e+00, 4.12149154e-03, 8.55071421e-04, 3.89923261e-03, 0.00000000e+00}},
    {20.0, 1.1, 3.4467837684e+00, 9.7593143213e-01, {1.00000000e+00, 1.07849594e-04, 3.82494490e-05, 1.29355606e-05, 1.60096151e-05}, {0.00000000e+00, -2.32538027e-05, -1.85206494e-05, -6.42437511e-06, 0.00000000e+00}},
};

static void test_mie(void) {
    double* mu = (double*)calloc(NANGLES, sizeof(double));
    float4* smatrix = (float4*)calloc(NANGLES, sizeof(float4));
    const double mus[5] = {1.0, 0.5, 0.0, -0.5, -1.0};

    for (int i = 0; i < NANGLES; i++) {
        mu[i] = (i < 5) ? mus[i] : cos(3.14159265358979323846 * i / (NANGLES - 1));
    }

    for (int c = 0; c < (int)(sizeof(mieref) / sizeof(MieRef)); c++) {
        const MieRef* r = mieref + c;
        double qsca = 0., g = 0.;
        char msg[128];

        Mie(r->x, r->m, mu, smatrix, &qsca, &g);
        sprintf(msg, "Mie qsca, x=%g m=%g", r->x, r->m);
        EXPECT_REL(qsca, r->qsca, 1e-3, msg);
        sprintf(msg, "Mie g, x=%g m=%g", r->x, r->m);
        EXPECT_NEAR(g, r->g, 1e-4 + 1e-3 * fabs(r->g), msg);

        for (int j = 1; j < 5; j++) {
            sprintf(msg, "Mie S11(mu=%g)/S11(1), x=%g", mus[j], r->x);
            EXPECT_NEAR(smatrix[j].x / smatrix[0].x, r->s11[j], 1e-4 + 1e-3 * fabs(r->s11[j]), msg);
            sprintf(msg, "Mie S12(mu=%g)/S11(1), x=%g", mus[j], r->x);
            EXPECT_NEAR(smatrix[j].y / smatrix[0].x, r->s12[j], 1e-4 + 1e-3 * fabs(r->s12[j]), msg);
        }
    }

    free(mu);
    free(smatrix);
}

/*-----------------------------------------------------------------------------
 *  JData array encoding/decoding with every compression method
 *---------------------------------------------------------------------------*/

static void test_jdata(void) {
    Config cfg;
    uint dims[3] = {3, 4, 5};
    float data[60];

    init_cfg(&cfg);
    cfg.isdumpjson = 1;  /* do not print the compression ratio */

    for (int i = 0; i < 60; i++) {
        data[i] = i * 0.25f - 3.f;
    }

    for (int zipid = 0; zipformat[zipid][0]; zipid++) {
        char msg[128];

        if (strcmp(zipformat[zipid], "lzip") == 0) {
            continue;  /* zmat decodes lzip but does not encode it */
        }

        cJSON* obj = cJSON_CreateObject();
        int ret = mcx_jdataencode(data, 3, dims, (char*)"single", sizeof(float), zipid, obj, 0, 0, &cfg);
        sprintf(msg, "mcx_jdataencode with %s", zipformat[zipid]);
        EXPECT_EQ(ret, 0, msg);

        void* vol = NULL;
        char* type = NULL;
        int ndim = 0;
        uint odims[3] = {0};
        mcx_jdatadecode(&vol, &ndim, odims, 3, &type, obj, &cfg);
        sprintf(msg, "mcx_jdatadecode with %s restores the dimensions", zipformat[zipid]);
        EXPECT_TRUE(ndim == 3 && odims[0] == 3 && odims[1] == 4 && odims[2] == 5, msg);
        sprintf(msg, "mcx_jdatadecode with %s restores the data", zipformat[zipid]);
        EXPECT_TRUE(vol && memcmp(vol, data, sizeof(data)) == 0, msg);
        free(vol);
        cJSON_Delete(obj);
    }

    /* an uncompressed base64 _ArrayData_ string and a plain numeric _ArrayData_ list */
    cJSON* obj = cJSON_Parse("{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2,2],\"_ArrayData_\":\"AQIDBA==\"}");
    void* vol = NULL;
    char* type = NULL;
    int ndim = 0;
    uint odims[3] = {0};
    mcx_jdatadecode(&vol, &ndim, odims, 3, &type, obj, &cfg);
    EXPECT_TRUE(vol && ((unsigned char*)vol)[3] == 4, "base64 _ArrayData_ string is decoded");
    free(vol);
    cJSON_Delete(obj);

    obj = cJSON_Parse("{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2,2],\"_ArrayData_\":[1,2,3,4]}");
    vol = NULL;
    cfg.mediabyte = 1;
    mcx_jdatadecode(&vol, &ndim, odims, 3, &type, obj, &cfg);
    EXPECT_TRUE(vol && ((unsigned char*)vol)[2] == 3, "numeric _ArrayData_ list is decoded");
    free(vol);
    cJSON_Delete(obj);

    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  JData arrays over 4 MB: zlib/gzip deflate them in parallel blocks and save
 *  the block index as _ArrayZipOffsets_, which the reader uses to inflate them
 *  in parallel; a malformed index falls back to a serial inflate
 *---------------------------------------------------------------------------*/

static void test_jdata_blocks(void) {
    Config cfg;
    const uint len = 3u << 20;  /* 12 MB of float32 */
    uint dims[1] = {len};
    float* data = (float*)malloc(len * sizeof(float));
    const char* zips[] = {"zlib", "gzip"};

    init_cfg(&cfg);
    cfg.isdumpjson = 1;

    for (uint i = 0; i < len; i++) {
        data[i] = (float)((i * 2654435761u) % 1000u) * 0.5f;
    }

    for (int z = 0; z < 2; z++) {
        int zipid = 0;
        char msg[128];

        while (strcmp(zipformat[zipid], zips[z])) {
            zipid++;
        }

        /* text JSON: the index is a list of [compressed, uncompressed] offset pairs */
        cJSON* obj = cJSON_CreateObject();
        int ret = mcx_jdataencode(data, 1, dims, (char*)"single", sizeof(float), zipid, obj, 0, 0, &cfg);
        cJSON* idx = cJSON_GetObjectItem(obj, "_ArrayZipOffsets_");
        sprintf(msg, "a 12 MB %s array saves _ArrayZipOffsets_", zips[z]);
        EXPECT_TRUE(ret == 0 && idx && cJSON_GetArraySize(idx) >= 3, msg);

        void* vol = NULL;
        char* type = NULL;
        int ndim = 0;
        uint odims[1] = {0};
        mcx_jdatadecode(&vol, &ndim, odims, 1, &type, obj, &cfg);
        sprintf(msg, "a 12 MB %s array is inflated in parallel blocks", zips[z]);
        EXPECT_TRUE(vol && odims[0] == len && memcmp(vol, data, len * sizeof(float)) == 0, msg);
        free(vol);

        /* malformed indices: an odd count, too few numbers, a negative or a
           non-numeric offset; each is ignored and the stream inflated serially */
        const char* bad[] = {"[[0,0],[1,1],[2]]", "[0,0]", "[[0,0],[-1,5],[9,9]]", "[[0,0],[\"a\",1],[9,9]]"};

        for (int b = 0; b < 4; b++) {
            cJSON_ReplaceItemInObject(obj, "_ArrayZipOffsets_", cJSON_Parse(bad[b]));
            vol = NULL;
            mcx_jdatadecode(&vol, &ndim, odims, 1, &type, obj, &cfg);
            sprintf(msg, "a %s array with a malformed index %s is inflated serially", zips[z], bad[b]);
            EXPECT_TRUE(vol && memcmp(vol, data, len * sizeof(float)) == 0, msg);
            free(vol);
        }

        cJSON_Delete(obj);

        /* binary JSON: the index is written as an array of uint64 pairs */
        size_t cap = len * sizeof(float) + (1 << 20);
        unsigned char* out = (unsigned char*)malloc(cap);
        ubjw_context_t* ctx = ubjw_open_memory(out, out + cap);
        ubjw_begin_object(ctx, UBJ_MIXED, 0);
        ret = mcx_jdataencode(data, 1, dims, (char*)"single", sizeof(float), zipid, ctx, 1, 0, &cfg);
        ubjw_end(ctx);
        size_t n = ubjw_close_context(ctx);
        sprintf(msg, "a 12 MB %s BJData array saves _ArrayZipOffsets_", zips[z]);
        EXPECT_TRUE(ret == 0 && n > 0 && has_key((const char*)out, (long)n, "_ArrayZipOffsets_"), msg);
        free(out);
    }

    free(data);
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  shape rasterization
 *---------------------------------------------------------------------------*/

static unsigned int voxel(unsigned int* vol, GridDim* dim, int i, int j, int k) {
    return vol[k * dim->x * dim->y + j * dim->x + i];
}

static void test_shapes(void) {
    unsigned int* vol = NULL;
    GridDim dim = {0};
    Grid3D g = {&vol, &dim, {0.f, 0.f, 0.f}, 0};

    const char* shapes = "{\"Shapes\":["
                         "{\"Name\":\"unit test\"},"
                         "{\"Grid\":{\"Tag\":1,\"Size\":[10,12,14]}},"
                         "{\"ZLayers\":[[1,2,2],[13,14,3]]},"
                         "{\"XSlabs\":{\"Tag\":4,\"Bound\":[2.5,3.5]}},"
                         "{\"YSlabs\":{\"Tag\":5,\"Bound\":[[10,11]]}},"
                         "{\"UpperSpace\":{\"Coef\":[1,0,0,8.9],\"Tag\":6}},"
                         "{\"Subgrid\":{\"O\":[4,4,4],\"Size\":[2,2,2],\"Tag\":7}},"
                         "{\"Origin\":[0,0,0]},"
                         "{\"Sphere\":{\"O\":[5,5,7],\"R\":1.2,\"Tag\":8}},"
                         "{\"Cylinder\":{\"C0\":[1.5,7.5,4],\"C1\":[1.5,7.5,10],\"R\":0.6,\"Tag\":9}},"
                         "{\"Box\":{\"O\":[6,1,6],\"Size\":[2,2,2],\"Tag\":10}}"
                         "]}";

    int status = mcx_parse_shapestring(&g, (char*)shapes);
    EXPECT_EQ(status, 0, "mcx_parse_shapestring returns 0");
    EXPECT_TRUE(dim.x == 10 && dim.y == 12 && dim.z == 14, "Grid sets the dimensions");

    if (status == 0 && vol) {
        EXPECT_EQ(voxel(vol, &dim, 5, 6, 0), 2, "ZLayers tags the bottom layers");
        EXPECT_EQ(voxel(vol, &dim, 5, 6, 13), 3, "ZLayers tags the top layer");
        EXPECT_EQ(voxel(vol, &dim, 0, 6, 6), 1, "Grid tag is kept elsewhere");
        EXPECT_EQ(voxel(vol, &dim, 2, 0, 6), 4, "XSlabs tags x in [2.5,3.5]");
        EXPECT_EQ(voxel(vol, &dim, 5, 10, 6), 5, "YSlabs tags y in [10,11]");
        EXPECT_EQ(voxel(vol, &dim, 9, 5, 6), 6, "UpperSpace tags x>8.9");
        EXPECT_EQ(voxel(vol, &dim, 4, 4, 4), 7, "Subgrid tags its block");
        EXPECT_EQ(voxel(vol, &dim, 5, 5, 7), 8, "Sphere tags its center");
        EXPECT_EQ(voxel(vol, &dim, 1, 7, 6), 9, "Cylinder tags its axis");
        EXPECT_EQ(voxel(vol, &dim, 6, 1, 6), 10, "Box tags its corner voxel");
    }

    /* a JSON shape file, loaded into an existing row-major volume */
    write_file("shapes.json", shapes, strlen(shapes));
    free(vol);
    vol = NULL;
    g.rowmajor = 1;
    status = mcx_load_jsonshapes(&g, (char*)"shapes.json");
    EXPECT_EQ(status, 0, "mcx_load_jsonshapes loads a shape file");
    EXPECT_EQ(mcx_find_shapeid((char*)"Sphere") >= 0, 1, "mcx_find_shapeid knows Sphere");

    /* error reporting */
    status = mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"UpperSpace\":{\"Tag\":1}}]}");
    EXPECT_TRUE(status != 0 && strlen(mcx_last_shapeerror()) > 0, "a malformed shape reports an error");
    status = mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"XLayers\":{\"Tag\":1}}]}");
    EXPECT_TRUE(status != 0, "a non-array XLayers reports an error");

    free(vol);
}

/*-----------------------------------------------------------------------------
 *  volume formats: each file format of mcx_loadvolume and mcx_updatemua
 *---------------------------------------------------------------------------*/

static void test_loadvolume(void) {
    Config cfg;
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 2;
    cfg.medianum = 3;
    cfg.prop = (Medium*)calloc(3, sizeof(Medium));
    cfg.prop[1].mua = 0.01f;
    cfg.prop[2].mua = 0.03f;

    unsigned char b[8] = {0, 1, 2, 1, 0, 1, 2, 1};
    write_file("v8.bin", b, sizeof(b));
    cfg.mediabyte = 1;
    mcx_loadvolume((char*)"v8.bin", &cfg, 0);
    EXPECT_EQ(cfg.vol[2], 2, "byte volume");
    EXPECT_NEAR(mcx_updatemua(cfg.vol[2], &cfg), 0.03f, 1e-7, "mcx_updatemua for a label");

    unsigned short s[8] = {0, 1, 2, 1, 0, 1, 2, 1};
    write_file("v16.bin", s, sizeof(s));
    cfg.mediabyte = 2;
    mcx_loadvolume((char*)"v16.bin", &cfg, 0);
    EXPECT_EQ(cfg.vol[6], 2, "short volume");

    float f[8] = {0.1f, 0.2f, 0.f, NAN, 0.1f, 0.2f, 0.3f, 0.4f};
    write_file("vf.bin", f, sizeof(f));
    cfg.mediabyte = MEDIA_MUA_FLOAT;
    cfg.unitinmm = 1.f;
    mcx_loadvolume((char*)"vf.bin", &cfg, 0);
    EXPECT_NEAR(mcx_updatemua(cfg.vol[1], &cfg), 0.2f, 1e-7, "mua_float volume and mcx_updatemua");
    EXPECT_TRUE(cfg.vol[2] != 0, "a zero mua is not a 0-label voxel");
    EXPECT_EQ(cfg.vol[3], 0, "a NaN mua becomes a 0-label voxel");

    float as[16];

    for (int i = 0; i < 16; i++) {
        as[i] = 0.01f * (i + 1);
    }

    write_file("vas.bin", as, sizeof(as));
    cfg.mediabyte = MEDIA_AS_F2H;
    mcx_loadvolume((char*)"vas.bin", &cfg, 0);
    EXPECT_REL(mcx_half2float(cfg.vol[1] & 0xFFFF), 0.03f, 1e-3, "muamus_float volume, mua as half");
    EXPECT_REL(mcx_half2float(cfg.vol[1] >> 16), 0.04f, 2e-3, "muamus_float volume, mus as half");

    float asgn[32];

    for (int i = 0; i < 32; i++) {
        asgn[i] = 0.01f * (i + 1);
    }

    write_file("vasgn.bin", asgn, sizeof(asgn));
    cfg.mediabyte = MEDIA_ASGN_F2H;
    mcx_loadvolume((char*)"vasgn.bin", &cfg, 0);
    EXPECT_REL(mcx_half2float(cfg.vol[8 + 1] & 0xFFFF), 0.07f, 2e-3, "asgn_float volume, g in the second half");

    unsigned int packed[8];

    for (int i = 0; i < 8; i++) {
        packed[i] = 0x01020300u + i;
    }

    cfg.mediabyte = MEDIA_ASGN_BYTE;
    mcx_loadvolume((char*)packed, &cfg, 1);
    EXPECT_EQ(cfg.vol[5], packed[5], "a packed 4-byte format copied from a buffer");
    EXPECT_NEAR(mcx_updatemua(0x000000FFu, &cfg), 0.03f, 1e-6, "mcx_updatemua for asgn_byte");
    cfg.mediabyte = MEDIA_AS_SHORT;
    EXPECT_NEAR(mcx_updatemua(0x0000FFFFu, &cfg), 0.03f, 1e-6, "mcx_updatemua for muamus_short");

    uint64_t svmc[8];

    for (int i = 0; i < 8; i++) {
        svmc[i] = 0x0102030405060000ull + (i % 3);  /* the lower label (low byte) indexes the 3 media of check_dumpjson */
    }

    write_file("vsvmc.bin", svmc, sizeof(svmc));
    cfg.mediabyte = MEDIA_2LABEL_SPLIT;
    mcx_loadvolume((char*)"vsvmc.bin", &cfg, 0);
    EXPECT_TRUE(memcmp(cfg.vol, svmc, sizeof(svmc)) == 0, "svmc volume is copied as is");

    /* a .json volume file is rasterized from shapes */
    const char* shapes = "{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}}]}";
    write_file("vshape.json", shapes, strlen(shapes));
    cfg.mediabyte = 1;
    free(cfg.vol);
    cfg.vol = NULL;
    mcx_loadvolume((char*)"vshape.json", &cfg, 0);
    EXPECT_TRUE(cfg.dim.x == 4 && cfg.vol && cfg.vol[63] == 1, "a .json volume file is rasterized");

    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  volume layout conversions
 *---------------------------------------------------------------------------*/

static void test_layout(void) {
    uint4 dim4 = {2, 3, 4, 2};  /* 2 values per voxel of a 2x3x4 volume */
    unsigned int* v = (unsigned int*)malloc(sizeof(unsigned int) * 48);

    for (int i = 0; i < 48; i++) {
        v[i] = i;
    }

    mcx_convertcol2row4d(&v, &dim4);
    /* column-major [w][x][y][z] index of (w=1,x=1,y=0,z=0) is 1+2*1=3 */
    EXPECT_TRUE(v != NULL, "mcx_convertcol2row4d returns a volume");
    free(v);

    uint3 dim3 = {2, 3, 4};
    size_t* v64 = (size_t*)malloc(sizeof(size_t) * 24);
    unsigned int* v32 = (unsigned int*)malloc(sizeof(unsigned int) * 24);

    for (int i = 0; i < 24; i++) {
        v64[i] = i;
        v32[i] = i;
    }

#ifdef MCX_OPENCL
    uint4 dim4c = {2, 3, 4, 0};
    mcx_convertrow2col64(&v64, &dim4c);
#else
    mcx_convertrow2col64(v64, &dim3);
#endif
    /* row-major (x,y,z) = x*12+y*4+z; column-major position of (1,0,0) is 1 */
    EXPECT_EQ(v64[1], 12, "mcx_convertrow2col64 moves (1,0,0)");
    mcx_convertcol2row(&v32, &dim3);
#ifdef MCX_OPENCL
    mcx_convertrow2col(&v32, &dim4c);
#else
    mcx_convertrow2col(v32, &dim3);
#endif
    EXPECT_EQ(v32[5], 5, "mcx_convertcol2row and mcx_convertrow2col are inverse");
    free(v64);
    free(v32);
}

/*-----------------------------------------------------------------------------
 *  output writers
 *---------------------------------------------------------------------------*/

static void setup_output_cfg(Config* cfg) {
    init_cfg(cfg);
    cfg->dim.x = 3;
    cfg->dim.y = 4;
    cfg->dim.z = 2;
    cfg->maxgate = 2;
    cfg->tstep = 1e-9f;
    cfg->tend = 2e-9f;
    cfg->steps.x = cfg->steps.y = cfg->steps.z = 1.f;
    cfg->srcnum = 1;
    strcpy(cfg->session, "ut");
}

static void test_savedata(void) {
    Config cfg;
    float dat[96];  /* 3x4x2 voxels x 2 gates, x2 for the real and imaginary parts of RF output */
    char* buf;
    long len;

    for (int i = 0; i < 96; i++) {
        dat[i] = i + 0.5f;
    }

    setup_output_cfg(&cfg);

    cfg.outputformat = ofMC2;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.mc2", &buf);
    EXPECT_TRUE(len == 24 * 4 && ((float*)buf)[23] == 23.5f, "mc2 output holds the raw floats");
    free(buf);

    cfg.outputformat = ofTX3;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.tx3", &buf);
    EXPECT_TRUE(len == 16 + 24 * 4 && ((unsigned int*)buf)[1] == 3, "tx3 output starts with the format and dimensions");
    free(buf);

    cfg.outputformat = ofNifti;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.nii", &buf);
    EXPECT_TRUE(len == 352 + 24 * 4 && *(int*)buf == 348, "nii output has a 348-byte header");

    if (len > 352) {
        EXPECT_EQ(((short*)(buf + 40))[1], 3, "nii dim[1]");
        EXPECT_EQ(((float*)(buf + 352))[5], 5.5f, "nii data follow the header");
    }

    free(buf);

    cfg.outputformat = ofAnalyze;
    mcx_savedata(dat, 24, &cfg);
    EXPECT_TRUE(read_file("ut.hdr", &buf) == 348, "analyze .hdr header");
    free(buf);
    EXPECT_TRUE(read_file("ut.img", &buf) == 24 * 4, "analyze .img data");
    free(buf);

    cfg.outputformat = ofJNifti;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.jnii", &buf);
    EXPECT_TRUE(len > 0 && has_key(buf, len, "NIFTIHeader") && has_key(buf, len, "_ArrayZipData_"), "jnii output");
    free(buf);

    cfg.outputformat = ofBJNifti;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.bnii", &buf);
    EXPECT_TRUE(len > 0 && buf[0] == '{' && has_key(buf, len, "NIFTIData") && has_key(buf, len, "_ArrayZipData_"), "bnii output");
    free(buf);

    /* RF output doubles the data, adjoint output uses Ns x Nd slots */
    cfg.outputtype = otRF;
    cfg.omega = 1e8f;
    cfg.outputformat = ofBJNifti;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.bnii", &buf);
    EXPECT_TRUE(len > 0, "RF bnii output");
    free(buf);

    cfg.outputtype = otAdjointMuaD;
    cfg.detnum = 1;
    cfg.extrasrclen = 1;
    cfg.detdir = (float4*)calloc(1, sizeof(float4));
    cfg.outputformat = ofJNifti;
    mcx_savedata(dat, 24, &cfg);
    len = read_file("ut.jnii", &buf);
    EXPECT_TRUE(len > 0, "dual adjoint jnii output");
    free(buf);

    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  detected photons: .mch, .jdat and .jdb writers, and replay from an .mch file
 *---------------------------------------------------------------------------*/

static void test_detphoton(void) {
    Config cfg;
    const int count = 4, nmedia = 2;
    float ppath[4 * 4];   /* detid, nscat(1 column per medium skipped), ppath x2, weight */
    unsigned char seeds[4 * 16];
    char* buf;
    long len;

    setup_output_cfg(&cfg);
    cfg.medianum = nmedia + 1;
    cfg.prop = (Medium*)calloc(cfg.medianum, sizeof(Medium));
    cfg.prop[1].mua = 0.01f;
    cfg.prop[2].mua = 0.02f;
    cfg.prop[1].n = cfg.prop[2].n = 1.37f;
    cfg.detnum = 2;
    cfg.issaveseed = 1;
    cfg.savedetflag = 0x1 | 0x4;  /* detector ID and partial path */
    cfg.his.maxmedia = nmedia;
    cfg.his.detnum = 2;
    cfg.his.colcount = 1 + nmedia + 1;
    cfg.his.savedetflag = cfg.savedetflag;
    cfg.his.seedbyte = 16;
    cfg.his.detected = count;
    cfg.his.savedphoton = count;
    cfg.his.unitinmm = 1.f;
    cfg.his.totalphoton = 1000;
    cfg.tend = 1.f;

    for (int i = 0; i < count; i++) {
        ppath[i * 4] = 1 + (i & 1);   /* detector ID */
        ppath[i * 4 + 1] = 2.f;       /* partial path in medium 1 */
        ppath[i * 4 + 2] = 3.f;       /* partial path in medium 2 */
        ppath[i * 4 + 3] = 0.5f;
    }

    for (int i = 0; i < count * 16; i++) {
        seeds[i] = (unsigned char)i;
    }

    cfg.outputformat = ofMC2;
    mcx_savedetphoton(ppath, seeds, count, 0, &cfg);
    len = read_file("ut.mch", &buf);
    EXPECT_EQ(len, (long)(sizeof(History) + count * 4 * 4 + count * 16), "mch file holds the header, photons and seeds");
    free(buf);

    /* replay the photons of detector 2 from the .mch file */
    strcpy(cfg.seedfile, "ut.mch");
    cfg.outputtype = otJacobian;
    cfg.replaydet = 2;
    mcx_loadseedfile(&cfg);
    EXPECT_EQ(cfg.seed, SEED_FROM_FILE, "mcx_loadseedfile sets the replay mode");
    EXPECT_EQ(cfg.nphoton, 2, "mcx_loadseedfile keeps the photons of detector 2");

    if (cfg.nphoton == 2) {
        EXPECT_NEAR(cfg.replay.weight[0], expf(-0.01f * 2.f - 0.02f * 3.f), 1e-6, "replay weight from the partial paths");
        EXPECT_EQ(cfg.replay.detid[1], 2, "replay detector ID");
        EXPECT_EQ(((unsigned char*)cfg.replay.seed)[0], 16, "replay seed of the 2nd photon in the file");
    }

    cfg.outputformat = ofJNifti;
    mcx_savedetphoton(ppath, seeds, count, 0, &cfg);
    len = read_file("ut_detp.jdt", &buf);

    EXPECT_TRUE(len > 0 && has_key(buf, len, "MCXData") && has_key(buf, len, "ppath"), "jdat detected photon output");
    free(buf);

    cfg.outputformat = ofBJNifti;
    cfg.issaveseed = 1;  /* cleared by mcx_loadseedfile above */
    mcx_savedetphoton(ppath, seeds, count, 0, &cfg);
    len = read_file("ut_detp.jdb", &buf);
    EXPECT_TRUE(len > 0 && has_key(buf, len, "ppath") && has_key(buf, len, "seed") && has_key(buf, len, "_ArrayZipData_"),
                "jdb detected photon output with the seeds");
    free(buf);

    /* trajectory records (detected 0) with the seeds, in a .jdb file */
    {
        float traj[2 * MCX_DEBUG_REC_LEN];

        for (int i = 0; i < 2 * MCX_DEBUG_REC_LEN; i++) {
            traj[i] = (float)(i % MCX_DEBUG_REC_LEN);
        }

        cfg.his.detected = 0;
        cfg.his.savedphoton = 2;
        cfg.his.colcount = MCX_DEBUG_REC_LEN;
        strcpy(cfg.session, "uttraj");
        mcx_savedetphoton(traj, seeds, 2, 0, &cfg);
        len = read_file("uttraj_traj.jdb", &buf);
        EXPECT_TRUE(len > 0 && has_key(buf, len, "Trajectory") && has_key(buf, len, "seed"), "jdb trajectory output with the seeds");
        free(buf);
    }

    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  legacy .inp input file
 *---------------------------------------------------------------------------*/

static void test_inp(void) {
    Config cfg;
    unsigned char vol[4 * 4 * 4];
    float pattern[2 * 3] = {1, 0.5f, 0.25f, 0, 1, 1};

    memset(vol, 1, sizeof(vol));
    vol[0] = 2;
    write_file("inp.bin", vol, sizeof(vol));
    write_file("inp_pat.bin", pattern, sizeof(pattern));

    const char* inp =
        "1000      # total photon\n"
        "12345     # RNG seed\n"
        "2 2 1     # source position\n"
        "0 0 1     # source direction\n"
        "0 5e-9 1e-9  # time gates\n"
        "inp.bin   # volume\n"
        "1 4 1 4   # x\n"
        "1 4 1 4   # y\n"
        "1 4 1 4   # z\n"
        "2         # number of media\n"
        "1 0.9 0.01 1.37\n"
        "2 0.8 0.02 1.40\n"
        "2 1.5     # detectors and radius\n"
        "1 1 1\n"
        "3 3 1  2.5\n"
        "pattern   # source type\n"
        "0 2 0 2\n"
        "0 0 3 3\n"
        "inp_pat.bin\n";
    write_file("ut.inp", inp, strlen(inp));

    init_cfg(&cfg);
    cfg.seed = 0;  /* a seed given in the command line takes precedence over the file */
    mcx_readconfig((char*)"ut.inp", &cfg);
    EXPECT_EQ(cfg.nphoton, 1000, ".inp photon number");
    EXPECT_EQ(cfg.seed, 12345, ".inp seed");
    EXPECT_NEAR(cfg.srcpos.x, 1.f, 1e-6, ".inp source position is converted to 0-based");
    EXPECT_EQ(cfg.maxgate, 5, ".inp time gates");
    EXPECT_EQ(cfg.medianum, 3, ".inp media count includes medium 0");
    EXPECT_NEAR(cfg.prop[2].mua, 0.02f, 1e-7, ".inp medium properties are mus g mua n");
    EXPECT_EQ(cfg.detnum, 2, ".inp detector count");
    EXPECT_NEAR(cfg.detpos[1].w, 2.5f, 1e-6, ".inp per-detector radius");
    EXPECT_EQ(cfg.srctype, MCX_SRC_PATTERN, ".inp source type");
    EXPECT_TRUE(cfg.srcpattern && cfg.srcpattern[1] == 0.5f, ".inp pattern file");
    EXPECT_EQ(cfg.vol[0], 2, ".inp volume file");
    clear_cfg(&cfg);

    /* mcx_saveconfig writes the same format back */
    init_cfg(&cfg);
    mcx_readconfig((char*)"ut.inp", &cfg);
    mcx_writeconfig((char*)"ut2.inp", &cfg);
    char* buf;
    EXPECT_TRUE(read_file("ut2.inp", &buf) > 0, "mcx_writeconfig writes an .inp file");
    free(buf);
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  JSON input branches that the example inputs do not use
 *---------------------------------------------------------------------------*/

static void test_loadjson(void) {
    Config cfg;
    float pattern[6] = {1, 0, 0.5f, 0.5f, 0, 1};

    write_file("lj_pat.bin", pattern, sizeof(pattern));

    const char* json =
        "{\"Session\":{\"Photons\":1000,\"RNGSeed\":7,\"ID\":\"lj\",\"MaxDetPhoton\":500,\"MinEnergy\":1e-3,"
        "\"BCFlags\":\"cc_rmr001000\",\"OutputType\":\"F\",\"DebugFlag\":\"R\",\"SaveDataMask\":\"DPX\"},"
        "\"Forward\":{\"T0\":0,\"T1\":5e-9,\"Dt\":1e-9},"
        "\"Domain\":{\"Media\":[[0,0,1,1],[0.01,1,0.9,1.37]],\"OriginType\":1,\"LengthUnit\":0.5,"
        "\"CacheBoxP0\":[1,2,3],\"CacheBoxP1\":[4,5,6],"
        "\"MieScatter\":[[0.001,0.5,1.03,1.59,1.33]]},"
        "\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[8,8,8]}}],"
        "\"Optode\":{\"Source\":{\"Type\":\"pattern\",\"Pos\":[[1,1,0,1],[4,4,0,0.5]],"
        "\"Dir\":[[0,0,1,\"_NaN_\"],[0,0,1,\"_Inf_\"]],\"Param1\":[[2,0,0,2],[2,0,0,2]],"
        "\"Param2\":[[0,3,0,3],[0,3,0,3]],\"Weight\":2,\"WaveLength\":650,\"IQUV\":[1,0.5,0,0],"
        "\"Pattern\":{\"Nx\":2,\"Ny\":3,\"Data\":\"lj_pat.bin\"}},"
        "\"Detector\":[{\"Pos\":[4,4,0],\"R\":1,\"Dir\":[0,0,-1]},{\"Pos\":[2,2,0],\"R\":2,\"Dir\":[0,0,-1]}]}}";

    init_cfg(&cfg);
    cfg.maxgate = 100;  /* larger than the 5 gates of the input */
    cfg.sradius = -2.f; /* use the cache box of the input, the default of mcx but not of mcx-cl */
    mcx_readconfig((char*)json, &cfg);
    EXPECT_EQ(cfg.maxgate, 5, "maxgate is capped by the number of time gates");
    EXPECT_EQ(cfg.extrasrclen, 1, "a Nsrc x 4 Pos array adds extra sources");
    EXPECT_TRUE(isnan(cfg.srcdir.w), "a \"_NaN_\" focal length in Dir");
    EXPECT_TRUE(cfg.extrasrclen == 1 && isinf(cfg.srcdata[0].srcdir.w), "an \"_Inf_\" focal length of an extra source");
    EXPECT_NEAR(cfg.srcpos.w, 2.f, 1e-6, "Source.Weight");
    EXPECT_TRUE(cfg.srcpattern && cfg.srcpattern[2] == 0.5f, "a pattern read from a binary file");
    EXPECT_TRUE(cfg.detdir && cfg.detdir[1].z == -1.f, "Detector.Dir");
    EXPECT_TRUE(cfg.polmedianum == 1 && cfg.polprop && fabs(cfg.polprop[0].r - 0.5f) < 1e-6, "MieScatter given as an array of arrays");
    EXPECT_TRUE(cfg.crop0.x == 1 && cfg.crop1.z == 6, "CacheBoxP0/P1");
    EXPECT_EQ(cfg.maxdetphoton, 500, "MaxDetPhoton");
    EXPECT_NEAR(cfg.minenergy, 1e-3f, 1e-9, "MinEnergy");
    EXPECT_EQ(cfg.outputtype, otFluence, "an upper-case OutputType");
    EXPECT_NEAR(cfg.lambda, 650.f, 1e-6, "WaveLength");
    clear_cfg(&cfg);

    /* Domain.Step and a Volume given in the JData format */
    const char* json2 =
        "{\"Session\":{\"Photons\":10},\"Forward\":{\"T0\":0,\"T1\":1e-9,\"Dt\":1e-9},"
        "\"Domain\":{\"Media\":[{\"mua\":0,\"mus\":0,\"g\":1,\"n\":1},{\"mua\":0.1,\"mus\":1,\"g\":0,\"n\":1}],"
        "\"Dim\":[2,2,2],\"Step\":[0.5,0.5,0.5],\"MediaFormat\":\"byte\"},"
        "\"Shapes\":{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2,2,2],\"_ArrayData_\":[1,1,1,1,1,1,1,0]},"
        "\"Optode\":{\"Source\":{\"Pos\":[1,1,0],\"Dir\":[0,0,1]}}}";
    init_cfg(&cfg);
    mcx_readconfig((char*)json2, &cfg);
    EXPECT_NEAR(cfg.unitinmm, 0.5f, 1e-6, "Domain.Step sets the length unit");
    EXPECT_TRUE(cfg.vol && cfg.vol[7] == 0 && cfg.vol[0] == 1, "a JData volume in Shapes");
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  command line options, each parsed with the built-in cube60 benchmark
 *---------------------------------------------------------------------------*/

/** run mcx_parsecmd on "mcx --bench cube60" followed by the given options */
static void parse_opts(Config* cfg, const char* opts) {
    char buf[1024], *argv[64];
    int argc = 0;

    snprintf(buf, sizeof(buf), "mcx --bench cube60 %s", opts);

    for (char* tok = strtok(buf, " "); tok && argc < 63; tok = strtok(NULL, " ")) {
        argv[argc++] = tok;
    }

    argv[argc] = NULL;
    init_cfg(cfg);
    mcx_parsecmd(argc, argv, cfg);
    cfg->flog = devnull;
}

static void test_parsecmd(void) {
    Config cfg;

#ifdef MCX_OPENCL
    parse_opts(&cfg, "-a 1 -g 2 -r 3 -p 4 -e 0.001 -U 0 -u 0.5 -z 1 -H 1000 --voidtime 0 -V 1 -x 1 -X 1 -n 100");
#else
    parse_opts(&cfg, "-a 1 -g 2 -r 3 -p 4 -e 0.001 -U 0 -u 0.5 -z 1 -H 1000 -k 0 -V 1 -x 1 -X 1 -n 100");
#endif
    EXPECT_EQ(cfg.isrowmajor, 0, "-a 1: mcx_preprocess converts the row-major volume to column-major");
    EXPECT_EQ(cfg.maxgate, 1, "-g is capped by the gates of cube60");
    EXPECT_EQ(cfg.respin, 3, "-r");
    EXPECT_EQ(cfg.printnum, 4, "-p");
    EXPECT_NEAR(cfg.minenergy, 0.001f, 1e-9, "-e");
    EXPECT_EQ(cfg.isnormalized, 0, "-U");
    EXPECT_NEAR(cfg.unitinmm, 0.5f, 1e-6, "-u");
    EXPECT_EQ(cfg.issrcfrom0, 1, "-z");
    EXPECT_EQ(cfg.maxdetphoton, 1000, "-H");
    EXPECT_EQ(cfg.voidtime, 0, "-k");
    EXPECT_EQ(cfg.isspecular, 1, "-V");
    EXPECT_EQ(cfg.issaveexit, 1, "-x");
    EXPECT_EQ(cfg.issaveref, 1, "-X");
    clear_cfg(&cfg);

    parse_opts(&cfg, "-m 1 -R 2 -G 11 -I -Y 1 -K short --maxvoidstep 10 --maxjumpdebug 100 --gscatter 5 "
#ifndef MCX_OPENCL
               "--faststep 1 --reflectin 1 "
#endif
               "--atomic 1 --srcid 1 --trajstokes 1 --internalsrc 1");
    EXPECT_EQ(cfg.ismomentum, 1, "-m");
    EXPECT_TRUE(SAVE_MOM(cfg.savedetflag), "-m saves the momentum transfer");
    EXPECT_NEAR(cfg.sradius, 2.f, 1e-6, "-R");
    EXPECT_TRUE(cfg.deviceid[0] == '1' && cfg.deviceid[1] == '1', "-G with a device mask");
    EXPECT_EQ(cfg.isgpuinfo, 1, "-I");
    EXPECT_EQ(cfg.replaydet, 1, "-Y");
    EXPECT_EQ(cfg.mediabyte, 2, "-K with a format name");
    EXPECT_EQ(cfg.maxvoidstep, 10, "--maxvoidstep");
    EXPECT_EQ(cfg.maxjumpdebug, 100, "--maxjumpdebug");
    EXPECT_EQ(cfg.gscatter, 5, "--gscatter");
#ifndef MCX_OPENCL
    EXPECT_EQ(cfg.faststep, 1, "--faststep");
    EXPECT_EQ(cfg.isrefint, 1, "--reflectin");
#endif
    EXPECT_EQ(cfg.srcid, 1, "--srcid");
    EXPECT_EQ(cfg.istrajstokes, 0, "--trajstokes is cleared without trajectory and polarization output");
    EXPECT_EQ(cfg.internalsrc, 1, "--internalsrc");
    clear_cfg(&cfg);

    parse_opts(&cfg, "-G 2 -K 1 -M 2 -s utcmd -F jnii --root .");
#ifdef MCX_OPENCL
    EXPECT_TRUE(cfg.deviceid[1] == '1', "-G with a device number");
#else
    EXPECT_TRUE(cfg.gpuid == 2 && cfg.deviceid[1] == '1', "-G with a device number");
#endif
    EXPECT_EQ(cfg.mediabyte, 1, "-K with a number");
    EXPECT_EQ(cfg.isdumpmask, 2, "-M");
    EXPECT_EQ(strcmp(cfg.rootpath, "."), 0, "--root");
    clear_cfg(&cfg);

#ifdef MCX_OPENCL
    /* mcx-cl only: JIT options, a user kernel file and the device workload */
    const char* kernel = "__kernel void mcx_main_loop(){}\n";
    write_file("ut.cl", kernel, strlen(kernel));
    parse_opts(&cfg, "-o 2 -J -DUT_FLAG -k ut.cl -W 30,70 --showkernel 0 --devicelist 0");
    EXPECT_EQ(cfg.optlevel, 2, "-o");
    EXPECT_TRUE(strstr(cfg.compileropt, "-DUT_FLAG") != NULL, "-J");
    EXPECT_TRUE(cfg.clsource && strstr(cfg.clsource, "mcx_main_loop") != NULL, "-k reads the kernel file");
    EXPECT_NEAR(cfg.workload[1], 70.f, 1e-6, "-W");
    clear_cfg(&cfg);
#endif

    parse_opts(&cfg, "-l 1 -s utlog");
    EXPECT_TRUE(cfg.flog != NULL, "-l writes the log to a file");
    clear_cfg(&cfg);
    char* buf;
    EXPECT_TRUE(read_file("utlog.log", &buf) >= 0, "-l creates <session>.log");
    free(buf);

    parse_opts(&cfg, "-y zh_CN");
#ifdef MCX_OPENCL
    EXPECT_TRUE(mcx_lang != NULL, "-y selects a language");
    cJSON_Delete(mcx_lang);
    mcx_lang = NULL;
#else
    EXPECT_TRUE(cfg.langid >= 0, "-y selects a language");
#endif
    clear_cfg(&cfg);
    parse_opts(&cfg, "-y en");
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  -G/cfg.gpuid device name patterns "name" and "name:N"
 *-----------------------------------------------------------------------------*/

static void test_devicename(void) {
    const char* names[] = {"NVIDIA TITAN V", "NVIDIA GeForce RTX 4070", "NVIDIA TITAN V"};
    const char* pattern[] = {"titan", "titan:2", "4070", "TiTaN v:2", ":3", "nvidia:2"};
    const int expect[] = {0, 2, 1, 2, 2, 1};  /* the device each pattern selects */
    char mask[MAX_DEVICE], selected[3] = {1, 2, 0};
    int i, j, n = 0;

    EXPECT_TRUE(mcx_isdevicename("4070") && mcx_isdevicename("titan:2"), "a string with characters other than 0/1 is a device name");
    EXPECT_TRUE(!mcx_isdevicename("0110") && !mcx_isdevicename(""), "a 0/1 mask or an empty string is not a device name");
    memset(mask, '0', MAX_DEVICE);  /* a full mask has no terminating 0 */
    mask[1] = '1';
    EXPECT_TRUE(!mcx_isdevicename(mask), "an unterminated 0/1 mask is not a device name");
    EXPECT_TRUE(!mcx_isdevicename(selected), "the numbers of the selected devices are not a device name");
    EXPECT_TRUE(mcx_isdeviceindex("2") && mcx_isdeviceindex("-1") && mcx_isdeviceindex("256"), "-G 2, -G -1 and -G 256 are device numbers");
    EXPECT_TRUE(!mcx_isdeviceindex("4070") && !mcx_isdeviceindex("titan"), "-G 4070 and -G titan are device names");

    for (i = 0; i < 6; i++) {
        int nmatched = 0, chosen = -1;

        for (j = 0; j < 3; j++)
            if (mcx_devicenamematch(pattern[i], names[j], &nmatched)) {
                chosen = (chosen < 0) ? j : 99;
            }

        EXPECT_EQ(chosen, expect[i], pattern[i]);
    }

    EXPECT_EXIT_OK(mcx_devicenamecheck("titan:2", 2), "titan:2 with two matching devices");
    EXPECT_EXIT(mcx_devicenamecheck("titan:3", 2), "titan:3 with two matching devices is an error");
    EXPECT_EXIT(mcx_devicenamecheck("4070", 0), "a name that matches no device is an error");
    EXPECT_EXIT(mcx_devicenamematch("titan:", names[0], &n), "a ':' without a number is an error");
    EXPECT_EXIT(mcx_devicenamematch("titan:0", names[0], &n), "N after ':' must be positive");

    Config cfg;

    parse_opts(&cfg, "-G 4070");
    EXPECT_TRUE(strcmp(cfg.deviceid, "4070") == 0, "-G 4070 is a device name");
    parse_opts(&cfg, "-G titan:2");
    EXPECT_TRUE(strcmp(cfg.deviceid, "titan:2") == 0, "-G titan:2 is a device name");
    parse_opts(&cfg, "-G 2");
    EXPECT_TRUE(cfg.deviceid[0] != '1' && cfg.deviceid[1] == '1', "-G 2 is a device number");
    parse_opts(&cfg, "-G 300");
    EXPECT_TRUE(strcmp(cfg.deviceid, "300") == 0, "-G 300, beyond 256, is a device name");
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  mcx_validatecfg and mcx_replayinit, the setup path of mcxlab and pmcx
 *---------------------------------------------------------------------------*/

static void test_validatecfg(void) {
    Config cfg;
    const char* json =
        "{\"Session\":{\"Photons\":4,\"SaveDataMask\":\"DP\"},\"Forward\":{\"T0\":0,\"T1\":5e-9,\"Dt\":5e-9},"
        "\"Domain\":{\"Media\":[[0,0,1,1],[0.01,1,0.9,1.37],[0.02,1,0.9,1.37]],\"OriginType\":1},"
        "\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[6,6,6]}},{\"ZLayers\":[4,6,2]}],"
        "\"Optode\":{\"Source\":{\"Pos\":[3,3,1],\"Dir\":[0,0,1]},\"Detector\":[{\"Pos\":[3,4,1],\"R\":1},{\"Pos\":[4,3,1],\"R\":1}]}}";

    init_cfg(&cfg);
    mcx_readconfig((char*)json, &cfg);  /* OriginType 1 keeps the positions as given, as mcxlab passes them */
    cfg.issrcfrom0 = 0;
    cfg.issavedet = 1;
    cfg.sradius = 1.f;
    cfg.seed = SEED_FROM_FILE;
    cfg.nphoton = 4;
    cfg.replay.seed = calloc(4, 16);
    cfg.replaydet = 2;

    for (int i = 0; i < 4 * 16; i++) {
        ((unsigned char*)cfg.replay.seed)[i] = (unsigned char)i;
    }

    /* 4 detected photons: detector ID and partial paths in media 1 and 2 */
    float detps[4 * 3] = {1, 2, 0, 2, 1, 1, 1, 0, 0, 2, 0, 3};
    int dims[2] = {3, 4};
    float x0 = cfg.srcpos.x;

    mcx_validatecfg(&cfg, detps, dims, 16);
    EXPECT_NEAR(cfg.srcpos.x, x0 - 1.f, 1e-6, "mcx_validatecfg converts the 1-based source position");
    EXPECT_TRUE(cfg.crop0.x == 1 && cfg.crop1.x == 3 && cfg.crop1.z == 1, "mcx_validatecfg sets the cache box from sradius");
    EXPECT_NEAR(cfg.detpos[0].y, 3.f, 1e-6, "mcx_validatecfg converts the 1-based detector positions");
    EXPECT_EQ(cfg.nphoton, 2, "mcx_replayinit keeps the photons of detector 2");

    if (cfg.nphoton == 2) {
        EXPECT_NEAR(cfg.replay.weight[0], expf(-0.01f - 0.02f), 1e-6, "mcx_replayinit replay weight");
        EXPECT_EQ(((unsigned char*)cfg.replay.seed)[0], 16, "mcx_replayinit moves the seeds of the kept photons");
    }

    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  detector masks
 *---------------------------------------------------------------------------*/

static void test_maskdet(void) {
    Config cfg;
    const int formats[3] = {ofJNifti, ofBJNifti, ofNifti};
    const char* files[3] = {"utm_vol.jnii", "utm_vol.bnii", "utm_vol.nii"};

    for (int f = 0; f < 3; f++) {
        init_cfg(&cfg);
        cfg.dim.x = cfg.dim.y = cfg.dim.z = 6;
        cfg.vol = (unsigned int*)malloc(sizeof(unsigned int) * 216);

        for (int i = 0; i < 216; i++) {
            cfg.vol[i] = 1;
        }

        cfg.detnum = 1;
        cfg.detpos = (float4*)calloc(1, sizeof(float4));
        cfg.detpos[0].x = 3.f;
        cfg.detpos[0].y = 3.f;
        cfg.detpos[0].z = 0.f;
        cfg.detpos[0].w = 1.5f;
        cfg.isdumpmask = 2;
        cfg.outputformat = formats[f];
        strcpy(cfg.session, "utm");
        mcx_maskdet(&cfg);
        EXPECT_TRUE((cfg.vol[0 * 36 + 3 * 6 + 3] & DET_MASK) != 0, "mcx_maskdet marks the surface voxel below a detector");
        EXPECT_TRUE((cfg.vol[5 * 36 + 3 * 6 + 3] & DET_MASK) == 0, "mcx_maskdet leaves the far side alone");
        EXPECT_EQ(cfg.vol[1 * 36 + 3 * 6 + 3], 1, "mcx_dumpmask does not modify the volume");
        char* buf;
        EXPECT_TRUE(read_file(files[f], &buf) > 0, "mcx_dumpmask writes the mask file");
        free(buf);
        clear_cfg(&cfg);
    }

    /* split-voxel volume: a voxel with only an upper label holds background */
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 4;
    cfg.mediabyte = MEDIA_2LABEL_SPLIT;
    cfg.vol = (unsigned int*)calloc(64 * 2, sizeof(unsigned int));

    for (int i = 0; i < 64; i++) {
        cfg.vol[i] = 0x01000000u;  /* lower label 1 */
    }

    cfg.vol[1 * 4 + 1] = 0x00010000u;  /* z=0 voxel (1,1): upper label only */
    cfg.detnum = 1;
    cfg.detpos = (float4*)calloc(1, sizeof(float4));
    cfg.detpos[0].x = 1.5f;
    cfg.detpos[0].y = 1.5f;
    cfg.detpos[0].w = 1.f;
    mcx_maskdet(&cfg);
    EXPECT_TRUE((cfg.vol[1 * 4 + 1] & DET_MASK) != 0, "mcx_maskdet marks a split voxel containing background");
    clear_cfg(&cfg);

    /* a split voxel of two media (lower and upper labels) next to a background voxel */
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 4;
    cfg.mediabyte = MEDIA_2LABEL_SPLIT;
    cfg.vol = (unsigned int*)calloc(64 * 2, sizeof(unsigned int));

    for (int i = 0; i < 64; i++) {
        cfg.vol[i] = 0x01000000u;
    }

    cfg.vol[1 * 16 + 1 * 4 + 1] = 0x00010000u;  /* z=1 voxel (1,1): background */
    cfg.vol[1 * 16 + 1 * 4 + 2] = 0x01020000u;  /* z=1 voxel (2,1): media 1 and 2 */
    cfg.detnum = 1;
    cfg.detpos = (float4*)calloc(1, sizeof(float4));
    cfg.detpos[0].x = 2.5f;
    cfg.detpos[0].y = 1.5f;
    cfg.detpos[0].z = 1.5f;
    cfg.detpos[0].w = 0.6f;
    mcx_maskdet(&cfg);
    EXPECT_TRUE((cfg.vol[1 * 16 + 1 * 4 + 2] & DET_MASK) != 0, "mcx_maskdet marks a split voxel of two media next to background");
    clear_cfg(&cfg);

    /* a detector larger than the domain covers its whole surface and none of its inside */
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 4;
    cfg.vol = (unsigned int*)malloc(sizeof(unsigned int) * 64);

    for (int i = 0; i < 64; i++) {
        cfg.vol[i] = 1;
    }

    cfg.detnum = 1;
    cfg.detpos = (float4*)calloc(1, sizeof(float4));
    cfg.detpos[0].x = 2.f;
    cfg.detpos[0].y = 2.f;
    cfg.detpos[0].w = 100.f;
    mcx_maskdet(&cfg);
    {
        int nsurf = 0, ninside = 0;

        for (int k = 0; k < 4; k++)
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 4; i++) {
                    int onsurf = (i == 0 || i == 3 || j == 0 || j == 3 || k == 0 || k == 3);
                    int masked = (cfg.vol[k * 16 + j * 4 + i] & DET_MASK) != 0;
                    nsurf += onsurf && masked;
                    ninside += !onsurf && masked;
                }

        EXPECT_TRUE(nsurf == 56 && ninside == 0, "a detector larger than the domain marks all 56 surface voxels");
    }
    clear_cfg(&cfg);

    EXPECT_EXIT({
        Config c;
        init_cfg(&c);
        c.dim.x = c.dim.y = c.dim.z = 4;
        c.vol = (unsigned int*)malloc(sizeof(unsigned int) * 64);

        for (int i = 0; i < 64; i++) {
            c.vol[i] = 1;
        }

        c.detnum = 1;
        c.detpos = (float4*)calloc(1, sizeof(float4));
        c.detpos[0].x = 2e6f;
        c.detpos[0].w = 1e7f;
        mcx_maskdet(&c);
    }, "a detector placed millions of voxels away");

    /* a detector inside the volume, away from any interface, marks nothing */
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 8;
    cfg.vol = (unsigned int*)malloc(sizeof(unsigned int) * 512);

    for (int i = 0; i < 512; i++) {
        cfg.vol[i] = 1;
    }

    cfg.issavedet = 1;
    cfg.detnum = 1;
    cfg.detpos = (float4*)calloc(1, sizeof(float4));
    cfg.detpos[0].x = cfg.detpos[0].y = cfg.detpos[0].z = 4.f;
    cfg.detpos[0].w = 1.f;
    {
        FILE* olderr = stderr;
        int nmask = 0;

        stderr = devnull;  /* the "not located on an interface" warning */
        mcx_maskdet(&cfg);
        stderr = olderr;

        for (int i = 0; i < 512; i++) {
            nmask += (cfg.vol[i] & DET_MASK) != 0;
        }

        EXPECT_EQ(nmask, 0, "a detector inside the volume marks no voxel");
    }
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  --dumpjson writer: every media format must load back as the same volume
 *---------------------------------------------------------------------------*/

/** load json, dump it with --dumpjson 2 (before mcx_preprocess) and --dumpjson 1 (after),
    and check that both dumps load back as the same preprocessed volume */
static void check_dumpjson(const char* json, const char* name, int nword) {
    Config cfg, raw, dump;
    char msg[128];

    init_cfg(&cfg);
    mcx_readconfig((char*)json, &cfg);
    cfg.isdumpjson = 2;
    mcx_savejdata((char*)"sj_raw.json", &cfg);
    cfg.isdumpjson = 0;
    mcx_preprocess(&cfg);
    mcx_savejdata((char*)"sj_dump.json", &cfg);

    init_cfg(&raw);
    mcx_readconfig((char*)"sj_raw.json", &raw);
    mcx_preprocess(&raw);
    init_cfg(&dump);
    mcx_readconfig((char*)"sj_dump.json", &dump);
    mcx_preprocess(&dump);  /* converts the row-major JData volume to column-major */

    sprintf(msg, "--dumpjson 2 round trip of a %s volume", name);
    EXPECT_TRUE(raw.vol && cfg.vol && raw.mediabyte == cfg.mediabyte &&
                memcmp(cfg.vol, raw.vol, sizeof(unsigned int) * nword) == 0, msg);
    sprintf(msg, "--dumpjson round trip of a %s volume", name);
    EXPECT_TRUE(dump.vol && cfg.vol && dump.mediabyte == cfg.mediabyte &&
                memcmp(cfg.vol, dump.vol, sizeof(unsigned int) * nword) == 0, msg);
    clear_cfg(&cfg);
    clear_cfg(&raw);
    clear_cfg(&dump);
}

static void test_savejdata(void) {
    const struct {
        const char* format;
        const char* file;
        int bytes;
    } fmt[] = {
        {"byte", "v8.bin", 1}, {"short", "v16.bin", 2}, {"mua_float", "vf.bin", 4},
        {"muamus_float", "vas.bin", 8}, {"asgn_float", "vasgn.bin", 16}, {"svmc", "vsvmc.bin", 8},
        {"asgn_byte", "vab.bin", 4}, {"muamus_short", "vms.bin", 4}, {"labelplus", "vlp.bin", 4},
        {"integer", "vi32.bin", 4}
    };
    unsigned char vab[32];
    unsigned int vlp[8], vi32[8] = {0, 1, 2, 1, 1, 2, 1, 1};

    for (int i = 0; i < 32; i++) {
        vab[i] = (unsigned char)(10 + i);
    }

    for (int i = 0; i < 8; i++) {
        vlp[i] = (0x3C00u << 16) | (1u << 14) | 1u;  /* labelplus: half 1.0, property 1, label 1 */
    }

    write_file("vab.bin", vab, sizeof(vab));
    write_file("vms.bin", vab, sizeof(vab));
    write_file("vlp.bin", vlp, sizeof(vlp));
    write_file("vi32.bin", vi32, sizeof(vi32));

    for (int f = 0; f < (int)(sizeof(fmt) / sizeof(fmt[0])); f++) {
        char json[1024];

        snprintf(json, sizeof(json),
                 "{\"Session\":{\"Photons\":10,\"ID\":\"sj\"},\"Forward\":{\"T0\":0,\"T1\":1e-9,\"Dt\":1e-9},"
                 "\"Domain\":{\"Media\":[[0,0,1,1],[0.01,1,0.9,1.37],[0.03,2,0.8,1.4]],\"Dim\":[2,2,2],"
                 "\"VolumeFile\":\"%s\",\"MediaFormat\":\"%s\"},"
                 "\"Optode\":{\"Source\":{\"Pos\":[1,1,0],\"Dir\":[0,0,1]}}}", fmt[f].file, fmt[f].format);
        check_dumpjson(json, fmt[f].format, 8 * (1 + (strcmp(fmt[f].format, "asgn_float") == 0 || strcmp(fmt[f].format, "svmc") == 0)));
    }

    /* a row-major JData label volume, dumped before mcx_preprocess converts it */
    check_dumpjson("{\"Session\":{\"Photons\":10,\"ID\":\"sj\"},\"Forward\":{\"T0\":0,\"T1\":1e-9,\"Dt\":1e-9},"
                   "\"Domain\":{\"Media\":[[0,0,1,1],[0.01,1,0.9,1.37],[0.03,2,0.8,1.4]],\"Dim\":[2,2,2]},"
                   "\"Shapes\":{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2,2,2],\"_ArrayData_\":[0,1,2,1,1,2,1,1]},"
                   "\"Optode\":{\"Source\":{\"Pos\":[1,1,0],\"Dir\":[0,0,1]}}}", "JData byte", 8);

    /* the source-detector channels of Optode.SDMap are saved 1-based and read back */
    {
        Config cfg, dump;
        char json[] = "{\"Session\":{\"Photons\":10,\"ID\":\"sjmap\",\"OutputType\":\"a\"},"
                      "\"Forward\":{\"T0\":0,\"T1\":1e-9,\"Dt\":1e-9},"
                      "\"Domain\":{\"Media\":[[0,0,1,1],[0.01,1,0.9,1.37]],\"Dim\":[4,4,4]},"
                      "\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}}],"
                      "\"Optode\":{\"Source\":{\"Pos\":[2,2,0],\"Dir\":[0,0,1]},"
                      "\"Detector\":[{\"Pos\":[1,2,0],\"R\":1,\"Dir\":[0,0,1]},{\"Pos\":[3,2,0],\"R\":1,\"Dir\":[0,0,1]}],"
                      "\"SDMap\":[[1,2],[1,1]]}}";

        init_cfg(&cfg);
        mcx_readconfig(json, &cfg);
        mcx_savejdata((char*)"sjmap.json", &cfg);
        init_cfg(&dump);
        mcx_readconfig((char*)"sjmap.json", &dump);
        EXPECT_TRUE(dump.sdmaplen == 2 && dump.sdmap && memcmp(dump.sdmap, cfg.sdmap, 4 * sizeof(unsigned int)) == 0,
                    "--dumpjson saves and restores Optode.SDMap");
        clear_cfg(&cfg);
        clear_cfg(&dump);
    }
}

/*-----------------------------------------------------------------------------
 *  helpers for the error tests: JSON written with single quotes for readability
 *---------------------------------------------------------------------------*/

/** return a copy of str with single quotes turned into double quotes */
static char* jq(const char* str) {
    char* out = strdup(str);

    for (char* p = out; *p; p++) {
        if (*p == '\'') {
            *p = '"';
        }
    }

    return out;
}

#define JBASE_SESSION "'Session':{'Photons':10,'ID':'e'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}"
#define JBASE_DOMAIN  "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4]}"
#define JBASE_SHAPES  "'Shapes':[{'Grid':{'Tag':1,'Size':[4,4,4]}}]"
#define JBASE_OPTODE  "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]}}"

/** load a JSON input (single-quoted), as mcx -f does; optionally run mcx_preprocess */
static void load_json(Config* cfg, const char* json, int prep) {
    char* buf = jq(json);
    init_cfg(cfg);
    mcx_readconfig(buf, cfg);

    if (prep) {
        mcx_preprocess(cfg);
    }

    free(buf);
}

static void load_prep(const char* json) {
    Config cfg;
    load_json(&cfg, json, 1);
}

/** parse "mcx <opts>" without the --bench prefix */
static void parse_raw(const char* opts) {
    Config cfg;
    char buf[2048], *argv[64];
    int argc = 0;

    snprintf(buf, sizeof(buf), "mcx %s", opts);

    for (char* tok = strtok(buf, " "); tok && argc < 63; tok = strtok(NULL, " ")) {
        argv[argc++] = tok;
    }

    argv[argc] = NULL;
    init_cfg(&cfg);
    mcx_parsecmd(argc, argv, &cfg);
}

#define EXPECT_JSON_ERROR(json, msg) EXPECT_EXIT(load_prep(json), msg)

/*-----------------------------------------------------------------------------
 *  error reporting of the JSON reader and mcx_preprocess
 *---------------------------------------------------------------------------*/

static void test_json_errors(void) {
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1]],'Dim':[4,4,4],'MediaFormat':'foo'}," JBASE_SHAPES "," JBASE_OPTODE "}", "unsupported MediaFormat");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':5,'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "Media is not an array");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'MieScatter':[5],'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "MieScatter is not an array");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'VolumeFile':'nosuch.bin'}," JBASE_OPTODE "}", "no Dim for a volume file");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'Step':[1,1]}," JBASE_SHAPES "," JBASE_OPTODE "}", "Step with 2 elements");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'Step':[1,2,1]}," JBASE_SHAPES "," JBASE_OPTODE "}", "anisotropic Step");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'InverseCDF':[0.5,0.2]}," JBASE_SHAPES "," JBASE_OPTODE "}", "decreasing InverseCDF");
#ifndef MCX_OPENCL
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VoxelSize':{'Dx':[1,1]}}," JBASE_SHAPES "," JBASE_OPTODE "}", "VoxelSize.Dx length");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VoxelSize':{'Dy':[1,1]}}," JBASE_SHAPES "," JBASE_OPTODE "}", "VoxelSize.Dy length");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VoxelSize':{'Dz':[1,1]}}," JBASE_SHAPES "," JBASE_OPTODE "}", "VoxelSize.Dz length");
#endif
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'CacheBoxP0':[1,2]}," JBASE_SHAPES "," JBASE_OPTODE "}", "CacheBoxP0 length");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'CacheBoxP1':[1]}," JBASE_SHAPES "," JBASE_OPTODE "}", "CacheBoxP1 length");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[1,1,0],[2,2,0]],'Dir':[[0,0,1],[0,0,1],[0,0,1]]}}}", "Dir rows differ from Pos rows");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Dir':[[0,0,1],[0,0,1]],'Pos':[[1,1,0],[2,2,0],[3,3,0]]}}}", "Pos rows differ from Dir rows");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[1,1,0],[2,2,0]],'Param1':[[1,0,0,0],[1,0,0,0],[1,0,0,0]]}}}", "Param1 rows differ");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[1,1,0],[2,2,0]],'Param2':[[1,0,0,0],[1,0,0,0],[1,0,0,0]]}}}", "Param2 rows differ");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'Type':'pattern','Param1':[2,0,0,2],'Param2':[0,2,0,2],"
                      "'Pattern':{'_ArrayType_':'uint8','_ArraySize_':[2,2],'_ArrayData_':[1,1,1,1]}}}}", "a non-single JData pattern");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'Type':'pattern','Param1':[2,0,0,2],'Param2':[0,2,0,2],"
                      "'Pattern':{'Nx':2,'Ny':2,'Data':[1,1]}}}}", "incomplete pattern data");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'AngleInverseCDF':[0.5,0.1]}}}", "decreasing AngleInverseCDF");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'OutputFormat':'xyz'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "unknown OutputFormat");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'OutputType':'z'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "unknown OutputType");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10},'Forward':{'T0':2e-9,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "T0 > T1");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN ",'Shapes':[{'Grid':{'Tag':1,'Size':[4,4,4]}},{'Sphere':{'Tag':2}}]," JBASE_OPTODE "}", "a Sphere without O and R");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_OPTODE "}", "neither a volume file nor shapes");
    write_file("e8.bin", "\x01\x01\x01\x01\x01\x01\x01\x01", 8);
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[2,2,2],'VolumeFile':'e8.bin'},"
                      "'Shapes':{'_ArrayType_':'uint8','_ArraySize_':[2,2,2],'_ArrayData_':[1,1,1,1,1,1,1,1]}," JBASE_OPTODE "}", "both a volume file and a JData volume");

    /* mcx_preprocess: optical properties, voxel size, roulette threshold and workloads that would
       have the photons step forever */
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[-0.01,1,0.9,1.37]],'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "a negative mua");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,1.5,1.37]],'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "an anisotropy beyond 1");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'LengthUnit':1e-6}," JBASE_SHAPES "," JBASE_OPTODE "}", "a voxel size below 0.1 um");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'MinEnergy':1},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "a roulette threshold of 1");
    EXPECT_EXIT({Config c; load_json(&c, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", 0); c.workload[0] = -1.f; mcx_preprocess(&c);}, "a negative device workload");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'OutputType':'a'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},"
                      "'Detector':[{'Pos':[1,2,0],'R':1,'Dir':[0,0,1]}],'SDMap':[[1,2]]}}", "an Optode.SDMap detector beyond the detectors");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[2,2,0],[1,1,0]],'Dir':[0,0,1],'Type':'pattern','Param1':[2,0,0,2],'Param2':[0,2,0,2],"
                      "'SrcNum':2,'Pattern':{'Nx':2,'Ny':2,'Nz':1,'Data':[1,1,1,1,1,1,1,1]}}}}", "several sources and a multi-pattern source");

    /* the replay of every detector (-Y -1) with a single detector replays that detector */
    {
        Config c;
        load_json(&c, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},'Detector':[{'Pos':[1,2,0],'R':1}]}}", 0);
        c.replaydet = -1;
        mcx_preprocess(&c);
        EXPECT_EQ(c.replaydet, 1, "-Y -1 with one detector replays detector 1");
        clear_cfg(&c);
    }

    /* mcx_preprocess */
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,0]}}}", "a zero-length source direction");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'OutputType':'J'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "a Jacobian outside of the replay mode");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'OutputType':'R'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "RF output without omega");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'BCFlags':'xxxxxx'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "an unknown boundary condition");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'BCFlags':'______x'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "an unknown boundary detection flag");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'MieScatter':[[0.001,0.5,1.03,1.59,1.33],[0.001,0.5,1.03,1.59,1.33]],'Dim':[4,4,4]}," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'WaveLength':650}}}", "MieScatter count != media count");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'MieScatter':[[0.001,0.5,1.03,1.59,1.33]],'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "polarized without a wavelength");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'MieScatter':[[0.001,0.5,1.03,1.59,1.33]],'Dim':[4,4,4]}," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'WaveLength':650,'IQUV':[-1,0,0,0]}}}", "a negative I");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'MieScatter':[[0.001,0.5,1.03,1.59,1.33]],'Dim':[4,4,4]}," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'WaveLength':650,'IQUV':[1,2,0,0]}}}", "Q out of [-1,1]");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "no media");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'Type':'pattern','Param1':[2,0,0,2],'Param2':[0,2,0,2]}}}", "a pattern source without a pattern");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[2,2,0],[1,1,0]],'Dir':[0,0,1],'ID':5}}}", "srcid beyond the source count");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN ",'Shapes':[{'Grid':{'Tag':3,'Size':[4,4,4]}}]," JBASE_OPTODE "}", "a label beyond the media count");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'DoSaveRef':2,'DoPartialPath':0},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "DoSaveRef 2 without detected photons");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1]],'Dim':[2,2,2],'VolumeFile':'vf.bin','MediaFormat':'mua_float'}," JBASE_OPTODE "}", "mua_float with one medium");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[2,2,2],'VolumeFile':'vab.bin','MediaFormat':'asgn_byte'}," JBASE_OPTODE "}", "asgn_byte with two media");

    {
        /* more media, detectors and sources than MAX_PROP_AND_DETECTORS */
        char* buf = (char*)malloc(200000), *p = buf;
        p += sprintf(p, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},'Detector':[");

        for (int i = 0; i < MAX_PROP_AND_DETECTORS + 1; i++) {
            p += sprintf(p, "%s{'Pos':[1,1,0],'R':1}", i ? "," : "");
        }

        sprintf(p, "]}}");
        EXPECT_JSON_ERROR(buf, "too many detectors");
        free(buf);
    }

    /* mcx_readconfig and mcx_writeconfig */
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_readconfig((char*)"nosuch.json", &c);}, "a missing input file");
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_readconfig((char*)"{\"bad", &c);}, "an invalid JSON input");
    write_file("bad.json", "{\"Session\":", 11);
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_readconfig((char*)"bad.json", &c);}, "an invalid JSON file");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'RootPath':'/proc/nosuch/dir'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "an output folder that can not be created");
    EXPECT_EXIT({Config c; load_json(&c, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", 0); mcx_writeconfig((char*)"/proc/nosuch/x.inp", &c);}, "writing an .inp file to a missing folder");
}

/*-----------------------------------------------------------------------------
 *  error reporting of the command line parser
 *---------------------------------------------------------------------------*/

static void test_cmd_errors(void) {
    EXPECT_EXIT(parse_raw("--nosuchoption 1"), "an unknown long option");
    EXPECT_EXIT(parse_raw("-% 1"), "an unknown short option");
    EXPECT_EXIT(parse_raw("-f a.json -i"), "-i together with -f");
    EXPECT_EXIT(parse_raw("--bench cube60 -P"), "-P without shapes");
    EXPECT_EXIT(parse_raw("--bench cube60 --json"), "--json without a fragment");
    EXPECT_EXIT(parse_raw("--bench cube60 -O z"), "an unknown -O type");
    EXPECT_EXIT(parse_raw("--bench cube60 -K foo"), "an unknown -K format");
    EXPECT_EXIT(parse_raw("--bench cube60 -F"), "-F without a value");
    EXPECT_EXIT(parse_raw("--bench cube60 -F xyz"), "an unknown -F format");
    EXPECT_EXIT(parse_raw("--bench nosuchbench"), "an unknown benchmark");
    EXPECT_EXIT(parse_raw("--bench cube60 -O J"), "-O J without replay");
    EXPECT_EXIT(parse_raw("--bench cube60 --json {bad"), "an invalid --json fragment");
    EXPECT_EXIT(parse_raw("--bench cube60 -n"), "-n without a value");
    EXPECT_EXIT(parse_raw("--bench cube60 -r 0"), "-r 0");
    EXPECT_EXIT(parse_raw("--bench cube60 -Y 5 -E 1"), "-Y beyond the detector count");
    EXPECT_EXIT(parse_raw("-f nosuch.inp"), "a missing .inp file");
}

/*-----------------------------------------------------------------------------
 *  error reporting of the file readers, writers and mcxlab/pmcx entry points
 *---------------------------------------------------------------------------*/

static void test_io_errors(void) {
    Config cfg;

    /* volume files */
    EXPECT_EXIT({Config c; init_cfg(&c); c.dim.x = c.dim.y = c.dim.z = 2; mcx_loadvolume((char*)"nosuch.bin", &c, 0);}, "a missing volume file");
    write_file("short.bin", "\x01\x01", 2);
    EXPECT_EXIT({Config c; init_cfg(&c); c.dim.x = c.dim.y = c.dim.z = 2; c.medianum = 2; mcx_loadvolume((char*)"short.bin", &c, 0);}, "a volume file that is too short");
    write_file("big.bin", "\x05\x05\x05\x05\x05\x05\x05\x05", 8);
    EXPECT_EXIT({Config c; init_cfg(&c); c.dim.x = c.dim.y = c.dim.z = 2; c.medianum = 2; mcx_loadvolume((char*)"big.bin", &c, 0);}, "a label beyond the media count");
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_loadvolume((char*)"bad.json", &c, 0);}, "an invalid shape file as a volume");

    /* writers: a session in a missing folder */
    setup_output_cfg(&cfg);
    strcpy(cfg.session, "/proc/nosuch/ut");
    float dat[48] = {0};
    const int formats[] = {ofMC2, ofNifti, ofAnalyze, ofJNifti, ofBJNifti};

    for (int i = 0; i < 5; i++) {
        cfg.outputformat = formats[i];
        EXPECT_EXIT(mcx_savedata(dat, 24, &cfg), "saving to a missing folder");
    }

    EXPECT_EXIT(mcx_savenii(dat, 24, (char*)"ut", NIFTI_TYPE_FLOAT32, ofMC2, &cfg), "mcx_savenii with an unsupported format");
    EXPECT_EXIT({Config c; load_json(&c, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", 0);
                 mcx_savejdata((char*)"/proc/nosuch/x.json", &c);}, "mcx_savejdata to a missing folder");

    cfg.his.colcount = 2;
    cfg.his.savedphoton = 1;
    cfg.his.detected = 1;
    cfg.medianum = 2;
    cfg.prop = (Medium*)calloc(2, sizeof(Medium));
    float ppath[2] = {1, 1};
    unsigned char seed[16] = {0};
    cfg.his.seedbyte = 16;
    cfg.issaveseed = 1;
    cfg.outputformat = ofMC2;
    EXPECT_EXIT(mcx_savedetphoton(ppath, seed, 1, 0, &cfg), ".mch to a missing folder");
    cfg.outputformat = ofJNifti;
    EXPECT_EXIT(mcx_savedetphoton(ppath, seed, 1, 0, &cfg), ".jdt to a missing folder");
    cfg.outputformat = ofBJNifti;
    EXPECT_EXIT(mcx_savedetphoton(ppath, seed, 1, 0, &cfg), ".jdb to a missing folder");
    clear_cfg(&cfg);

    /* seed files */
    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "nosuch.mch"); mcx_loadseedfile(&c);}, "a missing .mch file");
    write_file("trunc.mch", "MCXH", 4);
    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "trunc.mch"); mcx_loadseedfile(&c);}, "a truncated .mch header");
    {
        History his;
        memset(&his, 0, sizeof(his));
        his.maxmedia = 1;
        his.colcount = 3;
        write_file("noseed.mch", &his, sizeof(his));
        his.savedphoton = 2;
        his.seedbyte = 16;
        write_file("media.mch", &his, sizeof(his));
    }
    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "noseed.mch"); c.medianum = 2; mcx_loadseedfile(&c);}, "an .mch file without seeds");
    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "media.mch"); c.medianum = 5; mcx_loadseedfile(&c);}, "an .mch file with other media");
    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "media.mch"); c.medianum = 2; mcx_loadseedfile(&c);}, "an .mch file without photon data");

    if (system("cp ut.mch ut_nopp.mch") != 0) {
        printf("warning: can not copy ut.mch\n");
    }

    EXPECT_EXIT({Config c; init_cfg(&c); strcpy(c.seedfile, "ut_nopp.mch"); c.medianum = 3; c.outputtype = otJacobian; c.detnum = 2;
    FILE* fp = fopen("ut_nopp.mch", "r+b"); History h; if (fp && fread(&h, sizeof(h), 1, fp) == 1) {
    h.savedetflag = 0;
    rewind(fp);
        fwrite(&h, sizeof(h), 1, fp);
    }
    if (fp) {
    fclose(fp);
    }
    mcx_loadseedfile(&c);}, "replay from an .mch file without partial paths");
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_loadseedjdat((char*)"nosuch.jdt", &c);}, "a missing .jdt file");
    write_file("nodet.jdt", "{\"MCXData\":{}}", 14);
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_loadseedjdat((char*)"nodet.jdt", &c);}, "a .jdt file without photon data");
    write_file("notjd.jdt", "[1,2]", 5);
    EXPECT_EXIT({Config c; init_cfg(&c); mcx_loadseedjdat((char*)"notjd.jdt", &c);}, "a .jdt file without MCXData");

    /* legacy .inp files */
    const char* inp_head = "1000\n12345\n2 2 1\n0 0 1\n";
    char inp[1024];
    snprintf(inp, sizeof(inp), "%s5e-9 0 1e-9\ninp.bin\n1 4 1 4\n1 4 1 4\n1 4 1 4\n1\n1 0.9 0.01 1.37\n0 1\n", inp_head);
    write_file("e1.inp", inp, strlen(inp));
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e1.inp", &c);}, ".inp with T0 > T1");
    snprintf(inp, sizeof(inp), "%s0 5e-9 1e-9\ninp.bin\n1 4 1 4\n2 4 1 4\n1 4 1 4\n1\n1 0.9 0.01 1.37\n0 1\n", inp_head);
    write_file("e2.inp", inp, strlen(inp));
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e2.inp", &c);}, ".inp with anisotropic voxels");
    snprintf(inp, sizeof(inp), "%s0 5e-9 1e-9\ninp.bin\n1 4 1 4\n1 4 1 4\n1 4 1 4\n1\n1 0.9 0.01 1.37\n0 1\nnosuchsrc\n", inp_head);
    write_file("e3.inp", inp, strlen(inp));
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e3.inp", &c);}, ".inp with an unknown source type");
    snprintf(inp, sizeof(inp), "%s0 5e-9 1e-9\ninp.bin\n1 4 1 4\n1 4 1 4\n1 4 1 4\n1\n1 0.9 0.01 1.37\n0 1\npattern\n0 2 0 2\n0 0 2 2\nnosuch_pat.bin\n", inp_head);
    write_file("e4.inp", inp, strlen(inp));
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e4.inp", &c);}, ".inp with a missing pattern file");
    snprintf(inp, sizeof(inp), "%s0 5e-9 1e-9\ninp.bin\n1 4 1 4\n1 4 1 4\n1 4 1 4\n1\n1 0.9 0.01 1.37\n0 1\npattern3d\n2 2 2 0\n0 0 0 0\nnosuch_pat.bin\n", inp_head);
    write_file("e5.inp", inp, strlen(inp));
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e5.inp", &c);}, ".inp with a missing pattern3d file");
    write_file("e6.inp", "1000\n", 5);
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = 0; mcx_readconfig((char*)"e6.inp", &c);}, "a truncated .inp file");

    /* mcxlab/pmcx entry points */
    EXPECT_EXIT({Config c; init_cfg(&c); c.seed = SEED_FROM_FILE; mcx_replayinit(&c, NULL, NULL, 0);}, "replay without detected photons");
    {
        float detps[6] = {0};
        int dims[2] = {3, 2};
        EXPECT_EXIT({Config c; init_cfg(&c); c.seed = SEED_FROM_FILE; c.nphoton = 5; mcx_replayinit(&c, detps, dims, 16);}, "replay with a mismatched photon count");
        EXPECT_EXIT({Config c; init_cfg(&c); c.seed = SEED_FROM_FILE; c.nphoton = 2; mcx_replayinit(&c, detps, dims, 0);}, "replay without seeds");
        EXPECT_EXIT({Config c; init_cfg(&c); c.seed = SEED_FROM_FILE; c.nphoton = 2; c.detnum = 2; c.savedetflag = 0x4; mcx_replayinit(&c, detps, dims, 16);}, "replay without detector IDs");
    }

    const char* vjson = "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}";
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.mediabyte = 3; mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg with an unsupported media format");
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.tstart = 1e-9f; mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg with T0 >= T1");
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.srcdir.z = 2.f; mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg with a non-unit srcdir");
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.steps.y = 0.f; mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg with a zero voxel size");
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.mediabyte = MEDIA_MUA_FLOAT; c.shapedata = strdup("{\"Shapes\":[]}"); mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg rasterizing shapes on a continuous volume");
    EXPECT_EXIT({Config c; load_json(&c, vjson, 0); c.shapedata = strdup("{\"Shapes\":[{\"Sphere\":{}}]}"); mcx_validatecfg(&c, NULL, NULL, 0);}, "mcx_validatecfg with an invalid shape");

    /* JData volumes */
    EXPECT_EXIT({Config c; init_cfg(&c); void* v = NULL; char* t = NULL; int nd; uint d[3];
                 cJSON* o = cJSON_Parse("{\"_ArrayType_\":\"double\",\"_ArraySize_\":[2],\"_ArrayData_\":[1,2]}"); mcx_jdatadecode(&v, &nd, d, 3, &t, o, &c);}, "a double JData volume");
    EXPECT_EXIT({Config c; init_cfg(&c); void* v = NULL; char* t = NULL; int nd; uint d[3];
                 cJSON* o = cJSON_Parse("{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2],\"_ArrayData_\":[[1],[2]]}"); mcx_jdatadecode(&v, &nd, d, 3, &t, o, &c);}, "a nested JData array");
    EXPECT_EXIT({Config c; init_cfg(&c); void* v = NULL; char* t = NULL; int nd; uint d[3];
                 cJSON* o = cJSON_Parse("{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2]}"); mcx_jdatadecode(&v, &nd, d, 3, &t, o, &c);}, "a JData object without data");

    /* --dumpmask 1 and the help text exit with status 0 */
    {
        int status = 0;
        pid_t pid = fork();

        if (pid == 0) {
            if (freopen("/dev/null", "w", stdout)) {}

            parse_raw("-h");
            exit(1);
        }

        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0, "-h prints the help and exits");
    }
}

/*-----------------------------------------------------------------------------
 *  more of the normal code paths
 *---------------------------------------------------------------------------*/

static void test_more_paths(void) {
    Config cfg;
    char* buf;

    /* the language set through MCX_LANG, and a translated string */
    setenv("MCX_LANG", "zh_CN", 1);
    init_cfg(&cfg);
    EXPECT_TRUE(strlen(T_("simulated")) > 0, "T_ with MCX_LANG set");
    clear_cfg(&cfg);
    setenv("MCX_LANG", "xx_YY", 1);
    init_cfg(&cfg);
    clear_cfg(&cfg);
    unsetenv("MCX_LANG");
#ifdef MCX_OPENCL
    cJSON_Delete(mcx_lang);
#endif
    mcx_lang = NULL;

    init_cfg(&cfg);
    mcx_printlog(&cfg, (char*)"mcx_printlog");
    clear_cfg(&cfg);

    /* a single-row Dir/Param1/Param2 applies to every source of a multi-row Pos */
    load_json(&cfg, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[2,2,0],[1,1,0]],'Dir':[0,1,0],'Param1':[1,2,3,4],'Param2':[5,6,7,8]}}}", 0);
    EXPECT_TRUE(cfg.extrasrclen == 1 && cfg.srcdata[0].srcdir.y == 1.f && cfg.srcdata[0].srcparam1.w == 4.f && cfg.srcdata[0].srcparam2.x == 5.f,
                "a single-row Dir/Param applies to the extra sources");
    clear_cfg(&cfg);

    /* a nested output folder, VoxelSize and the adjoint detector sources */
    load_json(&cfg, "{'Session':{'Photons':10,'ID':'nest','RootPath':'r1/r2/r3','OutputType':'a','DebugFlag':'T'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
              "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VoxelSize':{'Dx':[1,1,1,1],'Dy':[1],'Dz':[1,1,1,1]}}," JBASE_SHAPES ","
              "'Optode':{'Source':{'Pos':[[2,2,0],[-1,2,0]],'Dir':[[0,0,1],[0,0,1]],'Weight':0},'Detector':[{'Pos':[2,3,0],'R':1,'Dir':[0,0,1]}]}}", 1);
    EXPECT_TRUE(access("r1/r2/r3", F_OK) == 0, "RootPath creates nested folders");
    EXPECT_EQ(cfg.srcid, -1, "the adjoint output simulates each source and detector slot");
    EXPECT_EQ(cfg.extrasrclen, 2, "the detectors are appended as adjoint sources");
#ifndef MCX_OPENCL
    EXPECT_TRUE(cfg.dx && cfg.dz, "VoxelSize allocates the per-axis voxel sizes");
#endif
    clear_cfg(&cfg);

    /* replay output layout with multiple detectors */
    setup_output_cfg(&cfg);
    float dat[192] = {0};  /* 3x4x2 voxels x 2 gates x 2 detectors x 2 (RF) */
    cfg.seed = SEED_FROM_FILE;
    cfg.replaydet = -1;
    cfg.detnum = 2;
    cfg.outputtype = otRFmus;
    cfg.outputformat = ofJNifti;
    mcx_savedata(dat, 24, &cfg);
    EXPECT_TRUE(read_file("ut.jnii", &buf) > 0, "replay output for all detectors");
    free(buf);
    cfg.extrasrclen = 1;
    cfg.srcid = -1;
    cfg.seed = 1;
    cfg.outputtype = otFlux;
    cfg.outputformat = ofBJNifti;
    mcx_savedata(dat, 24, &cfg);
    EXPECT_TRUE(read_file("ut.bnii", &buf) > 0, "output of each source slot");
    free(buf);
    clear_cfg(&cfg);

    /* trajectory files: his.detected == 0 marks the records as trajectory steps */
    setup_output_cfg(&cfg);
    float traj[3 * 9];

    for (int i = 0; i < 27; i++) {
        traj[i] = (float)i;
    }

    cfg.medianum = 2;
    cfg.prop = (Medium*)calloc(2, sizeof(Medium));
    cfg.his.detected = 0;
    cfg.his.savedphoton = 3;
    cfg.his.colcount = 9;
    cfg.outputformat = ofJNifti;
    mcx_savedetphoton(traj, NULL, 3, 0, &cfg);
    EXPECT_TRUE(read_file("ut_traj.jdt", &buf) > 0 && has_key(buf, strlen(buf), "Trajectory"), ".jdt trajectory output");
    free(buf);
    cfg.outputformat = ofBJNifti;
    mcx_savedetphoton(traj, NULL, 3, 0, &cfg);
    long len = read_file("ut_traj.jdb", &buf);
    EXPECT_TRUE(len > 0 && has_key(buf, len, "Trajectory"), ".jdb trajectory output");
    free(buf);
    cfg.outputformat = ofMC2;
    mcx_savedetphoton(traj, NULL, 3, 1, &cfg);
    EXPECT_TRUE(read_file("ut.mct", &buf) > 0, ".mct trajectory output, appended");
    free(buf);
    clear_cfg(&cfg);

    /* typed _ArrayData_ buffers of BJData (base64 means no compression) */
    {
        const char* types[] = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "single", "double"};
        const int bytes[] = {1, 1, 2, 2, 4, 4, 8, 8, 4, 8};
        unsigned char data[64] = {1, 2, 3, 4};
        unsigned char out[4096];
        uint dims[2] = {2, 2};

        init_cfg(&cfg);
        cfg.isdumpjson = 1;

        for (int i = 0; i < 10; i++) {
            ubjw_context_t* ctx = ubjw_open_memory(out, out + sizeof(out));
            ubjw_begin_object(ctx, UBJ_MIXED, 0);
            int ret = mcx_jdataencode(data, 2, dims, (char*)types[i], bytes[i], zmBase64, ctx, 1, 1, &cfg);
            ubjw_end(ctx);
            size_t n = ubjw_close_context(ctx);
            EXPECT_TRUE(ret == 0 && n > 0, "a typed BJData _ArrayData_ buffer");
        }

        clear_cfg(&cfg);
    }

    /* JData numeric arrays of 2-, 4- and 8-byte volumes */
    {
        const char* arr[] = {"{\"_ArrayType_\":\"uint16\",\"_ArraySize_\":[2],\"_ArrayData_\":[300,2]}",
                             "{\"_ArrayType_\":\"uint32\",\"_ArraySize_\":[2],\"_ArrayData_\":[70000,2]}",
                             "{\"_ArrayType_\":\"uint64\",\"_ArraySize_\":[2],\"_ArrayData_\":[1,2]}"
                            };
        const int mb[] = {2, 4, MEDIA_2LABEL_SPLIT};

        for (int i = 0; i < 3; i++) {
            void* v = NULL;
            char* t = NULL;
            int nd = 0;
            uint d[3] = {0};
            init_cfg(&cfg);
            cfg.mediabyte = mb[i];
            cJSON* o = cJSON_Parse(arr[i]);
            mcx_jdatadecode(&v, &nd, d, 3, &t, o, &cfg);
            EXPECT_TRUE(v != NULL && d[0] == 2, "a numeric JData array of a wider type");
            free(v);
            cJSON_Delete(o);
            clear_cfg(&cfg);
        }
    }

    /* an integer volume, NaN values in muamus/asgn float volumes */
    init_cfg(&cfg);
    cfg.dim.x = cfg.dim.y = cfg.dim.z = 2;
    cfg.medianum = 3;
    cfg.prop = (Medium*)calloc(3, sizeof(Medium));
    unsigned int iv[8] = {0, 1, 2, 1, 0, 1, 2, 1};
    write_file("v32.bin", iv, sizeof(iv));
    cfg.mediabyte = 4;
    mcx_loadvolume((char*)"v32.bin", &cfg, 0);
    EXPECT_EQ(cfg.vol[2], 2, "integer volume");
    float fn[32];

    for (int i = 0; i < 32; i++) {
        fn[i] = (i == 2 || i == 9) ? NAN : 0.1f;
    }

    write_file("vnan.bin", fn, sizeof(fn));
    cfg.mediabyte = MEDIA_AS_F2H;
    mcx_loadvolume((char*)"vnan.bin", &cfg, 0);
    EXPECT_EQ(cfg.vol[1], 0, "a NaN mua in muamus_float becomes a 0-label voxel");
    cfg.mediabyte = MEDIA_ASGN_F2H;
    mcx_loadvolume((char*)"vnan.bin", &cfg, 0);
    EXPECT_EQ(cfg.vol[2], 0, "a NaN mus in asgn_float becomes a 0-label voxel");
    clear_cfg(&cfg);

    /* .inp read from the console (stdin), with its prompts */
    {
        int status = 0;
        pid_t pid = fork();

        if (pid == 0) {
            Config c;

            if (freopen("ut.inp", "r", stdin) == NULL || freopen("/dev/null", "w", stdout) == NULL) {
                exit(2);
            }

            init_cfg(&c);
            c.seed = 0;
            mcx_readconfig((char*)"", &c);
            exit(c.nphoton == 1000 ? 0 : 3);
        }

        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0, "an .inp file read from stdin");
    }

    /* shapes: single-triplet layers, reversed bounds, malformed shapes and files */
    {
        unsigned int* vol = NULL;
        GridDim dim = {0};
        Grid3D g = {&vol, &dim, {0.f, 0.f, 0.f}, 0};
        int status = mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}},{\"XLayers\":[3,1,2]},{\"YLayers\":[[4,3,5]]},"
                                           "{\"ZSlabs\":{\"Tag\":6,\"Bound\":[[3,1]]}},{\"XSlabs\":{\"Tag\":7,\"Bound\":[]}},{\"YSlabs\":{\"Tag\":7}}]}");
        EXPECT_TRUE(status != 0, "a Slabs without Bound is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}},{\"XLayers\":[3,1,2]},{\"YLayers\":[[4,3,5]]},{\"ZSlabs\":{\"Tag\":6,\"Bound\":[[3,1]]}}]}") == 0,
                    "single-triplet layers and reversed bounds");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}},{\"XLayers\":[[1,2]]}]}") != 0, "layers with pairs are an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"NoSuchShape\":{}}]}") != 0, "an unknown shape is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}}, oops]}") != 0, "invalid shape JSON is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Box\":{\"Tag\":1}}]}") != 0, "a Box without O is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Box\":{\"O\":[1,1,1]}}]}") != 0, "a Box without Size is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Subgrid\":{\"Tag\":1}}]}") != 0, "a Subgrid without O is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Subgrid\":{\"O\":[1,1,1]}}]}") != 0, "a Subgrid without Size is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Sphere\":{\"O\":[1,1,1]}}]}") != 0, "a Sphere without R is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Cylinder\":{\"Tag\":1}}]}") != 0, "a Cylinder without C0 is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Cylinder\":{\"C0\":[1,1,1]}}]}") != 0, "a Cylinder without C1 is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Cylinder\":{\"C0\":[1,1,1],\"C1\":[1,1,3]}}]}") != 0, "a Cylinder without R is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Grid\":{\"Tag\":1}}]}") != 0, "a Grid without Size is an error");
        EXPECT_TRUE(mcx_parse_shapestring(&g, (char*)"{\"Shapes\":[{\"Origin\":[1,1]}]}") != 0, "an Origin with 2 elements is an error");
        EXPECT_TRUE(mcx_load_jsonshapes(&g, (char*)"nosuch_shapes.json") != 0, "a missing shape file is an error");
        EXPECT_TRUE(mcx_load_jsonshapes(&g, (char*)"bad.json") != 0, "an invalid shape file is an error");
        free(vol);
    }
}

/*-----------------------------------------------------------------------------
 *  --dumpjson of an input that sets every optional field, and reloading a config
 *---------------------------------------------------------------------------*/

#define JRICH \
    "{'Session':{'Photons':10,'ID':'rich','RootPath':'r1','DoMismatch':0,'DoNormalize':0,'DoSaveRef':1,'DoPartialPath':1," \
    "'MinEnergy':1e-3,'BCFlags':'cc_rmr001000'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," \
    "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'MieScatter':[{'mua':0.001,'radius':0.5,'rho':1.03,'nsph':1.59,'nmed':1.33}]," \
    "'InverseCDF':[-0.5,0,0.5]}," JBASE_SHAPES "," \
    "'Optode':{'Source':{'Type':'pattern','Pos':[[2,2,0,2],[1,1,0,1]],'Dir':[[0,0,1,'-_Inf_'],[0,0,1,'_NaN_']],'Param1':[[2,0,0,2],[2,0,0,2]]," \
    "'Param2':[[0,2,0,2],[0,2,0,2]],'WaveLength':650,'IQUV':[1,0,0,0],'Frequency':1e8,'AngleInverseCDF':[0.1,0.5,0.9],'ID':1," \
    "'Pattern':{'Nx':2,'Ny':2,'Data':[1,0.5,0.5,1]}},'Detector':[{'Pos':[2,3,0],'R':1}]}}"

#ifdef MCX_OPENCL
    char* print_cl_errstring(int err);  /* cl_int is a 32-bit int */
#endif

/** the OpenCL error strings of mcx-cl */
static void test_clerrors(void) {
#ifdef MCX_OPENCL

    for (int err = 1; err >= -72; err--) {
        EXPECT_TRUE(print_cl_errstring(err) != NULL, "print_cl_errstring");
    }

    for (int err = -1000; err >= -1010; err--) {
        EXPECT_TRUE(print_cl_errstring(err) != NULL, "print_cl_errstring of an extension error");
    }

#endif
}

static void test_rich_dump(void) {
    Config cfg, cfg2;
    char* buf;

    if (system("mkdir -p r1") != 0) {
        printf("warning: can not create r1\n");
    }

    load_json(&cfg, JRICH, 0);
    /* a second load into the same config frees and replaces every array */
    buf = jq(JRICH);
    mcx_readconfig(buf, &cfg);
    free(buf);
    mcx_savejdata((char*)"rich.json", &cfg);
    EXPECT_TRUE(read_file("r1/rich.json", &buf) > 0 && has_key(buf, strlen(buf), "MieScatter") && has_key(buf, strlen(buf), "AngleInverseCDF")
                && has_key(buf, strlen(buf), "Frequency") && has_key(buf, strlen(buf), "IQUV") && has_key(buf, strlen(buf), "Pattern"),
                "--dumpjson writes every optional field, into RootPath");
    free(buf);
    init_cfg(&cfg2);
    mcx_readconfig((char*)"r1/rich.json", &cfg2);
    EXPECT_TRUE(cfg2.polmedianum == 1 && cfg2.extrasrclen == 1 && cfg2.srcpattern && fabs(cfg2.omega - cfg.omega) < 1e-3 * cfg.omega,
                "the dumped input loads back");
    clear_cfg(&cfg);
    clear_cfg(&cfg2);

    /* multiple patterns (srcnum 2) are dumped as a 3-D array */
    load_json(&cfg, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern','Pos':[1,1,0],'Dir':[0,0,1],"
              "'Param1':[2,0,0,2],'Param2':[0,2,0,2],'SrcNum':2,'Pattern':{'Nx':2,'Ny':2,'Data':[1,0,0,1,0,1,1,0]}}}}", 0);
    mcx_savejdata((char*)"pat2.json", &cfg);
    EXPECT_TRUE(read_file("pat2.json", &buf) > 0 && has_key(buf, strlen(buf), "_ArrayZipData_"), "--dumpjson of 2 patterns");
    free(buf);
    clear_cfg(&cfg);

    /* VoxelSize of length 1 or Dim, CacheBox with sradius 0, Dir/Param1/Param2 that set the source count */
#ifndef MCX_OPENCL
    load_json(&cfg, "{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VoxelSize':{'Dx':[1],'Dy':[1,1,1,1],'Dz':[2]},"
              "'CacheBoxP0':[1,1,1],'CacheBoxP1':[2,2,2]}," JBASE_SHAPES "," JBASE_OPTODE "}", 0);
    EXPECT_TRUE(cfg.steps.x == -1.f && cfg.steps.y == -2.f && cfg.steps.z == -1.f, "VoxelSize lengths of 1 and Dim");
    clear_cfg(&cfg);
#endif
    init_cfg(&cfg);
    cfg.sradius = 0.f;
    buf = jq("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'CacheBoxP0':[1,1,1]}," JBASE_SHAPES ","
             "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[[0,0,1,'_Inf_'],[0,0,1,'-_Inf_']]}}}");
    mcx_readconfig(buf, &cfg);
    free(buf);
    EXPECT_TRUE(cfg.crop0.x == 0 && cfg.extrasrclen == 1 && isinf(cfg.srcdir.w), "sradius 0 clears the cache box; Dir rows add sources");
    clear_cfg(&cfg);
    load_json(&cfg, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Param1':[[1,0,0,0],[2,0,0,0]]}}}", 0);
    EXPECT_EQ(cfg.extrasrclen, 1, "Param1 rows add sources");
    clear_cfg(&cfg);
    load_json(&cfg, "{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Param2':[[1,0,0,0],[2,0,0,0]]}}}", 0);
    EXPECT_EQ(cfg.extrasrclen, 1, "Param2 rows add sources");
    clear_cfg(&cfg);

    /* a volume file in RootPath with shapes overlaid, and a seed file given in the input */
    write_file("r1/rv.bin", "\x01\x01\x01\x01\x01\x01\x01\x01", 8);
    load_json(&cfg, "{'Session':{'Photons':10,'RootPath':'r1'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
              "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[2,2,2],'VolumeFile':'rv.bin'},"
              "'Shapes':[{'Box':{'Tag':0,'O':[0,0,0],'Size':[1,1,1]}}]," JBASE_OPTODE "}", 0);
    EXPECT_TRUE(cfg.vol && cfg.vol[7] == 1, "a volume file in RootPath with shapes overlaid");
    clear_cfg(&cfg);
}

/*-----------------------------------------------------------------------------
 *  more .inp variants and option forms
 *---------------------------------------------------------------------------*/

static void test_more_inputs(void) {
    Config cfg;
    char* buf;
    float pat3[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    /* polarized media: the sphere density gives the mus of the medium, or, with g < 1, its
       mus' = mus*(1-g), the medium then taking the anisotropy of the Mie phase function */
    {
        const float gs[2] = {1.f, 0.9f};

        for (int i = 0; i < 2; i++) {
            char json[1024];
            snprintf(json, sizeof(json), "{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,10,%g,1.37]],"
                     "'MieScatter':[[0.001,0.5,1.03,1.59,1.33]],'Dim':[4,4,4]}," JBASE_SHAPES ","
                     "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1],'WaveLength':650}}}", gs[i]);
            load_json(&cfg, json, 1);

            if (i == 0) {
                EXPECT_REL(cfg.prop[1].mus, 10.f, 1e-4, "a polarized medium with g=1 keeps its mus");
            } else {
                EXPECT_REL(cfg.prop[1].mus * (1.f - cfg.prop[1].g), 1.f, 1e-3, "a polarized medium with g<1 keeps its mus'");
            }

            clear_cfg(&cfg);
        }
    }

    write_file("inp_pat3.bin", pat3, sizeof(pat3));
    /* issrcfrom0 and the focal length given as trailing numbers, pattern3d, maxgate capping */
    const char* inp =
        "1000\n12345\n2 2 1  1\n0 0 1 20\n0 5e-9 1e-9\ninp.bin\n1 4 1 4\n1 4 1 4\n1 4 1 4\n2\n1 0.9 0.01 1.37\n2 0.8 0.02 1.4\n1 1.5\n2 2 1\n"
        "pattern3d\n2 2 2 0\n0 0 0 0\ninp_pat3.bin\n";
    write_file("ut3.inp", inp, strlen(inp));
    init_cfg(&cfg);
    cfg.seed = 0;
    cfg.maxgate = 100;
    cfg.sradius = 1.f;
    mcx_readconfig((char*)"ut3.inp", &cfg);
    EXPECT_TRUE(cfg.issrcfrom0 == 1 && cfg.srcdir.w == 20.f && cfg.maxgate == 5, ".inp trailing values and maxgate capping");
    EXPECT_TRUE(cfg.srctype == MCX_SRC_PATTERN3D && cfg.srcpattern && cfg.crop1.x == 3, ".inp pattern3d and the cache box from sradius");
    /* reading it again into the same config frees the previous pattern */
    mcx_readconfig((char*)"ut3.inp", &cfg);
    clear_cfg(&cfg);

    /* an .inp in RootPath, read with --root */
    if (system("mkdir -p r2 && cp ut.inp inp.bin inp_pat.bin r2/") == 0) {
        init_cfg(&cfg);
        cfg.seed = 0;
        strcpy(cfg.rootpath, "r2");
        mcx_readconfig((char*)"r2/ut.inp", &cfg);
        EXPECT_EQ(cfg.vol[0], 2, ".inp with its volume in RootPath");
        clear_cfg(&cfg);
    }

    /* option forms: repeated -P/--json, integer -D/-Z/-w, a -W list, -y without a language */
    parse_opts(&cfg, "-P {\"Shapes\":[]} -P {\"Shapes\":[{\"Box\":{\"Tag\":1,\"O\":[1,1,1],\"Size\":[2,2,2]}}]} "
               "--json {\"Session\":{\"Photons\":5}} --json {\"Session\":{\"Photons\":7}} -D 4 -Z 1 -w 5 -W 30,70");
    EXPECT_TRUE(cfg.debuglevel == 4 && cfg.zipid == 1 && cfg.savedetflag == 5 && cfg.nphoton == 7, "repeated and integer options");
    EXPECT_NEAR(cfg.workload[1], 70.f, 1e-6, "-W list");
    clear_cfg(&cfg);
    EXPECT_EXIT_OK(parse_opts(&cfg, "-y"), "-y without a language lists the languages and exits");
    parse_opts(&cfg, "-l 1 -s /proc/nosuch/log");
    clear_cfg(&cfg);

    /* -f - reads the JSON input from stdin */
    {
        int status = 0;
        pid_t pid = fork();

        if (pid == 0) {
            Config c;
            char* json = jq("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}");
            write_file("stdin.json", json, strlen(json));

            if (freopen("stdin.json", "r", stdin) == NULL || freopen("/dev/null", "w", stdout) == NULL) {
                exit(2);
            }

            char a0[] = "mcx", a1[] = "-f", a2[] = "-", *argv[] = {a0, a1, a2, NULL};
            init_cfg(&c);
            mcx_parsecmd(3, argv, &c);
            exit(c.nphoton == 10 ? 0 : 3);
        }

        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0, "-f - reads the input from stdin");
    }

    /* -i reads an .inp from stdin */
    {
        int status = 0;
        pid_t pid = fork();

        if (pid == 0) {
            if (freopen("ut.inp", "r", stdin) == NULL || freopen("/dev/null", "w", stdout) == NULL) {
                exit(2);
            }

            parse_raw("-i -E 12345");
            exit(0);
        }

        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0, "-i reads an .inp from stdin");
    }

    /* runcommand returns the output of a shell command */
    char* out = NULL;
    runcommand((char*)"echo", (char*)" mcx-runcommand", &out);
    EXPECT_TRUE(out && strstr(out, "mcx-runcommand"), "runcommand captures the command output");
    free(out);

    /* -E with an .mch file: mcx_prepdomain loads its seeds for the replay */
    {
        char* json = jq("{'Session':{'Photons':10},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
                        "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37],[0.02,1,0.9,1.37]],'Dim':[4,4,4]}," JBASE_SHAPES ","
                        "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},'Detector':[{'Pos':[2,3,0],'R':1},{'Pos':[3,2,0],'R':1}]}}");
        write_file("seedin.json", json, strlen(json));
        free(json);
        char a0[] = "mcx", a1[] = "-f", a2[] = "seedin.json", a3[] = "-E", a4[] = "ut.mch", a5[] = "-O", a6[] = "J";
        char* argv[] = {a0, a1, a2, a3, a4, a5, a6, NULL};
        init_cfg(&cfg);
        mcx_parsecmd(7, argv, &cfg);
        cfg.flog = devnull;
        EXPECT_TRUE(cfg.seed == SEED_FROM_FILE && cfg.replay.seed != NULL, "-E file.mch loads the seeds for the replay");
        clear_cfg(&cfg);
    }

    /* .jdb with seeds, a detector ID column, source IDs, RootPath and appending */
    setup_output_cfg(&cfg);
    float ppath[4 * 3];

    for (int i = 0; i < 4; i++) {
        ppath[i * 3] = (float)((1u + (i & 1)) | ((1u + (i & 1)) << 16));  /* the source ID is packed in the upper 16 bits */
        ppath[i * 3 + 1] = 1.f;
        ppath[i * 3 + 2] = 0.5f;
    }

    unsigned char seeds[4 * 16] = {0};
    cfg.medianum = 2;
    cfg.prop = (Medium*)calloc(2, sizeof(Medium));
    cfg.extrasrclen = 2;
    cfg.issaveseed = 1;
    cfg.savedetflag = 0x1 | 0x4;
    cfg.his.colcount = 3;
    cfg.his.maxmedia = 1;
    cfg.his.totalsource = 2;
    cfg.his.seedbyte = 16;
    cfg.his.detected = 4;
    cfg.his.savedphoton = 4;
    strcpy(cfg.rootpath, "r1");
    cfg.outputformat = ofBJNifti;
    mcx_savedetphoton(ppath, seeds, 4, 0, &cfg);
    mcx_savedetphoton(ppath, seeds, 4, 1, &cfg);
    long len = read_file("r1/ut_detp.jdb", &buf);
    EXPECT_TRUE(len > 0 && has_key(buf, len, "srcid") && has_key(buf, len, "seed"), ".jdb with source IDs and seeds, appended in RootPath");
    free(buf);
    cfg.outputformat = ofJNifti;
    mcx_savedetphoton(ppath, seeds, 4, 0, &cfg);
    EXPECT_TRUE(read_file("r1/ut_detp.jdt", &buf) > 0 && has_key(buf, strlen(buf), "srcid"), ".jdt with source IDs in RootPath");
    free(buf);
    clear_cfg(&cfg);

    /* mcx_preprocess: DoSaveRef 2 needs a large enough maxdetphoton, savedetflag defaults */
    load_json(&cfg, "{'Session':{'Photons':10,'DoSaveRef':2,'DoPartialPath':1,'MaxDetPhoton':10,'SaveDataMask':0},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
              JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},'Detector':[{'Pos':[2,3,0],'R':1}]}}", 1);
    EXPECT_TRUE(cfg.maxdetphoton >= 64 && cfg.savedetflag != 0, "DoSaveRef 2 raises maxdetphoton; a default SaveDataMask");
    clear_cfg(&cfg);
}


/*-----------------------------------------------------------------------------
 *  NeuroJSON.io downloads (-N, answered offline by a stand-in curl), inline JSON
 *  input, option parsing helpers and less common branches of the JSON writer
 *---------------------------------------------------------------------------*/

static void test_net_and_branches(void) {
    Config cfg;
    char opts[2048], cwd[4096], path[8192];
    char* json = jq("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}");

    /* -f with the JSON input given inline */
    write_file("utnetdoc.json", json, strlen(json));
    snprintf(opts, sizeof(opts), "-f %s", json);
    EXPECT_EXIT_OK(parse_raw(opts), "-f with an inline JSON input");
    free(json);

    /* -N with a stand-in curl on the PATH */
    {
        const char* curl = "#!/bin/sh\n"
                           "case \"$*\" in\n"
                           "  *nosuchdb*) echo '{\"error\":\"not_found\"}';;\n"
                           "  *_find*) echo '{\"docs\":[{\"_id\":\"_design\"},{\"_id\":\"utdoc1\"},{\"_id\":\"utdoc2\"}]}';;\n"
                           "  *_all_docs*) echo '{\"rows\":[{\"id\":\"utdoc1\"},{\"id\":\".hidden\"}]}';;\n"
                           "  *) cat utnetdoc.json;;\n"
                           "esac\n";
        const char* oldpath = getenv("PATH");

        if (mkdir("utbin", 0755) == 0 && getcwd(cwd, sizeof(cwd))) {
            char* savedpath = strdup(oldpath ? oldpath : "/usr/bin:/bin");

            write_file("utbin/curl", curl, strlen(curl));
            chmod("utbin/curl", 0755);
            snprintf(path, sizeof(path), "%s/utbin:%s", cwd, savedpath);
            setenv("PATH", path, 1);

            EXPECT_EXIT_OK(parse_raw("-N"), "-N lists the simulations of the database");
            EXPECT_EXIT_OK(parse_raw("-N {\"selector\":{}}"), "-N with a query selector");
            EXPECT_EXIT_OK(parse_raw("-N limit=2"), "-N with a query string");
            EXPECT_EXIT(parse_raw("-N nosuchdb=1"), "-N with a failed query");
            EXPECT_EXIT_OK(parse_raw("-N utdoc1"), "-N downloads a simulation by name");
            EXPECT_EXIT_OK(parse_raw("-N mcx/utdoc1"), "-N downloads a simulation by path");
            EXPECT_EXIT_OK(parse_raw("-N https://neurojson.io:7777/mcx/utdoc1"), "-N downloads a simulation by URL");

            setenv("PATH", savedpath, 1);
            free(savedpath);
        }
    }

    EXPECT_EXIT(parse_raw("--bench cube60 -xy"), "an unknown short option");
    parse_opts(&cfg, "--atomic 0");
    EXPECT_TRUE(1, "--atomic 0 is accepted with a warning");
    clear_cfg(&cfg);

    /* Mie scattering of an invalid sphere size */
    {
        double mu[NANGLES] = {0.}, qsca, g;
        float4* sm = (float4*)calloc(NANGLES, sizeof(float4));
        EXPECT_EXIT(Mie(0., 1.2, mu, sm, &qsca, &g), "Mie with a sphere size of 0");
        EXPECT_EXIT(Mie(30000., 1.2, mu, sm, &qsca, &g), "Mie with a sphere size beyond 20000");
        free(sm);
    }

    /* shape errors: coincident cylinder ends, slab bounds that are not pairs, and a JSON
       shape file without a background grid */
    {
        unsigned int* vol = NULL;
        GridDim dim = {0};
        Grid3D g = {&vol, &dim, {0.f, 0.f, 0.f}, 0};
        const char* grid = "{\"Shapes\":[{\"Grid\":{\"Tag\":1,\"Size\":[4,4,4]}},";
        char buf[512];

        snprintf(buf, sizeof(buf), "%s{\"Cylinder\":{\"C0\":[1,1,1],\"C1\":[1,1,1],\"R\":1,\"Tag\":2}}]}", grid);
        EXPECT_TRUE(mcx_parse_shapestring(&g, buf) != 0, "a cylinder with coincident end points");
        free(vol);
        vol = NULL;
        snprintf(buf, sizeof(buf), "%s{\"XSlabs\":{\"Tag\":2,\"Bound\":[[1,2,3]]}}]}", grid);
        EXPECT_TRUE(mcx_parse_shapestring(&g, buf) != 0, "slab bounds that are not number pairs");
        free(vol);
        write_file("utshape.json", grid, strlen(grid));
        EXPECT_TRUE(mcx_load_jsonshapes(NULL, (char*)"utshape.json") != 0, "a JSON shape file without a background grid");
    }

    /* values given in the command line before an .inp file: the voxel size, the length
       unit and a negative cache-box radius, which keeps the 1-based crop0/crop1 given */
    {
        /* the x/y/z lines of an .inp file give the voxel size, the dimension and the 1-based
           cache box range */
        char* inp;
        long len = read_file("ut.inp", &inp);

        for (char* p = inp; len > 0 && (p = strstr(p, "1 4 1 4")) != NULL; p++) {
            *p = '2';
        }

        write_file("utstep.inp", inp, strlen(inp));
        free(inp);
    }
    init_cfg(&cfg);
    cfg.sradius = -1.f;
    mcx_readconfig((char*)"utstep.inp", &cfg);
    EXPECT_NEAR(cfg.unitinmm, 2.f, 1e-6, ".inp: the voxel size sets the length unit");
    EXPECT_TRUE(cfg.crop0.x == 0 && cfg.crop1.x == 3, ".inp: a negative radius converts the cache box to 0-based");
    clear_cfg(&cfg);
#ifdef MCX_OPENCL
    init_cfg(&cfg);
    cfg.unitinmm = 0.5f;
    mcx_readconfig((char*)"ut.inp", &cfg);
    EXPECT_NEAR(cfg.steps.x, 0.5f, 1e-6, ".inp: the length unit sets the voxel size");
    clear_cfg(&cfg);
#endif

    /* mcx_validatecfg with a negative cache-box radius, and a reversed time window */
    json = jq("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}");
    init_cfg(&cfg);
    mcx_readconfig(json, &cfg);
    cfg.issrcfrom0 = 0;
    cfg.sradius = -1.f;
    cfg.crop0.x = cfg.crop0.y = cfg.crop0.z = 2;
    cfg.crop1.x = cfg.crop1.y = cfg.crop1.z = 3;
    mcx_validatecfg(&cfg, NULL, NULL, 0);
    EXPECT_TRUE(cfg.crop0.x == 1 && cfg.crop1.x == 2, "mcx_validatecfg converts a given cache box to 0-based");
    clear_cfg(&cfg);
    init_cfg(&cfg);
    mcx_readconfig(json, &cfg);
    cfg.tstart = 2e-9f;
    EXPECT_EXIT(mcx_validatecfg(&cfg, NULL, NULL, 0), "mcx_validatecfg with a reversed time window");
    clear_cfg(&cfg);
    free(json);

    /* a cache box in the JSON input, and a voxel size that sets the length unit */
    load_json(&cfg, "{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'Step':[2,2,2],"
              "'CacheBoxP0':[2,2,2],'CacheBoxP1':[3,3,3]}," JBASE_SHAPES "," JBASE_OPTODE "}", 1);
    EXPECT_NEAR(cfg.unitinmm, 2.f, 1e-6, "Domain.Step sets the length unit");
    clear_cfg(&cfg);

    /* a cache box in the JSON input with a negative radius given before it (-R -1) */
    json = jq("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],"
              "'CacheBoxP0':[2,2,2],'CacheBoxP1':[3,3,3]}," JBASE_SHAPES "," JBASE_OPTODE "}");
    init_cfg(&cfg);
    cfg.sradius = -1.f;
    mcx_readconfig(json, &cfg);
    EXPECT_TRUE(cfg.crop0.x == 1 && cfg.crop1.x == 2, "a JSON cache box is converted to 0-based");
    clear_cfg(&cfg);
    free(json);

    /* JData arrays decoded into an existing buffer: compressed, base64 and numeric data */
    {
        const char* arr[] = {
            "{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2,2],\"_ArrayZipType_\":\"zlib\",\"_ArrayZipSize_\":4,\"_ArrayZipData_\":\"eJxjZGBgBAAACQAD\"}",
            "{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[4],\"_ArrayData_\":\"AQIDBA==\"}",
            "{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[4],\"_ArrayData_\":[1,2,3,4]}",
        };

        for (int i = 0; i < 3; i++) {
            cJSON* obj = cJSON_Parse(arr[i]);
            void* vol = malloc(16);
            int ndim = 0;
            uint dims[3] = {1, 1, 1};
            char* type = NULL;

            init_cfg(&cfg);
            mcx_jdatadecode(&vol, &ndim, dims, 3, &type, obj, &cfg);
            EXPECT_TRUE(vol != NULL, "mcx_jdatadecode replaces an existing buffer");
            free(vol);
            cJSON_Delete(obj);
            clear_cfg(&cfg);
        }
    }

    /* an .inp file with a missing source pattern file */
    {
        char* inp;
        long len = read_file("ut.inp", &inp);
        char* pat = (len > 0) ? strstr(inp, "inp_pat.bin") : NULL;

        if (pat) {
            memcpy(pat, "inp_nop.bin", 11);
            write_file("utnopat.inp", inp, strlen(inp));
            init_cfg(&cfg);
            EXPECT_EXIT(mcx_readconfig((char*)"utnopat.inp", &cfg), "an .inp file with a missing pattern file");
            cfg.srcpattern = (float*)malloc(4);  /* an earlier pattern is replaced */
            mcx_readconfig((char*)"ut.inp", &cfg);
            EXPECT_TRUE(cfg.srcpattern && cfg.srcpattern[1] == 0.5f, "an .inp pattern replaces an earlier one");
            clear_cfg(&cfg);
        }

        free(inp);
    }

#ifdef MCX_OPENCL
    /* a device number that does not exist */
    {
        char a0[] = "mcxcl", a1[] = "--bench", a2[] = "cube60", a3[] = "-G", a4[] = "9", a5[] = "-n", a6[] = "100";
        char* argv[] = {a0, a1, a2, a3, a4, a5, a6, NULL};

        init_cfg(&cfg);
        mcx_parsecmd(7, argv, &cfg);
        cfg.flog = devnull;
        EXPECT_EXIT(mcx_run_simulation(&cfg, NULL, NULL), "a device number that does not exist");
        clear_cfg(&cfg);
    }

    /* a user kernel (-k) that fails to build, and a 2-D domain with a source direction along
       its singular dimension */
    {
        const char* badkernel = "__kernel void mcx_main_loop(__global float* a) { this does not compile; }\n";
        char a0[] = "mcxcl", a1[] = "--bench", a2[] = "cube60", a3[] = "-k", a4[] = "utbad.cl", a5[] = "-n", a6[] = "100";
        char* argv[] = {a0, a1, a2, a3, a4, a5, a6, NULL};

        write_file("utbad.cl", badkernel, strlen(badkernel));
        init_cfg(&cfg);
        mcx_parsecmd(7, argv, &cfg);
        cfg.flog = devnull;
        EXPECT_EXIT(mcx_run_simulation(&cfg, NULL, NULL), "a user kernel that fails to build");
        clear_cfg(&cfg);

        json = jq("{'Session':{'Photons':100,'ID':'utrun2d','DoSaveVolume':0},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
                  "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,1]},'Shapes':[{'Grid':{'Tag':1,'Size':[4,4,1]}}],"
                  "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]}}}");
        init_cfg(&cfg);
        mcx_readconfig(json, &cfg);
        cfg.flog = devnull;
        EXPECT_EXIT(mcx_run_simulation(&cfg, NULL, NULL), "a 2-D domain with a source direction along the singular dimension");
        clear_cfg(&cfg);
        free(json);
    }
#endif

    /* --dumpjson of an adjoint input leaves out the detectors appended as sources */
    load_json(&cfg, "{'Session':{'Photons':10,'ID':'utadj','OutputType':'a'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
              JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]},"
              "'Detector':[{'Pos':[1,2,0],'R':1,'Dir':[0,0,1]},{'Pos':[3,2,0],'R':1,'Dir':[0,0,1]}]}}", 1);
    mcx_savejdata((char*)"utadjdump.json", &cfg);
    {
        char* buf;
        long len = read_file("utadjdump.json", &buf);
        EXPECT_TRUE(len > 0 && has_key(buf, len, "\"Detector\""), "--dumpjson of an adjoint input");
        free(buf);
    }
    clear_cfg(&cfg);

    /* JData arrays of arrays, source rows of different counts, and an unknown .inp source type */
    {
        cJSON* obj = cJSON_Parse("{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[2],\"_ArrayData_\":[[1],[2]]}");
        void* vol = NULL;
        int ndim = 0;
        uint dims[3] = {1, 1, 1};
        char* type = NULL;

        init_cfg(&cfg);
        EXPECT_EXIT(mcx_jdatadecode(&vol, &ndim, dims, 3, &type, obj, &cfg), "a JData array of arrays");
        clear_cfg(&cfg);
        cJSON_Delete(obj);
    }
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[2,2,0,1],[3,3,0,1]],"
                      "'Dir':[[0,0,1],[0,0,1],[0,0,1]]}}}", "source Dir rows that do not match Pos");
    {
        char* inp;
        long len = read_file("ut.inp", &inp);
        char* hit = (len > 0) ? strstr(inp, "pattern   # source type") : NULL;

        if (hit) {
            memcpy(hit, "nosuchsrc", 9);
            write_file("utbadsrc.inp", inp, strlen(inp));
            init_cfg(&cfg);
            EXPECT_EXIT(mcx_readconfig((char*)"utbadsrc.inp", &cfg), "an .inp file with an unknown source type");
            clear_cfg(&cfg);
        }

        free(inp);
    }

    /* the GPU list is freed and cleared */
    {
        GPUInfo* info = (GPUInfo*)calloc(2, sizeof(GPUInfo));
        mcx_cleargpuinfo(&info);
        EXPECT_TRUE(info == NULL, "mcx_cleargpuinfo frees the GPU list");
    }

#ifdef MCX_OPENCL
    /* the built-in benchmarks by name */
    init_cfg(&cfg);
    mcx_loadbenchmark((char*)"cube60", &cfg);
    EXPECT_EQ(cfg.dim.x, 60, "mcx_loadbenchmark loads cube60");
    clear_cfg(&cfg);
    EXPECT_EXIT(mcx_loadbenchmark((char*)"nosuchbench", &cfg), "an unknown benchmark name");
#endif

    /* mcx_readarg: a flag without a value, and a list of unique byte-sized numbers */
    {
        char flag = 0, list[8] = {0};
        char a0[] = "mcx", a1[] = "-x", a2[] = "-G", a3[] = "1,2;1 3";
        char* argv1[] = {a0, a1, NULL};
        char* argv2[] = {a0, a2, a3, NULL};

        mcx_readarg(2, argv1, 1, &flag, "char");
        EXPECT_EQ(flag, 1, "mcx_readarg: a flag without a value is set to 1");
        mcx_readarg(3, argv2, 1, list, "bytenumlist");
        EXPECT_TRUE(list[0] == 1 && list[1] == 2 && list[2] == 3 && list[3] == 0, "mcx_readarg: bytenumlist drops repeated numbers");
    }

    /* the JSON writer: -b 2, a source weight, a 3D pattern and a packed 32-bit volume */
    json = jq("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'MediaFormat':'integer'},"
              JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern3d','Pos':[1,1,1],'Dir':[0,0,1],'Param1':[2,2,2,0],"
              "'Pattern':{'Nx':2,'Ny':2,'Nz':2,'Data':[1,0,1,0,1,0,1,0]}}}}");
    init_cfg(&cfg);
    mcx_readconfig(json, &cfg);
    mcx_preprocess(&cfg);
    cfg.isreflect = 2;
    cfg.srcpos.w = 0.5f;
    cfg.isdumpjson = 2;
    mcx_savejdata((char*)"utbr.json", &cfg);
    {
        char* buf;
        long len = read_file("utbr.json", &buf);
        EXPECT_TRUE(len > 0 && has_key(buf, len, "\"DoMismatch\":\t2") + has_key(buf, len, "\"DoMismatch\": 2") + has_key(buf, len, "\"DoMismatch\":2") > 0,
                    "--dumpjson saves -b 2 as a number");
        EXPECT_TRUE(len > 0 && has_key(buf, len, "\"Weight\""), "--dumpjson saves the source weight");
        free(buf);
    }
    clear_cfg(&cfg);
    free(json);

    /* mcx_validatecfg: several sources with 1-based positions, a time step of 0, too many
       gates, and no cache box */
    json = jq("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Pos':[[2,2,1,1],[3,3,1,1]],'Dir':[[0,0,1],[0,0,1]]}}}");
    init_cfg(&cfg);
    mcx_readconfig(json, &cfg);
    cfg.issrcfrom0 = 0;
    cfg.tstep = 0.f;
    cfg.maxgate = 10;
    cfg.sradius = 0.f;
    {
        float x0 = (cfg.extrasrclen > 0 && cfg.srcdata) ? cfg.srcdata[0].srcpos.x : 0.f;
        mcx_validatecfg(&cfg, NULL, NULL, 0);
        EXPECT_TRUE(cfg.extrasrclen == 0 || fabs(cfg.srcdata[0].srcpos.x - (x0 - 1.f)) < 1e-6, "mcx_validatecfg converts the 1-based extra source positions");
    }
    EXPECT_NEAR(cfg.tstep, cfg.tend, 1e-15, "mcx_validatecfg: a time step of 0 is one gate");
    EXPECT_EQ(cfg.maxgate, 1, "mcx_validatecfg caps maxgate by the gates");
    EXPECT_TRUE(cfg.crop0.x == 0 && cfg.crop1.x == 0, "mcx_validatecfg: no cache box without sradius");
    clear_cfg(&cfg);
    free(json);

    /* the volume file with a shape string, which -P rasterizes into it (-z 1: 0-based origin) */
    {
        unsigned char vol[64];
        memset(vol, 1, sizeof(vol));
        write_file("utvf.bin", vol, sizeof(vol));
        json = jq("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37],[0.02,1,0.9,1.37]],'Dim':[4,4,4],'VolumeFile':'utvf.bin'},"
                  JBASE_OPTODE "}");
        write_file("utvf.json", json, strlen(json));
        free(json);
        json = jq("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,4],'VolumeFile':'utvf.bin','MediaFormat':'muamus_float'},"
                  JBASE_OPTODE "}");
        write_file("utvff.json", json, strlen(json));
        free(json);

        EXPECT_EXIT_OK(parse_raw("-f utvf.json -z 1 -P {\"Shapes\":[{\"Sphere\":{\"O\":[2,2,2],\"R\":1,\"Tag\":2}}]}"), "-P rasterizes shapes into a volume file");
        EXPECT_EXIT(parse_raw("-f utvf.json -P {\"Shapes\":[{\"Sphere\":{\"O\":[2,2,2],\"Tag\":2}}]}"), "-P with an incomplete shape");
        EXPECT_EXIT(parse_raw("-f utvff.json -P {\"Shapes\":[{\"Sphere\":{\"O\":[2,2,2],\"R\":1,\"Tag\":2}}]}"), "-P with a non-label volume");
        EXPECT_EXIT_OK(parse_raw("-f utvf.json --dumpjson 2"), "--dumpjson 2 of a volume file input");
    }

    /* JSON input errors: a medium that is neither an object nor an array, a pattern that is
       not single precision, and an RF output without a modulation frequency */
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],1],'Dim':[4,4,4]}," JBASE_SHAPES "," JBASE_OPTODE "}", "a medium given as a number");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern','Pos':[1,1,0],'Dir':[0,0,1],"
                      "'Param1':[2,0,0,2],'Param2':[0,2,0,2],'Pattern':{'_ArrayType_':'uint8','_ArraySize_':[2,2],'_ArrayZipType_':'zlib',"
                      "'_ArrayZipSize_':4,'_ArrayZipData_':'eJxjZGBgBAAACQAD'}}}}", "a pattern that is not single precision");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'ID':'e','OutputType':'rf'},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}",
                      "an RF output without a modulation frequency");

    /* a number where a string is expected: its valuestring is NULL */
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'ID':5},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}", "a numeric Session.ID");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'ID':'e','RootPath':5},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}",
                      "a numeric Session.RootPath");
    EXPECT_JSON_ERROR("{'Session':{'Photons':10,'ID':'e','OutputFormat':5},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}",
                      "a numeric Session.OutputFormat");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1]],'Dim':[4,4,4],'MediaFormat':5}," JBASE_SHAPES "," JBASE_OPTODE "}", "a numeric Domain.MediaFormat");
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1]],'Dim':[4,4,4],'VolumeFile':-5}," JBASE_OPTODE "}", "a numeric Domain.VolumeFile");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':5,'Pos':[2,2,0],'Dir':[0,0,1]}}}", "a numeric Source.Type");

    /* a pattern smaller than Param1/Param2 give is copied to the device at the larger size */
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern','Pos':[1,1,0],'Dir':[0,0,1],"
                      "'Param1':[2,0,0,3],'Param2':[0,2,0,2],'Pattern':{'Nx':2,'Ny':2,'Data':[1,0,0,1]}}}}", "a pattern smaller than Param1/Param2");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern','Pos':[1,1,0],'Dir':[0,0,1],"
                      "'Param1':[2,0,0,2],'Param2':[0,2,0,2],'Pattern':{'_ArrayType_':'single','_ArraySize_':[1.5,2],'_ArrayData_':[1,0,0,1]}}}}",
                      "a non-integer _ArraySize_");
    EXPECT_JSON_ERROR("{" JBASE_SESSION "," JBASE_DOMAIN "," JBASE_SHAPES ",'Optode':{'Source':{'Type':'pattern','Pos':[1,1,0],'Dir':[0,0,1],"
                      "'Param1':[2,0,0,1],'Param2':[0,2,0,2],'Pattern':{'Nx':2,'Ny':2,'Data':[0,0,1,1]}}}}", "a pattern without a non-zero pixel in the sampled area");
    /* the lower and upper labels of a split voxel index the media */
    EXPECT_JSON_ERROR("{" JBASE_SESSION ",'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[2,1,1],'MediaFormat':'svmc'},"
                      "'Shapes':{'_ArrayType_':'uint64','_ArraySize_':[2,1,1],'_ArrayZipType_':'zlib','_ArrayZipSize_':[1,2],"
                      "'_ArrayZipData_':'eJxjZIAAVkYIDQAATwAI'}," JBASE_OPTODE "}", "an svmc label beyond the media");

#ifndef MCX_OPENCL

    /* the library entry point that runs a JSON input on the GPU; skipped on a
       machine with no NVIDIA or AMD (ROCm) GPU device, as the CI runners are */
    if (access("/dev/nvidia0", F_OK) == 0 || access("/dev/kfd", F_OK) == 0) {
        json = jq("{'Session':{'Photons':1000,'ID':'utrun','DoSaveVolume':0},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9}," JBASE_DOMAIN "," JBASE_SHAPES "," JBASE_OPTODE "}");
        EXPECT_EXIT_OK(mcx_run_from_json(json), "mcx_run_from_json runs a JSON input");
        free(json);
    } else {
        printf("skip mcx_run_from_json on a GPU: no GPU device found\n");
    }

    /* a 2-D domain can not have a source direction along its singular dimension */
    json = jq("{'Session':{'Photons':1000,'ID':'utrun2d','DoSaveVolume':0},'Forward':{'T0':0,'T1':1e-9,'Dt':1e-9},"
              "'Domain':{'Media':[[0,0,1,1],[0.01,1,0.9,1.37]],'Dim':[4,4,1]},'Shapes':[{'Grid':{'Tag':1,'Size':[4,4,1]}}],"
              "'Optode':{'Source':{'Pos':[2,2,0],'Dir':[0,0,1]}}}");
    EXPECT_EXIT(mcx_run_from_json(json), "a 2-D domain with a source direction along the singular dimension");
    free(json);
#endif
}

/*-----------------------------------------------------------------------------
 *  entry point, called by the main() of each repository's unit test binary
 *---------------------------------------------------------------------------*/

/** the source slots of the adjoint mode that cfg.sdmap needs, and the size of the output field */
static void test_srcslots(void) {
    Config cfg;
    unsigned int n, *slots;
    unsigned int sdmap[4] = {0, 1, 2, 3};  /* channels (src 1, det 2) and (src 3, det 4), 0-based */

    init_cfg(&cfg);
    cfg.outputtype = otAdjoint;
    cfg.srcid = -1;
    cfg.detnum = 4;
    cfg.extrasrclen = 2 + 4;  /* 3 sources and 4 detectors: 7 slots */
    cfg.medianum = 2;
    EXPECT_TRUE(mcx_getsrcslots(&cfg, &n) == NULL && n == 0, "without sdmap, every slot is simulated");
    EXPECT_EQ(mcx_fieldslots(&cfg), 7, "the field stores Ns+Nd solutions without sdmap");

    cfg.sdmap = sdmap;
    cfg.sdmaplen = 2;
    slots = mcx_getsrcslots(&cfg, &n);
    EXPECT_TRUE(slots && n == 4 && slots[0] == 1 && slots[1] == 3 && slots[2] == 5 && slots[3] == 7, "sdmap uses sources 1, 3 and detectors 2, 4 (slots 1, 3, 5, 7)");
    free(slots);
    EXPECT_EQ(mcx_fieldslots(&cfg), 4, "the field only stores the slots sdmap uses");

    cfg.srcid = 0;
    EXPECT_EQ(mcx_fieldslots(&cfg), 1, "merged sources (srcid 0) store one solution");
    cfg.srcid = -1;
    sdmap[0] = 1;
    sdmap[2] = 0;
    sdmap[3] = 2;  /* (src 2, det 2), (src 1, det 3): slots 2, 1, 5, 6 in increasing order */
    slots = mcx_getsrcslots(&cfg, &n);
    EXPECT_TRUE(slots && n == 4 && slots[0] == 1 && slots[1] == 2 && slots[2] == 5 && slots[3] == 6, "the slots are listed in increasing order");
    free(slots);
    cfg.sdmap = NULL;
    clear_cfg(&cfg);
}

void run_utils_tests(void) {
    char tmpdir[] = "/tmp/mcxunitXXXXXX", cwd[4096];

    devnull = fopen("/dev/null", "w");

    if (getcwd(cwd, sizeof(cwd)) == NULL || mkdtemp(tmpdir) == NULL || chdir(tmpdir)) {
        printf("FAIL can not create a temporary folder\n");
        nfail++;
        return;
    }

    test_helpers();
    test_half();
    test_mie();
    test_jdata();
    test_jdata_blocks();
    test_shapes();
    test_loadvolume();
    test_layout();
    test_savedata();
    test_detphoton();
    test_inp();
    test_loadjson();
    test_srcslots();
    test_parsecmd();
    test_devicename();
    test_validatecfg();
    test_maskdet();
    test_savejdata();
    test_json_errors();
    test_cmd_errors();
    test_io_errors();
    test_more_paths();
    test_rich_dump();
    test_clerrors();
    test_more_inputs();
    test_net_and_branches();

    if (chdir(cwd) == 0) {
        char cmd[4200];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", tmpdir);

        if (system(cmd)) {
            printf("warning: can not remove %s\n", tmpdir);
        }
    }

    fclose(devnull);
}
