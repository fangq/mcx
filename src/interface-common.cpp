#include <iostream>
#include <string>
#include <cstring>
#include "interface-common.h"
#include "mcx_shapes.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include <cmath>

void mcx_replay_prep(Config* cfg, float* detps, int dimdetps[2], int seedbyte,
                     const std::function<void(const char*)>& error_function) {
    int i, j, hasdetid = 0, offset;
    float plen;

    if (cfg->seed == SEED_FROM_FILE && detps == nullptr)
        error_function(
            "you give cfg.seed for replay, but did not specify cfg.detphotons.\nPlease define it as the detphoton output from the baseline simulation\n");

    if (detps == nullptr || cfg->seed != SEED_FROM_FILE) {
        return;
    }

    if (cfg->nphoton != dimdetps[1]) {
        error_function("the column numbers of detphotons and seed do not match\n");
    }

    if (seedbyte == 0) {
        error_function("the seed input is empty");
    }

    hasdetid = SAVE_DETID(cfg->savedetflag);
    offset = SAVE_NSCAT(cfg->savedetflag) * (cfg->medianum - 1);

    if (((!hasdetid) && cfg->detnum > 1) || !SAVE_PPATH(cfg->savedetflag))
        error_function(
            "please rerun the baseline simulation and save detector ID (D) and partial-path (P) using cfg.savedetflag='dp' ");

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
                plen = detps[i * dimdetps[0] + offset + j] * cfg->unitinmm;
                cfg->replay.weight[cfg->nphoton] *= expf(-cfg->prop[j - hasdetid + 1].mua * plen);
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

void validate_config(Config* cfg, float* detps, int dimdetps[2], int seedbyte,
                     const std::function<void(const char*)>& error_function) {
    int i, gates, idx1d, isbcdet = 0;
    const char boundarycond[] = {'_', 'r', 'a', 'm', 'c', '\0'};
    const char boundarydetflag[] = {'0', '1', '\0'};
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
        std::cerr << "incorrect time gate settings" << std::endl;
    }

    if (ABS(cfg->srcdir.x * cfg->srcdir.x + cfg->srcdir.y * cfg->srcdir.y + cfg->srcdir.z * cfg->srcdir.z - 1.f) > 1e-5) {
        std::cerr << "field 'srcdir' must be a unitary vector" << std::endl;
    }

    if (cfg->steps.x == 0.f || cfg->steps.y == 0.f || cfg->steps.z == 0.f) {
        std::cerr << "field 'steps' can not have zero elements" << std::endl;
    }

    if (cfg->tend <= cfg->tstart) {
        std::cerr << "field 'tend' must be greater than field 'tstart'" << std::endl;
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

    if ((cfg->outputtype == otJacobian || cfg->outputtype == otWP || cfg->outputtype == otDCS || cfg->outputtype == otRF)
            && cfg->seed != SEED_FROM_FILE) {
        std::cerr << "Jacobian output is only valid in the reply mode. Please define cfg.seed" << std::endl;
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
            std::cerr << "rasterization of shapes must be used with label-based mediatype" << std::endl;
        }

        Grid3D grid = {&(cfg->vol), &(cfg->dim), {1.f, 1.f, 1.f}, 0};

        if (cfg->issrcfrom0) {
            memset(&(grid.orig.x), 0, sizeof(float3));
        }

        int status = mcx_parse_shapestring(&grid, cfg->shapedata);

        if (status) {
            std::cerr << mcx_last_shapeerror() << std::endl;
        }
    }

    mcx_preprocess(cfg);

    cfg->his.maxmedia = cfg->medianum - 1; /*skip medium 0*/
    cfg->his.detnum = cfg->detnum;
    cfg->his.srcnum = cfg->srcnum;
    cfg->his.colcount = hostdetreclen; /*column count=maxmedia+2*/
    cfg->his.savedetflag = cfg->savedetflag;
    mcx_replay_prep(cfg, detps, dimdetps, seedbyte, error_function);
}

