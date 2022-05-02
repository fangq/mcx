#include <iostream>
#include <string>
#include <cstring>
#include "interface-common.h"
#include "mcx_shapes.h"
#include "mcx_core.h"
#include "mcx_const.h"
#include <cmath>

/**
 * @brief Pre-computing the detected photon weight and time-of-fly from partial path input for replay
 *
 * When detected photons are replayed, this function recalculates the detected photon
 * weight and their time-of-fly for the replay calculations.
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[in, out] detps
 * @param[in, out] dimdetps
 * @param[in, out] seedbyte
 * @param[in] error_function
 */
void mcx_replay_prep(Config *cfg, float *detps, int dimdetps[2], int seedbyte,
                     const std::function<void(const char *)> &error_function) {
  int i, j, hasdetid = 0, offset;
  float plen;
  if (cfg->seed == SEED_FROM_FILE && detps == nullptr)
    error_function(
        "you give cfg.seed for replay, but did not specify cfg.detphotons.\nPlease define it as the detphoton output from the baseline simulation\n");
  if (detps == nullptr || cfg->seed != SEED_FROM_FILE)
    return;
  if (cfg->nphoton != dimdetps[1])
    error_function("the column numbers of detphotons and seed do not match\n");
  if (seedbyte == 0)
    error_function("the seed input is empty");

  hasdetid = SAVE_DETID(cfg->savedetflag);
  offset = SAVE_NSCAT(cfg->savedetflag) * (cfg->medianum - 1);

  if (((!hasdetid) && cfg->detnum > 1) || !SAVE_PPATH(cfg->savedetflag))
    error_function(
        "please rerun the baseline simulation and save detector ID (D) and partial-path (P) using cfg.savedetflag='dp' ");

  cfg->replay.weight = (float *) malloc(cfg->nphoton * sizeof(float));
  cfg->replay.tof = (float *) calloc(cfg->nphoton, sizeof(float));
  cfg->replay.detid = (int *) calloc(cfg->nphoton, sizeof(int));

  cfg->nphoton = 0;
  for (i = 0; i < dimdetps[1]; i++) {
    if (cfg->replaydet <= 0 || cfg->replaydet == (int) (detps[i * dimdetps[0]])) {
      if (i != cfg->nphoton)
        memcpy((char *) (cfg->replay.seed) + cfg->nphoton * seedbyte,
               (char *) (cfg->replay.seed) + i * seedbyte,
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
          || cfg->replay.tof[cfg->nphoton] > cfg->tend) /*need to consider -g*/
        continue;
      cfg->nphoton++;
    }
  }
  cfg->replay.weight = (float *) realloc(cfg->replay.weight, cfg->nphoton * sizeof(float));
  cfg->replay.tof = (float *) realloc(cfg->replay.tof, cfg->nphoton * sizeof(float));
  cfg->replay.detid = (int *) realloc(cfg->replay.detid, cfg->nphoton * sizeof(int));
}
