#ifndef MCX_INTERFACE_COMMON_H
#define MCX_INTERFACE_COMMON_H
#include "mcx_utils.h"
#include <functional>

/**
 * @brief contains functions used in both Matlab interface and Python interface of MCX.
 */
void mcx_replay_prep(Config *cfg, float *detps, int dimdetps[2], int seedbyte,
                     const std::function<void(const char *)> &error_function);

#endif
