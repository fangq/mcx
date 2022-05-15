#ifndef MCX_INTERFACE_COMMON_H
#define MCX_INTERFACE_COMMON_H
#include "mcx_utils.h"
#include <functional>

/**
 * @brief contains functions used in both Matlab interface and Python interface of MCX.
 */


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
                     const std::function<void(const char *)> &error_function);

/**
 * @brief Validate all input fields, and warn incompatible inputs
 *
 * Perform self-checking and raise exceptions or warnings when input error is detected
 *
 * @param[in,out] cfg: the simulation configuration structure
 */
void validate_config(Config *cfg, float *detps, int dimdetps[2], int seedbyte,
                     const std::function<void(const char *)> &error_function);

#endif
