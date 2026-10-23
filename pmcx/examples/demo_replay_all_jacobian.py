"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_replay_all_jacobian.m

In this example, we show how to compute any or all of the available
Jacobians for the continuous wave (CW), time (TD) and frequency domain (FD)
measurements with respect to voxel-wise absorption and scattering coefficients.

This file is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
Created by Pauliina Hirvi 4/2025
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## FORM TARGET MODEL

# Here we consider a simple layered slab model.
layered_model = np.ones([60, 60, 60], dtype="uint8")
layered_model[:, :, 3] = 2
layered_model[:, :, 4:8] = 3
layered_model[:, :, 8:] = 4

# Brain-like optical parameters.
# mua = absorption coeff, mus = scattering coeff, g = anisotropy coeff,
# n = index of refraction.
opt_params = np.array(
    [[0, 0, 1, 1], [0.0150, 14, 0.9, 1.4], [0.004, 1.6, 0.9, 1.4], [0.048, 5, 0.9, 1.4], [0.037, 10, 0.9, 1.4]]
)  # [mua,mus,g,n]
mus_matrix = np.where(layered_model > 0, opt_params[layered_model, 1], 1.0)  # mus value of each non-zero voxel
mus_matrix = mus_matrix + (mus_matrix == 0)  # avoid division with zero

# Select modulation frequency for FD.
f_mod = 100e6  # 100 MHz RF modulation frequency

## MCX FORWARD SIMULATION

# Settings for forward simulation.
cfg = {}
cfg["nphoton"] = 5e7  # large for scattering Jacobians
cfg["vol"] = layered_model
cfg["prop"] = opt_params
cfg["tstart"] = 0
cfg["tend"] = 1 / f_mod
cfg["tstep"] = cfg["tend"]  # cfg['tend'] for one time window
cfg["isnormalized"] = 0  # for Jacobians, avoid division with weight
cfg["unitinmm"] = 1.0
cfg["isspecular"] = 0  # photons have initial weight 1
cfg["issrcfrom0"] = 0  # first voxel is [1 1 1]
cfg["voidtime"] = 0  # only record travel time within medium
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["detpos"] = [[45, 30, 1, 2]]  # one detector
# cfg['maxdetphoton'] = cfg['nphoton']/10
cfg["gpuid"] = 1  # cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["issave2pt"] = 0  # faster forward simulation by not saving flux matrix
cfg["savedetflag"] = "dp"
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab

# Calculate the forward simulation distribution with the given config
# for one source and one detector.
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

# Compute total detected intensity as sum of detected weights.
detw = np.ravel(pmcx.detweight(detp, cfg["prop"], cfg["unitinmm"]))  # array with detected photon weights
I = detw.sum()

# Compute weighted mean time-of-flight (TOF).
dett = np.ravel(pmcx.dettime(detp, cfg["prop"], cfg["unitinmm"]))  # array with photon time-of-flights
meanTOF = np.sum(detw * dett) / I

## REPLAY MODE

# Common settings for replay mode.
cfg_jacobian = copy.deepcopy(cfg)
cfg_jacobian["seed"] = seeds
cfg_jacobian["detphotons"] = detp["data"]
cfg_jacobian["issave2pt"] = 1  # enable saving first output matrix from this on
cfg_jacobian["isnormalized"] = 0  # avoid double division with weight


def replay(outputtype):
    c = copy.deepcopy(cfg_jacobian)
    c["outputtype"] = outputtype
    c["issavedet"] = 0  # only the replayed volume is needed, as with a single output in mcxlab
    return pmcx.mcxlab(c)["flux"][:, :, :, 0]


## REPLAY JACOBIAN: CW/TD (LOG-)INTENSITY VERSUS ABSORPTION

# Absorption Jacobian for CW/TD (log-)intensity.
# See also: demo_mcxlab_replay.py
dIdmua = -cfg_jacobian["unitinmm"] * replay("jacobian")  # correct sign and units to [mm]
dlogIdmua = dIdmua / I  # convert to log-intensity Jacobian, i.e., weighted mean path for each voxel

## REPLAY JACOBIAN: CW/TD LOG-INTENSITY VERSUS SCATTERING

# Scattering Jacobian for CW/TD (log-)intensity.
meanP = replay("nscat") / I  # sum of weighted scattering counts in each voxel ('wp' or 'nscat')
dlogIdmus = meanP / mus_matrix + dlogIdmua

## REPLAY JACOBIAN: TD MEAN TIME-OF-FLIGHT VERSUS ABSORPTION

# Absorption Jacobian for TD weighted mean time-of-flight (TOF).
meanLTOF = cfg_jacobian["unitinmm"] * replay("wltof") / I  # correct units to [mm]
dTOFdmua = -meanLTOF - meanTOF * dlogIdmua

## REPLAY JACOBIAN: TD MEAN TIME-OF-FLIGHT VERSUS SCATTERING

# Scattering Jacobian for TD weighted mean time-of-flight (TOF).
meanPTOF = replay("wptof") / I
dTOFdmus = (meanPTOF - meanTOF * meanP) / mus_matrix + dTOFdmua

## REPLAY JACOBIAN: FD LOG-AMPLITUDE VERSUS ABSORPTION

# Absorption Jacobians for FD log-amplitude and phase shift in radians.
dlogAdmua, dphasedmua = pmcx.rfreplay(cfg, f_mod, detp, seeds, [1])

## JACOBIAN: FD LOG-AMPLITUDE VERSUS SCATTERING

# Scattering Jacobians for FD log-amplitude and phase shift in radians.
dlogAdmus, dphasedmus = pmcx.rfmusreplay(cfg, f_mod, detp, seeds, [1])

## VISUALIZE ALL JACOBIANS

# Visualize slices of each Jacobian type.
w = 2 * np.pi * f_mod
panels = [
    (dlogIdmua, "CW/TD d(lnI)/dmua [mm]"),
    (dlogAdmua[..., 0], "FD d(lnA)/dmua [mm]"),
    (dlogIdmus, "CW/TD d(lnI)/dmus [mm]"),
    (dlogAdmus[..., 0], "FD d(lnA)/dmus [mm]"),
    (w * dTOFdmua, "TD (2pi x f) x d(TOF)/dmua [mm]"),
    (dphasedmua[..., 0], "FD d(Phase)/dmua [mm]"),
    (w * dTOFdmus, "TD (2pi x f) x d(TOF)/dmus [mm]"),
    (dphasedmus[..., 0], "FD d(Phase)/dmus [mm]"),
]
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
for ax, (data, title) in zip(axs.flat, panels):
    im = ax.imshow(data[:, 30, :20].T)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.axis("off")
plt.show()
