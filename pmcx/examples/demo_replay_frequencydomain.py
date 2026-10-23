"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_replay_frequencydomain.m

In this example, we show how to compute the Jacobians for log-amplitude and
phase shift in the frequency domain (FD)/radio frequency (RF) mode.

Ref.: Hirvi et al. (2023). Effects of atlas-based anatomy on modelled
light transport in the neonatal head. Phys. Med. Biol.
https://doi.org/10.1088/1361-6560/acd48c

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

# FORM TARGET MODEL
# Here we consider a simple layered slab model with brain-like optical parameters.
layered_model = np.ones([90, 60, 60], dtype="uint8")
layered_model[:, :, 3] = 2
layered_model[:, :, 4:8] = 3
layered_model[:, :, 8:] = 4
opt_params = np.array(
    [[0, 0, 1, 1], [0.0150, 16, 0.9, 1.4], [0.004, 1.6, 0.9, 1.4], [0.048, 5, 0.9, 1.4], [0.037, 10, 0.9, 1.4]]
)
f_mod = 100e6  # 100 MHz RF modulation frequency

# MCX SETTINGS
cfg = {}
cfg["nphoton"] = 3e7
cfg["vol"] = layered_model
cfg["prop"] = opt_params
cfg["tstart"] = 0
cfg["tend"] = 1 / f_mod
cfg["tstep"] = cfg["tend"]
cfg["isnormalized"] = 0
cfg["unitinmm"] = 1.0
cfg["isspecular"] = 0  # photons have initial weight 1
cfg["issrcfrom0"] = 0  # first voxel is [1 1 1]
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["detpos"] = [[45, 30, 1, 2], [60, 30, 1, 2]]  # two detectors, one source
cfg["maxdetphoton"] = 5e7
cfg["gpuid"] = 1  # cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["savedetflag"] = "dp"
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab

# FORWARD SIMULATION
# Calculate the forward simulation distribution with the given config
# for one source and both detectors.
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

# RF REPLAY JACOBIANS
rfjac_lnA, rfjac_phase = pmcx.rfreplay(cfg, f_mod, detp, seeds, [1, 2])

# Visualize Jacobian slices for both data types and source-detector pairs.
fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
for ax, data, title in [
    (axs[0, 0], rfjac_lnA[:, 29, :20, 0], "d(lnA) det 1 [mm]"),
    (axs[0, 1], rfjac_lnA[:, 29, :20, 1], "d(lnA) det 2 [mm]"),
    (axs[1, 0], rfjac_phase[:, 29, :20, 0], "d(phase) det 1 [rad x mm]"),
    (axs[1, 1], rfjac_phase[:, 29, :20, 1], "d(phase) det 2 [rad x mm]"),
]:
    im = ax.imshow(data.T)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
plt.show()
