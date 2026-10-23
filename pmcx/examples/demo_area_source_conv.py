"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_area_source_conv.m

Comparing directly simulated widefield sources vs aperature-convolved
pencil beam solutions

Author: Qianqian Fang <q.fang at neu.edu>
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pmcx

## first, simulate a pencil beam solution

cfg = {}
cfg["nphoton"] = 1e8
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [29.5, 29.5, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = cfg["tend"]
# cfg['gpuid']=1
# cfg['autopilot']=1

flux = pmcx.mcxlab(cfg)["flux"][:, :, :, 0]

## next, perform a 2D convolution with a 3x7 kernel in the x/y plane

src = np.ones([3, 7])
fluxconv = np.zeros(flux.shape)

for i in range(flux.shape[2]):
    fluxconv[:, :, i] = convolve2d(flux[:, :, i], src, mode="same")

# normalize the convolved solution using total area
fluxconv = fluxconv / src.sum()

## then directly simulate such an area source using `planar` source type

cfg2 = copy.deepcopy(cfg)
cfg2["srctype"] = "planar"
cfg2["srcpos"] = [28, 26, 0]  # notice this is shifted by half grid in x/y to align the two
cfg2["srcparam1"] = [3, 0, 0, 0]
cfg2["srcparam2"] = [0, 7, 0, 0]
fluxplanar = pmcx.mcxlab(cfg2)["flux"][:, :, :, 0]

## comparing the two solutions

contournum = 30

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, a, b, title in [
    (axs[0], fluxplanar[:, :, 4], fluxconv[:, :, 4], "x-y plane (z=5)"),
    (axs[1], fluxplanar[:, 29, :], fluxconv[:, 29, :], "x-z plane (y=30)"),
    (axs[2], fluxplanar[29, :, :], fluxconv[29, :, :], "y-z plane (x=30)"),
]:
    ax.contourf(np.log10(a), contournum)
    ax.contour(np.log10(b), contournum, colors="r", linestyles="--")
    ax.set_title(title)
    ax.set_aspect("equal")

plt.show()
