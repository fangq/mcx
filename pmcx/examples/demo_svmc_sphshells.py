"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_svmc_sphshells.m

In this example, we compare SVMC algorithm against conventional VMC in a
heterogeneous domain. (see Fig. 3(a) of [Yan2020] and Fig. 1(e) of [Yan2019])

[Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte
Carlo algorithm for accurate and efficient photon transport modeling in
complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)

[Yan2019] Yan S, Tran AP, Fang Q. Dual-grid mesh-based Monte Carlo algorithm
for efficient photon transport simulations in complex three-dimensional
media. J Biomed Opt. 2019 Feb;24(2):1-4.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## common MC setup
cfg = {}
cfg["nphoton"] = 1e8
cfg["seed"] = int(np.random.randint(1, 2**31 - 1))  # random seed

# pencil beam light source
cfg["srcpos"] = [30.5, 30.5, 0]
cfg["srcdir"] = [0, 0, 1]
cfg["issrcfrom0"] = 1

# optical properties (tissue-like multi-layered media)
cfg["prop"] = [
    [0.0, 0.0, 1.0, 1.0],  # background (air, void)
    [0.02, 7.0, 0.89, 1.37],  # scalp/skull
    [0.004, 0.009, 0.89, 1.37],  # CSF
    [0.02, 9.0, 0.89, 1.37],  # gray matter
    [0.05, 0.0, 1.0, 1.37],  # non-scattering inclusion
]

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# enable boundary reflection/refraction
cfg["isreflect"] = 1

# spatial resolution
cfg["unitinmm"] = 1

# output fluence
cfg["outputtype"] = "fluence"

# GPU settings
cfg["gpuid"] = 1
cfg["autopilot"] = 1


def shells(r):
    xi, yi, zi = np.meshgrid(r, r, r, indexing="ij")
    dist = (xi - 30.5) ** 2 + (yi - 30.5) ** 2 + (zi - 30.5) ** 2
    vol = np.ones(xi.shape, dtype="uint8")
    vol[dist < 625] = 2
    vol[dist < 529] = 3
    vol[dist < 100] = 4
    return vol


## prepare mcx input volume
dim = 60
cfg_mcx = copy.deepcopy(cfg)
cfg_mcx["vol"] = shells(np.arange(0.5, dim))

## prepare svmc input volume
tic = time.time()
vol2 = shells(np.arange(1, dim + 1))
svmcvol = pmcx.svmc(vol2, smoothing=1)  # 1: enable gaussian smoothing 0: otherwise
print("SVMC preprocessing complete, %.2f s" % (time.time() - tic))

cfg_svmc = copy.deepcopy(cfg)
cfg_svmc["vol"] = svmcvol

## run simulations
output_vmc = pmcx.mcxlab(cfg_mcx)  # conventional vmc
output_svmc = pmcx.mcxlab(cfg_svmc)  # svmc

## convert time-resolved fluence to CW fluence
phi_vmc = output_vmc["flux"].sum(axis=3)
phi_svmc = output_svmc["flux"].sum(axis=3)

## compare CW fluence distributions using contour lines
fig, ax = plt.subplots(figsize=(8, 7))
clines = np.arange(-10, 10.5, 0.5)
grid = np.arange(1, dim + 1)
with np.errstate(divide="ignore"):
    cs = ax.contourf(grid, grid, np.log10(np.abs(phi_svmc[30, :, :])).T, clines)
    ax.contour(grid, grid, np.log10(np.abs(phi_vmc[30, :, :])).T, clines, colors="w", linestyles="--", linewidths=2)
fig.colorbar(cs, ax=ax)

# plot media boundaries
t = np.linspace(0, 2 * np.pi, 201)
for radius in (10, 23, 25):
    ax.plot(radius * np.cos(t) + 31, radius * np.sin(t) + 31, "--", linewidth=1.5, color=(0.5, 0.5, 0.5))

# other plot settings
ax.set_aspect("equal")
ax.set_ylabel("z(mm)")
ax.set_xlabel("y(mm)")
ax.set_title("filled: SVMC, dashed: VMC")
plt.show()
