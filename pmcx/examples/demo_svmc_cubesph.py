"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_svmc_cubesph.m

In this example, we compare SVMC algorithm against conventional VMC in a
heterogeneous domain. (see Fig. 3(b) of [Yan2020])

[Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte
Carlo algorithm for accurate and efficient photon transport modeling in
complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)

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

# optical properties
cfg["prop"] = [[0.0, 0.0, 1.0, 1.0], [0.005, 1.0, 0.89, 1.37]]  # background (air, void), spherical inclusion

# time-domain simulation parameters
cfg["srctype"] = "disk"
cfg["srcparam1"] = [3, 0, 0, 0]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# enable boundary reflection/refraction
cfg["isreflect"] = 1  # turn off boundary reflection to see matched results between VMC and SVMC

# spatial resolution
cfg["unitinmm"] = 1

# output fluence
cfg["outputtype"] = "fluence"

# GPU settings
cfg["gpuid"] = 1
cfg["autopilot"] = 1

## prepare mcx input volume
dim = 60
r = np.arange(0.25, dim - 0.25 + 1e-9, 0.5)
xi, yi, zi = np.meshgrid(r, r, r, indexing="ij")
dist = (xi - 30.5) ** 2 + (yi - 30.5) ** 2 + (zi - 30.5) ** 2
mcxvol = np.zeros(xi.shape, dtype="uint8")
mcxvol[dist < 625] = 1

cfg_mcx = copy.deepcopy(cfg)
cfg_mcx["vol"] = mcxvol

# finer spatial resolution
cfg_mcx["unitinmm"] = 0.5
cfg_mcx["srcpos"] = list(np.array(cfg_mcx["srcpos"]) / cfg_mcx["unitinmm"])

## prepare svmc input volume
tic = time.time()
r = np.arange(1, dim + 1)
xi, yi, zi = np.meshgrid(r, r, r, indexing="ij")
dist = (xi - 30.5) ** 2 + (yi - 30.5) ** 2 + (zi - 30.5) ** 2
vol2 = np.zeros(xi.shape, dtype="uint8")
vol2[dist < 625] = 1

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
yy_svmc = np.arange(0.5, 60, 1.0)
yy_vmc = np.arange(0.25, 60, 0.5)

fig, ax = plt.subplots(figsize=(8, 7))
clines = np.arange(-10, 10.5, 0.5)
with np.errstate(divide="ignore"):
    cs = ax.contourf(yy_svmc, yy_svmc, np.log10(np.abs(phi_svmc[30, :, :])).T, clines)
    ax.contour(yy_vmc, yy_vmc, np.log10(np.abs(phi_vmc[60, :, :])).T, clines, colors="w", linestyles="--", linewidths=2)
fig.colorbar(cs, ax=ax)

# plot media boundaries
t = np.linspace(0, 2 * np.pi, 201)
ax.plot(25 * np.cos(t) + 30.5, 25 * np.sin(t) + 30.5, "--", linewidth=1.5, color=(0.5, 0.5, 0.5))

# other plot settings
ax.set_aspect("equal")
ax.set_ylabel("z(mm)")
ax.set_xlabel("y(mm)")
ax.set_title("filled: SVMC, dashed: VMC")
plt.show()
