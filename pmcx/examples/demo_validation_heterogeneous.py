"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validation_heterogeneous.m

In this example, we validate the MCX solver with a heterogeneous
domain and the analytical solution of the diffusion model. We also
demonstrate how to use sub-pixel resolution to refine the representation
of heterogeneities. The domain is consisted of a 6x6x6 cm box with a
2cm diameter sphere embedded at the center.

This test is identical to that used for Fig. 3 in Fang2010.

[Fang2010] Fang Q, "Mesh-based Monte Carlo method using fast ray-tracing
   in Plucker coordinates," Biomed. Opt. Express 1, 165-175 (2010)

The analytical solution is computed by sphdiffusion.py, a port of the
sphere-diffusion toolbox in mmc/matlab, in this folder.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sphdiffusion import sphdiffusionslab

## preparing the input data
# structure for sphere-diffusion toolbox for the analytical solution
ana = {}
ana["v"] = 299792458000  # mm/s
ana["a"] = 10  # radius, mm
ana["omua"] = 0.002  # outside mua 1/mm
ana["omusp"] = 1.0  # outside mus' 1/mm
ana["imua"] = 0.05
ana["imusp"] = 0.5
ana["src"] = [30, np.pi, 0]
ana["maxl"] = 20
ana["omega"] = 0

clines = np.arange(-1, 8.25, 0.25)


def sphere_in_box(dim, center, radius2):
    xi, yi, zi = np.meshgrid(*([np.arange(1, dim + 1)] * 3))
    dist = (xi - center) ** 2 + (yi - center) ** 2 + (zi - center) ** 2
    vol = np.ones([dim, dim, dim], dtype="uint8")
    vol[dist < radius2] = 2
    return vol


cfg = {}
# set seed to make the simulation repeatible
cfg["seed"] = 0x623F9A9E

cfg["nphoton"] = 3e7

# define a 1cm radius sphere within a 6x6x6 cm box
dim = 60
cfg["vol"] = sphere_in_box(dim, 30.5, ana["a"] * ana["a"])

# define the source position
cfg["srcpos"] = [31, 31, 1]
cfg["srcdir"] = [0, 0, 1]

# format: [mua(1/mm) mus(1/mm) g n]
cfg["prop"] = [
    [0, 0, 1, 1],  # medium 0: the environment
    [ana["omua"], ana["omusp"] / (1 - 0.01), 0.01, 1.37],  # medium 1: cube
    [ana["imua"], ana["imusp"] / (1 - 0.01), 0.01, 1.37],  # medium 2: spherical inclusion
]

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# GPU thread configuration
cfg["autopilot"] = 1
cfg["gpuid"] = 1

cfg["isreflect"] = 0  # disable reflection at exterior boundaries

## running simulation with 1x1x1 mm input volume (60x60x60 grid)
print("running simulation ... this takes about 6 seconds on a GTX 470")
tic = time.time()
f1 = pmcx.mcxlab(copy.deepcopy(cfg))
print("elapsed %.2f s" % (time.time() - tic))

## set up simulation with 0.5x0.5x0.5 mm input volume (120x120x120 grid)
cfg2x = copy.deepcopy(cfg)
dim = 120
cfg2x["vol"] = sphere_in_box(dim, 60, 400)
cfg2x["srcpos"] = [61, 61, 1]

cfg2x["unitinmm"] = 0.5  # this statement specifies that each grid-edge is 0.5mm
cfg2x["nphoton"] = 8 * cfg["nphoton"]  # it needs 8x photons to get the same noise

print("running simulation ... this takes about 110 seconds on a GTX 470")
tic = time.time()
f2 = pmcx.mcxlab(cfg2x)
print("elapsed %.2f s" % (time.time() - tic))

## calculating the analytical solution from the sphere-diffusion toolbox
print("calculating the analytical solution. this may take a few minutes ...")
phi_ana, xa, ya, za = sphdiffusionslab(0, 0, dim, np.arange(-22, 22.01, 0.8), 0, np.arange(-30, 10.01, 0.8), ana)
phi_ana[((xa < -15) | (xa > 15)) & (za > -10)] = np.nan
phi_ana[((xa < -12) | (xa > 12)) & (za > 0)] = np.nan

## plotting the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

ax = axs[0]
cs = ax.contourf(np.arange(1, 61), np.arange(1, 61), np.log10(f1["flux"][:, 29, :, :].sum(axis=2).T), clines)
with np.errstate(invalid="ignore"):
    ax.contour(xa + 30.5, za + 31, np.log10(np.abs(phi_ana / cfg["tstep"])), clines, colors=[(0.7, 0.7, 0.7)], linewidths=2)
ax.contour(np.arange(1, 61), np.arange(1, 61), cfg["vol"][:, 29, :].T.astype(float), [1.5], colors="b", linestyles="--")
ax.set_aspect("equal")
ax.invert_yaxis()
fig.colorbar(cs, ax=ax, location="top")
ax.set_title("simulation with 1x1x1 mm resolution")

ax = axs[1]
xx = np.arange(0.5, dim / 2 + 0.01, 0.5)
cs = ax.contourf(xx, xx, np.log10(f2["flux"][:, 59, :, :].sum(axis=2).T), clines)
with np.errstate(invalid="ignore"):
    ax.contour(xa + 30, za + 30.5, np.log10(np.abs(phi_ana / cfg2x["tstep"])), clines, colors=[(0.7, 0.7, 0.7)], linewidths=2)
ax.contour(xx, xx, cfg2x["vol"][:, 59, :].T.astype(float), [1.5], colors="b", linestyles="--")
ax.set_aspect("equal")
ax.invert_yaxis()
fig.colorbar(cs, ax=ax, location="top")
ax.set_title("simulation with 0.5x0.5x0.5 mm resolution")
plt.show()
