"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_sphere_cube_subpixel.m

In this example, we demonstrate how to use sub-pixel resolution
to represent the problem domain. The domain is consisted of a
6x6x6 cm box with a 2cm diameter sphere embedded at the center.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx


def sphere_in_box(dim, center, radius2):
    xi, yi, zi = np.meshgrid(*([np.arange(1, dim + 1)] * 3))
    dist = (xi - center) ** 2 + (yi - center) ** 2 + (zi - center) ** 2
    vol = np.ones([dim, dim, dim], dtype="uint8")
    vol[dist < radius2] = 2
    return vol


cfg = {}
# set seed to make the simulation repeatible
cfg["seed"] = 0x623F9A9E

cfg["nphoton"] = 1e7

# define a 1cm radius sphere within a 6x6x6 cm box
cfg["vol"] = sphere_in_box(60, 30, 100)

# define the source position
cfg["srcpos"] = [31, 31, 1]
cfg["srcdir"] = [0, 0, 1]

# format: [mua(1/mm) mus(1/mm) g n]
cfg["prop"] = [
    [0, 0, 1, 1],  # medium 0: the environment
    [0.002, 1.0, 0.01, 1.37],  # medium 1: cube
    [0.050, 0.5, 0.01, 1.37],  # medium 2: spherical inclusion
]

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# GPU thread configuration
cfg["autopilot"] = 1
cfg["gpuid"] = 1

cfg["isreflect"] = 1  # enable reflection at exterior boundary
cfg["isrefint"] = 1  # enable reflection at interior boundary too

print("running simulation ... this takes about 3 seconds on a GTX 470")
tic = time.time()
f1 = pmcx.mcxlab(cfg)
print("elapsed %.2f s" % (time.time() - tic))

# define a 1cm radius sphere within a 6x6x6 cm box with a 0.5mm resolution
cfg["vol"] = sphere_in_box(120, 60, 400)
cfg["srcpos"] = [61, 61, 1]

cfg["unitinmm"] = 0.5  # define the pixel size in terms of mm
cfg["nphoton"] = 8e7  # you need to simulate 8x photons to get the same noise

print("running simulation ... this takes about 40 seconds on a GTX 470")
tic = time.time()
f2 = pmcx.mcxlab(cfg)
print("elapsed %.2f s" % (time.time() - tic))

# plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, res, iy, title in [
    (axs[0], f1, 29, "simulation with 1x1x1 mm resolution"),
    (axs[1], f2, 59, "simulation with 0.5x0.5x0.5 mm resolution"),
]:
    cs = ax.contourf(np.log10(res["flux"][:, iy, :, :].sum(axis=2).T), np.arange(1, 8.5, 0.5))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.colorbar(cs, ax=ax, orientation="horizontal")
    ax.set_title(title)
plt.show()
