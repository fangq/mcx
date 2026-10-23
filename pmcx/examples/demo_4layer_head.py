"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_4layer_head.m

In this example, we simulate a 4-layer brain model using pmcx.
We will investigate the differences between the solutions with and
witout boundary reflections (both external and internal) and show
you how to display and analyze the resulting data.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## preparing the input data
cfg = {}
# set seed to make the simulation repeatible
cfg["seed"] = 0x623F9A9E

cfg["nphoton"] = 5e7

# define a 4 layer structure
vol = np.ones([100, 100, 50], dtype="uint8")
vol[:, :, 20:25] = 2
vol[:, :, 25:35] = 3
vol[:, :, 35:] = 4
cfg["vol"] = vol

# define the source position
cfg["srcpos"] = [51, 51, 1]
cfg["srcdir"] = [0, 0, 1]

# use the brain optical properties defined at
# http://mcx.sourceforge.net/cgi-bin/index.cgi?MMC/CollinsAtlasMesh
# format: [mua(1/mm) mus(1/mm) g n]

cfg["prop"] = [
    [0, 0, 1, 1],  # medium 0: the environment
    [0.019, 7.8, 0.89, 1.37],  # medium 1: skin & skull
    [0.0004, 0.009, 0.89, 1.37],  # medium 2: CSF
    [0.02, 9.0, 0.89, 1.37],  # medium 3: gray matter
    [0.08, 40.9, 0.84, 1.37],  # medium 4: white matter
]

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# GPU thread configuration
cfg["autopilot"] = 1
cfg["gpuid"] = 1

## running simulation without boundary reflection
print("running simulation ... this takes about 35 seconds on a GTX 470")
cfg["isreflect"] = 0  # disable reflection at exterior boundary
tic = time.time()
f1 = pmcx.mcxlab(cfg)
print("elapsed %.2f s" % (time.time() - tic))

## running simulation with boundary reflection enabled
print("running simulation ... this takes about 50 seconds on a GTX 470")
cfg["isreflect"] = 1  # enable reflection at exterior boundary
cfg["isrefint"] = 1  # enable reflection at interior boundary too
cfg["issavedet"] = 1  # enable recording partial pathlength of detected photons
cfg["detpos"] = [[31, 51, 1, 2]]
tic = time.time()
f2 = pmcx.mcxlab(cfg)
det2 = f2["detp"]
print("elapsed %.2f s" % (time.time() - tic))

## plot the results
fig = plt.figure(figsize=(12, 9))

layers = dict(linestyle="--")
for k, (res, label) in enumerate([(f1, "no reflection"), (f2, "reflection at boundaries")]):
    ax = fig.add_subplot(2, 2, k + 1)
    cs = ax.contourf(np.log10(np.sum(res["flux"][:, 50, :, :], axis=2).T), np.arange(1, 8.5, 0.5))
    for z in (21, 26, 36):
        ax.plot([0, 100], [z - 1, z - 1], **layers)
    ax.invert_yaxis()
    ax.set_title("Fluence rate with " + label)
    fig.colorbar(cs, ax=ax)

ax = fig.add_subplot(2, 2, 3)
xi = 1e9 * (np.arange(cfg["tstart"], cfg["tend"] - cfg["tstep"] / 2, cfg["tstep"]) + 0.5 * cfg["tstep"])
for i in range(31, 50):
    c = 1 - (i - 25) / 25
    ax.semilogy(xi, f2["flux"][50, i - 1, 0, :], color=(c, c, 1))
ax.set_xlabel("time (ns)")
ax.set_ylabel("Fluence rate (1/mm^2/s)")
ax.set_title("time point spread functions (TPSF)")

ax = fig.add_subplot(2, 2, 4)
ppath = det2["ppath"]
for k, color in enumerate("rbgk"):
    pp = ppath[ppath[:, k] > 0, k]
    hs, edges = np.histogram(pp, 200)
    ax.bar(0.5 * (edges[:-1] + edges[1:]), hs, width=edges[1] - edges[0], color=color, edgecolor="none")
ax.legend(["tissue 1", "tissue 2", "tissue 3", "tissue 4"])
ax.set_xlabel("partial pathlength (mm)")
ax.set_title("detected %d photons" % ppath.shape[0])

plt.show()
