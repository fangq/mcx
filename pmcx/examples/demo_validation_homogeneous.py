"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validation_homogeneous.m

In this example, we validate pmcx with a homogeneous medium in a
cubic domain. This is exactly the example shown in Fig.5 of Fang2009.

You can also use the alternative optical properties that has a high g
value to observe the similarity between the two scattering/g configurations.

[Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation
of photon migration in 3D turbid media accelerated by graphics processing
units," Opt. Express 17, 20178-20190 (2009)

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx


def cwdiffusion(*args):
    phi = pmcx.cwdiffusion(*args)
    return np.ravel(phi[0] if isinstance(phi, tuple) else phi)


## preparing the input data
cfg = {}
# set seed to make the simulation repeatible
cfg["seed"] = 0x623F9A9E

cfg["nphoton"] = 3e7

# define a 6x6x6 cm box
dim = 60
cfg["vol"] = np.ones([dim, dim, dim], dtype="uint8")

# define the source position
cfg["srcpos"] = [dim / 2, dim / 2, 1]
cfg["srcdir"] = [0, 0, 1]

# define optical properties, format: [mua(1/mm) mus(1/mm) g n]
cfg["prop"] = np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.0]])  # medium 0: the environment, medium 1: the cube

# medium 1a: an alternative that has the same mus'=mus*(1-g) as medium 1.
# use medium 1a can significantly reduce racing
# cfg['prop'][1,:]=[0.005, 10.1, 0.9, 1.0]

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 1e-10

# GPU thread configuration
cfg["autopilot"] = 1
cfg["gpuid"] = 1

cfg["isreflect"] = 0  # disable reflection at exterior boundary

## running simulation without boundary reflection
print("running simulation ... this takes about 6 seconds on a GTX 470")
tic = time.time()
f1 = pmcx.mcxlab(cfg)["flux"]
print("elapsed %.2f s" % (time.time() - tic))

## running simulation with boundary reflection
cfg["prop"][1, 3] = 1.37
cfg["isreflect"] = 1  # enable reflection at exterior boundary

print("running simulation ... this takes about 11 seconds on a GTX 470")
tic = time.time()
f2 = pmcx.mcxlab(cfg)["flux"]
print("elapsed %.2f s" % (time.time() - tic))

## plotting the results
mua, musp = cfg["prop"][1, 0], cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2])
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

## plotting TPSF at voxel [30 14 9]
ax = axs[0, 0]
c0 = 299792458000
detpos = np.array([30, 14, 9])  # choose a point inside the domain

twin = np.arange(cfg["tstart"] + cfg["tstep"] / 2, cfg["tend"], cfg["tstep"])
d = detpos - 1  # 0-based voxel index
ax.semilogy(twin * 1e9, np.ravel(pmcx.tddiffusion(mua, musp, c0, 0, cfg["srcpos"], detpos + 1, twin)), "r")
ax.semilogy(twin * 1e9, f1[d[0], d[1], d[2], :], "o")
ax.semilogy(twin * 1e9, np.ravel(pmcx.tddiffusion(mua, musp, c0 / 1.37, 0.493, cfg["srcpos"], detpos, twin)), "r--")
ax.semilogy(twin * 1e9, f2[d[0], d[1], d[2], :], "+")
ax.set_ylim([1, 10**8.5])
ax.legend(["Diffusion (no reflection)", "MCX (no reflection)", "Diffusion (with reflection)", "MCX (with reflection)"], frameon=False)
ax.set_xlabel("time (ns)")
ax.set_ylabel("Fluence rate in 1/(mm^2 s)")

## plotting spatial decay profile along line y=30 && z=0
ax = axs[0, 1]
cwf1 = f1.sum(axis=3)
cwf2 = f2.sum(axis=3)

srcpos = [0, 0, 0]
detpos = np.column_stack([0.5 + np.arange(30), 0.5 * np.ones(30), 0.5 * np.ones(30)])

phicw = cwdiffusion(mua, musp, 0, srcpos, detpos)
phimcx = cwf1[29, :, 0]
phicwb = cwdiffusion(mua, musp, 0.493, srcpos, detpos)
phimcxb = cwf2[29, :, 0]

x = np.arange(1, 31)
ax.semilogy(x, phicw, "r-", x, phimcx[29:59] * cfg["tstep"], "o", x, phicwb, "r--", x, phimcxb[29:59] * cfg["tstep"], "b+")
ax.legend(["Diffusion (no reflection)", "MCX (no reflection)", "Diffusion (with reflection)", "MCX(with reflection)"], frameon=False)
ax.set_xlabel("x (mm)")
ax.set_ylabel("Fluence rate in 1/(mm^2 s)")

## plotting the cross-cut view along y=30
ax = axs[1, 0]
clines = np.arange(-4, -1.4, 0.5)
srcpos = np.array(cfg["srcpos"]) - 0.5
xi, yi = np.meshgrid(np.arange(0.5, 60), np.arange(0.5, 60))
detpos = np.column_stack([xi.ravel(), 30 * np.ones(xi.size), yi.ravel()])
phicw = cwdiffusion(mua, musp, 0, srcpos, detpos).reshape(xi.shape)

ax.contour(xi, yi, np.log10(np.maximum(phicw, 1e-8)), clines, colors="k", linestyles="-")
ax.contour(xi, yi, np.log10(np.maximum(cwf1[:, 29, :] * cfg["tstep"], 1e-8)).T, clines, colors="k", linestyles=":")
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("solid: diffusion, dotted: MCX")

## plotting the temporal evolution at the cross-cut along y=30
ax = axs[1, 1]
twin = np.arange(5e-11, 5e-9, 1e-10)
Reff = 0

srcpos = [29.5, 29.5, 0]
detpos = np.column_stack([xi.ravel(), 29.5 * np.ones(xi.size), yi.ravel()])
for i in range(0, 21, 5):
    phitd = np.ravel(pmcx.tddiffusion(mua, musp, c0, Reff, srcpos, detpos, twin[i])).reshape(xi.shape)
    halfmax = phitd.max() / 2
    ax.contour(xi, yi, np.maximum(phitd, 1e-12), [halfmax], colors="k", linestyles="-")
    halfmax = f1[:, 29, :, i].max() / 2
    ax.contour(xi, yi, np.maximum(f1[:, 29, :, i], 1e-12).T, [halfmax], colors="k", linestyles=":")
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("solid: diffusion, dotted: MCX")
plt.show()
