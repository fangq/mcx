"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_polarized_photon.m

In this example, we model propagation of polarized light inside a
heterogeneous media.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

## simulation configurations
cfg = {}
# domain bounding box
dim = [100, 100, 12]
vol = np.ones(dim, dtype="uint8")

# add spherical inclusions
c = [60, 60, 6]  # center
r = 5  # radius
xi, yi, zi = np.meshgrid(*[np.arange(0.5, d) for d in dim], indexing="ij")
dist_sq = (xi - c[0]) ** 2 + (yi - c[1]) ** 2 + (zi - c[2]) ** 2
vol[dist_sq < r**2] = 2
cfg["vol"] = vol

# dx, dy, dz in mm
cfg["unitinmm"] = 0.1

# Mie scattering parameters of spherical particles:
# mua[1/mm], radius[micron], rho[1/micron^3], n_sph, n_med
cfg["polprop"] = [
    [0.005, 0.05, 19.1142, 1.59, 1.33],  # label 1
    [0.005, 1.0, 1.11078e-3, 1.59, 1.33],  # label 2
]

# optical properties
cfg["prop"] = [[0, 0, 1.0, 1.0], [0, 0, 1.0, 1.0], [0, 0, 1.0, 1.0]]

# planar wide-field source
cfg["srctype"] = "planar"
cfg["srcpos"] = [0, 0, 0]
cfg["issrcfrom0"] = 1
cfg["srcparam1"] = [dim[0], 0, 0, 0]
cfg["srcparam2"] = [0, dim[1], 0, 0]

# source direction
cfg["srcdir"] = [0, 0, 1]

# incident Stokes vector
cfg["srciquv"] = [1.0, 1.0, 0.0, 0.0]

# GPU settings
cfg["gpuid"] = 1
cfg["autopilot"] = 1

# time gate settings
cfg["tstart"] = 0
cfg["tend"] = 5e-8
cfg["tstep"] = 5e-8

# light wavelength[nm]
cfg["lambda"] = 632.8

# disable boundary reflection/refraction
cfg["isreflect"] = 0

# For each detected photon, save stokes parameter[i], exit position[x] and
# direction[v] and initial photon energy[w]
cfg["savedetflag"] = "ixvw"

# cyclic boundary condition to mimic horizontally infinite slabs
cfg["bc"] = "cc_cc_001000"

# photon number
cfg["nphoton"] = 1e7
cfg["maxdetphoton"] = 1e7  # max number of detected photons

## run simulation and plot results
res = pmcx.mcxlab(cfg)
detphoton = res["detp"]

# compute backscattered IQUV
phi = np.arctan2(detphoton["v"][:, 1], detphoton["v"][:, 0])
s = detphoton["s"]
dets = np.column_stack(
    [
        s[:, 0],
        s[:, 1] * np.cos(2 * phi) + s[:, 2] * np.sin(2 * phi),
        -s[:, 1] * np.sin(2 * phi) + s[:, 2] * np.cos(2 * phi),
        s[:, 3],
    ]
)

# weights of exit photons
w = pmcx.detweight(detphoton, detphoton["prop"], cfg["unitinmm"])

# compute total reflectanced IQUV
R = dets.T @ w / cfg["nphoton"]
print("total reflectance IQUV:", R)

# plot 2D(40x40) distributions for I Q U V
NN = 40
ix = np.digitize(detphoton["p"][:, 0], np.linspace(0, dim[0], NN + 1))
iy = np.digitize(detphoton["p"][:, 1], np.linspace(0, dim[1], NN + 1))
inside = (ix >= 1) & (ix <= NN) & (iy >= 1) & (iy <= NN)
HIQUV = np.zeros([4, NN, NN])

fig, axs = plt.subplots(2, 2, figsize=(10, 9))
for i, name in enumerate("IQUV"):
    np.add.at(HIQUV[i], (iy[inside] - 1, ix[inside] - 1), dets[inside, i])
    ax = axs.flat[i]
    im = ax.imshow(HIQUV[i])
    ax.set_title(name)
    fig.colorbar(im, ax=ax)
plt.show()
