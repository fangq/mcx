"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_continuous_mua_mus.m

In this example, we show how to define continuously varying media
(i.e. media optical properties vary from voxel to voxel)

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9


def show(vol, title):
    plt.figure()
    plt.imshow(vol[:, 29, :].T)
    plt.colorbar()
    plt.title(title)


## Approach #1: a 1 x Nx x Ny x Nz floating-point array defines mua(x,y,z)
# define continuously varying media - use a 4D array - 1st dimension defines
# the optical property components of each spatial position

mua1 = 0.003
mua2 = 0.1
mua = np.broadcast_to(np.linspace(mua1, mua2, 60, dtype=np.float32)[:, None, None], (60, 60, 60))
cfg["vol"] = mua[np.newaxis, ...].copy()  # make 1st dimension the property dimension

flux = pmcx.mcxlab(cfg)["flux"]

show(cfg["vol"][0], "continuously varying absorption coeff mu_a (1/mm)")
show(np.log10(flux[:, :, :, 0]), "fluence in continuously varying media (1/mm^2)")

## Approach #2: a 2 x Nx x Ny x Nz float32 array defines [mua,mus] per voxel

mus1 = 1.0
mus2 = 5.0
mus = np.broadcast_to(np.linspace(mus1, mus2, 60, dtype=np.float32)[:, None, None], (60, 60, 60))
cfg["vol"] = np.stack([mua, mus])

flux = pmcx.mcxlab(cfg)["flux"]

show(cfg["vol"][1], "continuously varying scattering coeff mu_s (1/mm)")
show(np.log10(flux[:, :, :, 0]), "fluence in continuously varying media (1/mm^2)")

## Approach #3: a 4 x Nx x Ny x Nz float32 array defines [mua,mus,g,n] per voxel

cfg["vol"] = np.stack([mua, mus, np.zeros_like(mua), np.full_like(mua, 1.37)])

flux = pmcx.mcxlab(cfg)["flux"]

show(cfg["vol"][1], "continuously varying scattering coeff mu_s (1/mm)")
show(np.log10(flux[:, :, :, 0]), "fluence in continuously varying media (1/mm^2)")

plt.show()
