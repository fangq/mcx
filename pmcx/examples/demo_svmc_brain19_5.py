"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_svmc_brain19_5.m

In this example, we compare SVMC algorithm against conventional VMC in a
full-head atlas template domain(USC 19.5 year group[Sanchez2012]).
This demo is similar to the MCX simulation used for Fig. 3(c) in [Yan2020]
and Fig. 9(a) in [TranYan2019].

[Sanchez2012] C.E.Sanchez J.E.Richards and C.R.Almli, "Age-Specific MRI Templates
for Pediatric Neuroimaging," Developmental Neuropsychology 37, 379-399 (2012).

[Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte
Carlo algorithm for accurate and efficient photon transport modeling in
complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)

[TranYan2019] Anh Phong Tran, Shijie Yan, Qianqian Fang, "Improving model-based
functional near-infrared spectroscopy analysis using mesh-based anatomical and
light-transport models," Neurophoton. 7(1) 015008 (22 February 2020)

The atlas, fullhead_atlas.mat, is loaded from the mcxlab/examples folder.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
import pmcx

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mcxlab", "examples")

## common MC setup
cfg = {}
cfg["nphoton"] = 1e8
cfg["seed"] = int(np.random.randint(1, 2**31 - 1))  # random seed

# inward-pointing pencil beam light source
cfg["srctype"] = "pencil"
cfg["srcpos"] = [133.5370, 90.1988, 200.0700]
cfg["srcdir"] = [-0.5086, -0.1822, -0.8415]
cfg["issrcfrom0"] = 1

# optical properties
cfg["prop"] = [
    [0.0, 0.0, 1.0, 1.0],  # background medium
    [0.019, 7.8, 0.89, 1.37],  # scalp
    [0.019, 7.8, 0.89, 1.37],  # skull
    [0.0004, 0.009, 0.89, 1.37],  # CSF
    [0.02, 9.0, 0.89, 1.37],  # gray matter
    [0.08, 40.9, 0.84, 1.37],  # white matter
    [0.0, 0.0, 1.0, 1.0],  # air cavity
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

## prepare vmc input volume
USC_atlas = loadmat(os.path.join(datadir, "fullhead_atlas.mat"))["USC_atlas"]
cfg_mcx = copy.deepcopy(cfg)
cfg_mcx["vol"] = USC_atlas

## prepare svmc input volume
tic = time.time()
svmcvol = pmcx.svmc(USC_atlas, smoothing=1)
print("SVMC preprocessing complete, %.2f s" % (time.time() - tic))

cfg_svmc = copy.deepcopy(cfg)
cfg_svmc["srcpos"] = list(np.array(cfg_svmc["srcpos"]) + 0.5)
cfg_svmc["vol"] = svmcvol

## run simulations
output_vmc = pmcx.mcxlab(cfg_mcx)  # conventional vmc
output_svmc = pmcx.mcxlab(cfg_svmc)  # svmc

## convert time-resolved fluence to CW fluence
phi_vmc = output_vmc["flux"].sum(axis=3)
phi_svmc = output_svmc["flux"].sum(axis=3)

## replace zero with nan
phi_vmc[phi_vmc == 0] = np.nan
phi_svmc[phi_svmc == 0] = np.nan

## compare CW fluence distributions using contour lines
y_plane = 90.5  # coronal plane selected for fluence plot
iy = int(np.ceil(y_plane)) - 1
xx, zz = np.meshgrid(np.arange(0.5, USC_atlas.shape[0]), np.arange(0.5, USC_atlas.shape[2]))

# interpolate SVMC results to slice y=91 (as marching cube mesh has an spatial offset of 0.5)
y = np.arange(0.5, USC_atlas.shape[1])
phi_svmc_interp = interp1d(y, phi_svmc, axis=1)(y_plane + 0.5)

# plot CW fluence distribution using contour lines
fig, ax = plt.subplots(figsize=(12, 6))
clines = np.arange(-20, 0.5, 0.5)
with np.errstate(divide="ignore", invalid="ignore"):
    cs = ax.contourf(xx - 0.5, zz - 0.5, np.log10(np.abs(phi_svmc_interp)).T, clines)
    ax.contour(xx, zz, np.log10(np.abs(phi_vmc[:, iy, :])).T, clines, colors="w", linestyles="--", linewidths=2)
fig.colorbar(cs, ax=ax)

# plot tissue boundaries
ax.contour(xx, zz, USC_atlas[:, iy, :].T.astype(float), colors=[(0.5, 0.5, 0.5)], linestyles="--", linewidths=1.5)

ax.set_aspect("equal")
ax.set_ylim([160, 220])
ax.set_xlim([45, 165])
ax.set_ylabel("z(mm)")
ax.set_xlabel("x(mm)")
cs.set_clim(-12, 0)
ax.set_title("filled: SVMC, dashed: VMC")
plt.show()
