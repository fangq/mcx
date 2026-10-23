"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_fullhead_atlas.m

In this example, we demonstrate light transport simulation in a full-head
atlas template(USC 19.5 year group[Sanchez2012]).
This demo is identical to the MCX simulation used for Fig.9(a) in
TranYan2020.

[Sanchez2012] C.E.Sanchez J.E.Richards and C.R.Almli, "Age-Specific MRI Templates
for Pediatric Neuroimaging," Developmental Neuropsychology 37, 379-399 (2012).

[TranYan2020] Tran AP+, Yan S+, Fang Q*, (2020) "Improving model-based fNIRS
analysis using mesh-based anatomical and light-transport models,"
Neurophotonics, 7(1), 015008

The atlas, fullhead_atlas.mat, is loaded from the mcxlab/examples folder.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pmcx

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mcxlab", "examples")

## prepare cfg for MCX simulation
cfg = {}
cfg["nphoton"] = 1e8
cfg["outputtype"] = "fluence"

# tissue labels:0-ambient air,1-scalp,2-skull,3-csf,4-gray matter,5-white matter,6-air cavities
cfg["vol"] = loadmat(os.path.join(datadir, "fullhead_atlas.mat"))["USC_atlas"]
cfg["prop"] = [
    [0, 0, 1, 1],
    [0.019, 7.8, 0.89, 1.37],
    [0.02, 9.0, 0.89, 1.37],
    [0.0004, 0.009, 0.89, 1.37],
    [0.019, 7.8, 0.89, 1.37],
    [0.08, 40.9, 0.84, 1.37],
    [0, 0, 1, 1],
]

# light source
cfg["srcnum"] = 1
cfg["srcpos"] = [133.5370, 90.1988, 200.0700]  # pencil beam source placed at EEG 10-5 landmark:"C4h"
cfg["srctype"] = "pencil"
cfg["srcdir"] = [-0.5086, -0.1822, -0.8415]  # inward-pointing source
cfg["issrcfrom0"] = 1

# time windows
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10

# other simulation parameters
cfg["isspecular"] = 0
cfg["isreflect"] = 1
cfg["autopilot"] = 1
cfg["gpuid"] = 1

## run MCX simulation
flux = pmcx.mcxlab(cfg)["flux"]

## post-simulation data processing and visualization
# convert time-resolved fluence to CW fluence
CWfluence = flux.sum(axis=3)

# coronal plane selected for fluence plot
y_plane = 90.5
iy = int(np.ceil(y_plane)) - 1
xx, zz = np.meshgrid(np.arange(1, cfg["vol"].shape[0] + 1), np.arange(1, cfg["vol"].shape[2] + 1))

# plot CW fluence distribution using contour lines
fig, ax = plt.subplots(figsize=(10, 6))
clines = np.arange(-20, 0.5, 0.5)
cs = ax.contourf(xx, zz, np.log10(np.abs(CWfluence[:, iy, :].T)), clines, linestyles="--")
ax.set_aspect("equal")
fig.colorbar(cs, ax=ax)

# plot tissue boundary contour, source, legend, etc.
ax.contour(xx, zz, cfg["vol"][:, iy, :].T, colors="k", linestyles="--", linewidths=1.25)
ax.plot(cfg["srcpos"][0], cfg["srcpos"][2], "o", color="r", markersize=10, label="source")
ax.legend(loc="upper right")
ax.set_ylim([160, 225])
ax.set_xlim([10, 165])
ax.set_ylabel("z [mm]")
ax.set_xlabel("x [mm]")
cs.set_clim(-12, 0)
plt.show()
