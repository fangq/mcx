"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_qtest_subpixel.m

In this example, we test the sub-millimeter voxel feature using
the settings in examples/quicktest (i.e. comparing run_qtest.sh and
run_grid2x.sh)

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.0]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
# calculate the flux distribution with the given config
f1 = pmcx.mcxlab(cfg)

cfg["vol"] = np.ones([120, 120, 120], dtype="uint8")
cfg["srcpos"] = [60, 60, 1]
cfg["unitinmm"] = 0.5
cfg["nphoton"] = 8e7  # you need to simulate 8x photons to get the same noise

f2 = pmcx.mcxlab(cfg)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im = axs[0].imshow(np.log(f1["flux"][:, 29, :, 0]))
fig.colorbar(im, ax=axs[0])
axs[0].axis("off")
im2 = axs[1].imshow(np.log(f2["flux"][:, 59, :, 0]), clim=im.get_clim())
fig.colorbar(im2, ax=axs[1])
axs[1].axis("off")
plt.show()
