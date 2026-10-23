"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_infinite_slab_cyclic_bc.m

In this example, we show how to simulate an infinite slab using cyclic
boundary condition

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e8
cfg["issrcfrom0"] = 1
cfg["vol"] = np.ones([60, 60, 20], dtype="uint8")
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 2, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999

# a uniform planar source outside the volume
cfg["srctype"] = "planar"
cfg["srcpos"] = [0, 0, 0]
cfg["srcparam1"] = [60, 0, 0, 0]
cfg["srcparam2"] = [0, 60, 0, 0]
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["bc"] = "ccrccr"
flux = pmcx.mcxlab(cfg)["flux"]

fcw = flux[:, :, :, 0] * cfg["tstep"]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im = axs[0].imshow(np.log10(np.abs(fcw[:, 29, :])))
axs[0].set_xlim([0, cfg["vol"].shape[2]])
fig.colorbar(im, ax=axs[0])
axs[0].set_title("a uniform planar source incident along an infinite slab")

axs[1].semilogy(fcw.mean(axis=(0, 1)))
axs[1].set_title("averaged light attenuation profile")
axs[1].set_ylabel("averaged CW fluence 1/mm^2")
axs[1].set_xlabel("depth (mm)")
plt.show()
