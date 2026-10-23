"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_bc_det.m

In this example, we show how to set the 7-12 elements of cfg['bc'] to
indicate a planar detector along a boundary facet

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e5
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 1e-10

# calculate the flux distribution with the given config
cfg["bc"] = "______111110"  # capture photons existing from all faces except z=z_max
cfg["savedetflag"] = "dpx"

res = pmcx.mcxlab(cfg)
detpt = res["detp"]

ax = plt.figure().add_subplot(projection="3d")
ax.plot(detpt["p"][:, 0], detpt["p"][:, 1], detpt["p"][:, 2], "r.")
ax.view_init(elev=-90, azim=-90)
plt.show()
