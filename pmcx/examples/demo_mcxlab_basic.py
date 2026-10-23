"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_basic.m

In this example, we show the most basic usage of pmcx.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
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
res = pmcx.mcxlab(cfg)

## run multiple simulations, one config at a time (mcxlab also accepts a struct array)

cfgs = [copy.deepcopy(cfg), copy.deepcopy(cfg)]
cfgs[0]["isreflect"] = 0
cfgs[1]["isreflect"] = 1
cfgs[1]["detpos"] = [[30, 20, 1, 1], [30, 40, 1, 1], [20, 30, 1, 1], [40, 30, 1, 1]]

# calculate the flux and partial path lengths for the two configurations
results = [pmcx.mcxlab(c) for c in cfgs]

plt.imshow(np.log(results[0]["flux"][:, 29, :, 0]) - np.log(results[1]["flux"][:, 29, :, 0]))
plt.colorbar()
plt.show()
