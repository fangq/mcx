"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_speed_label_vs_continuous.m

In this example, we show why using label-based simulation is preferred
over per-voxel medium formats whenever possible.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import time
import numpy as np
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
vol = np.ones([60, 60, 60], dtype="uint8")
vol[:, :, 3:10] = 2
cfg["vol"] = vol
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["detpos"] = [[30, 33, 1, 2]]
cfg["issavedet"] = 0  # mcxlab skips saving detected photons when only the fluence output is requested
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["prop"] = np.array([[0, 0, 1, 1], [0.05, 1, 0, 1.37], [0.1, 1, 0, 1.37]])
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

# the optical properties of every voxel, looked up from its label
voxprop = cfg["prop"][vol].astype(np.float32)  # [60, 60, 60, 4]


def timed(cfg, label):
    tic = time.time()
    res = pmcx.mcxlab(cfg)
    print("%-28s %.2f s" % (label, time.time() - tic))
    return res


## simulate with lable-based medium
flux = timed(copy.deepcopy(cfg), "label-based medium")

## simulate with mua_float format
cfg2 = copy.deepcopy(cfg)
cfg2["vol"] = voxprop[np.newaxis, ..., 0]
flux2 = timed(cfg2, "mua_float format")

## simulate with muamus_float format
cfg3 = copy.deepcopy(cfg)
cfg3["vol"] = np.moveaxis(voxprop[..., 0:2], -1, 0)
flux3 = timed(cfg3, "muamus_float format")

## simulate with asgn_float format
cfg4 = copy.deepcopy(cfg)
cfg4["vol"] = np.moveaxis(voxprop, -1, 0)
flux4 = timed(cfg4, "asgn_float format")
