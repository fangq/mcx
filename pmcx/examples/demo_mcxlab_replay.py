"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_replay.m

In this example, we replay the detected photons of a baseline simulation to
compute the Jacobian of the detector reading, in CW and RF modes.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e8
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 0]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["issrcfrom0"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-10
# calculate the flux distribution with the given config
cfg["detpos"] = [[15, 30, 0, 2]]
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab
# cfg['savedetflag']='dsp'
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

newcfg = copy.deepcopy(cfg)
newcfg["seed"] = seeds
newcfg["outputtype"] = "jacobian"
newcfg["detphotons"] = detp["data"]
res2 = pmcx.mcxlab(copy.deepcopy(newcfg))
jac = res2["flux"].sum(axis=3)
plt.imshow(np.log10(np.abs(jac[:, 29, :])))

newcfg["outputtype"] = "rf"
newcfg["omega"] = 2 * np.pi * 100e6  # 100 MHz RF modulation
newcfg["detphotons"] = detp["data"]
newcfg["issavedet"] = 0  # mcxlab skips the detected photon output when only one output is requested
rfjac = pmcx.mcxlab(newcfg)
jac = rfjac["flux"].sum(axis=3)  # the last dimension stores the real and imaginary parts
plt.figure()
plt.imshow(np.log10(np.hypot(jac[:, 29, :, 0, 0], jac[:, 29, :, 0, 1])))
plt.show()
