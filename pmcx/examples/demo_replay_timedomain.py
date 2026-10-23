"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_replay_timedomain.m

In this example, we show how to use replay to obtain time-resolved
Jacobians - setting cfg['replaydet'] to -1 to replay all detectors

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e8
cfg["vol"] = np.ones([60, 60, 20], dtype="uint8")
cfg["srcpos"] = [30, 30, 0]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["issrcfrom0"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 2e-10
# calculate the flux distribution with the given config
cfg["detpos"] = [[30, 30, 20, 2], [30, 40, 20, 2], [30, 50, 20, 2]]
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

# cfg['replaydet']=0   # replay all det and sum all
# cfg['replaydet']=2   # replay only the 2nd detector
# cfg['replaydet']=3   # replay only the 3rd detector
cfg["replaydet"] = -1  # replay all det and save all

newcfg = copy.deepcopy(cfg)
newcfg["seed"] = seeds
newcfg["outputtype"] = "jacobian"
newcfg["detphotons"] = detp["data"]
res2 = pmcx.mcxlab(newcfg)
flux2 = res2["flux"]  # [Nx, Ny, Nz, time gates, detectors]

# time-resolved Jacobian of the first detector at a few time gates
gates = range(0, flux2.shape[3], 5)
fig, axs = plt.subplots(1, len(gates), figsize=(3 * len(gates), 3))
for ax, i in zip(axs, gates):
    with np.errstate(divide="ignore"):
        ax.imshow(np.log10(np.abs(flux2[29, :, :, i, 0])))
    ax.set_title("gate %d" % (i + 1))

jac = flux2.sum(axis=3)
fig, axs = plt.subplots(1, jac.shape[-1], figsize=(4 * jac.shape[-1], 4))
for i, ax in enumerate(np.atleast_1d(axs)):
    with np.errstate(divide="ignore"):
        ax.imshow(np.log10(np.abs(jac[29, :, :, i])))
    ax.set_title("detector %d" % (i + 1))
plt.show()
