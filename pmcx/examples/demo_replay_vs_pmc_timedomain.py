"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_replay_vs_pmc_timedomain.m

In this example, we compare perturbation MC and replay in predicting
time-resolved measurement change with respect to mua change in a layer

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 2e8
vol = np.ones([60, 60, 30], dtype="uint8")
vol[:, :, 14:] = 2
cfg["vol"] = vol
cfg["srcpos"] = [30, 30, 0]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["issrcfrom0"] = 1
cfg["prop"] = np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.01, 1, 0, 1.37]])
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 2e-10
cfg["detpos"] = [[15, 30, 0, 2]]
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

## replay to get time-domain J_mua

newcfg = copy.deepcopy(cfg)
newcfg["seed"] = seeds
newcfg["outputtype"] = "jacobian"
newcfg["detphotons"] = detp["data"]
jac = pmcx.mcxlab(newcfg)["flux"]

## predict time-domain measurement change using pMC and partial-path

dmua = 0.0001
w1 = np.ravel(pmcx.detweight(detp, cfg["prop"]))
dprop = cfg["prop"].copy()
dprop[2, 0] = dprop[2, 0] + dmua
w2 = np.ravel(pmcx.detweight(detp, dprop))
dw = w2 - w1
tof = np.ravel(pmcx.dettime(detp, cfg["prop"]))
idx = np.digitize(tof, np.arange(0, cfg["tend"] + cfg["tstep"] / 2, cfg["tstep"]))  # 1-based time gate, as histc

## predict time-domain measurement change using TD Jacobian and compare

ngate = jac.shape[3]
dphi = np.zeros(ngate)  # change of measurements using replay Jacobian
dphi2 = np.zeros(ngate)  # change of measurements from pMC

for i in range(ngate):
    dmeas = jac[:, :, 14:, i] * dmua
    dphi[i] = -dmeas.sum()
    dphi2[i] = dw[idx == i + 1].sum() / w1.sum()

gates = np.arange(1, ngate + 1)
plt.plot(gates, dphi, "r-+", gates, dphi2, "g-+")
plt.legend(["use replay Jacobian", "use pMC"])
plt.xlabel("time gates")
plt.ylabel("measurement change")
plt.show()
