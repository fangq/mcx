"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_replay_traj.m

In this example, we show how to handle photon trajectories and their
index mapping to the detected photon data in a replay simulation.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e6
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
cfg["issaveexit"] = 1
# calculate the flux distribution with the given config
cfg["detpos"] = [[15, 30, 0, 2]]
cfg["issaveseed"] = 1  # return the seeds of detected photons, as requesting the 4th output does in mcxlab

## base line simulation
res = pmcx.mcxlab(cfg)
detp, seeds = res["detp"], res["seeds"]

## setup replay simulation data

newcfg = copy.deepcopy(cfg)
newcfg["seed"] = seeds
newcfg["outputtype"] = "jacobian"
newcfg["detphotons"] = detp["data"]
newcfg["maxjumpdebug"] = 29371108
newcfg["debuglevel"] = "M"  # save photon trajectories, as requesting the 5th output does in mcxlab

## run replay
res2 = pmcx.mcxlab(newcfg)
detp2, traj2 = res2["detp"], res2["traj"]

jac = res2["flux"].sum(axis=3)

plt.figure()
plt.imshow(np.log10(np.abs(jac[:, :, 0].T)) / 10)

## plotphotons reorders trajectory positions in the sequential order
newtraj, _ = pmcx.plotphotons(traj2)
ax = plt.gca()
ax.plot([cfg["srcpos"][0]], [cfg["srcpos"][1]], [cfg["srcpos"][2]], "ro", markersize=10)
ax.plot([cfg["detpos"][0][0]], [cfg["detpos"][0][1]], [cfg["detpos"][0][2]], "yo", markersize=10)
ax.set_title("photon trajectories")

# idx is the index of the last trajectory position in the reordered newtraj
idx = np.nonzero(np.diff(newtraj["id"]))[0]

## visually plotting the detected photon exit position detp2['p'] with the last position stored in the trajectory
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
pos = newtraj["pos"][idx, :]
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "o")
ax.plot(detp2["p"][:, 0], detp2["p"][:, 1], detp2["p"][:, 2], "r.")
ax.legend(["using traj.pos", "using detp.p"])
ax.set_title("last position of detected photons")

## compare and map the detected photon data with trajectory data (MATLAB: ismember(...,'rows'))
lastpos = newtraj["pos"][np.append(idx, len(newtraj["id"]) - 1), :]
lookup = {tuple(row): k for k, row in enumerate(lastpos)}
loc = np.array([lookup.get(tuple(row), -1) for row in detp2["p"]])

# loc is the index in idx so that trajectory data can match detected photon
print(detp2["p"][:2, :])
print(lastpos[loc[:2], :])

ax = fig.add_subplot(1, 2, 2, projection="3d")
pos = newtraj["pos"][idx + 1, :]
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "o")
ax.set_title("launch position of each photon")
plt.show()
