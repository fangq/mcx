"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_lambertian_exit_angle.m

Script to verify Lambertian photon exiting angular profile

Author: Qianqian Fang <q.fang at neu.edu>
Initial version: May 5, 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e8
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [20, 30, 0]
cfg["srcdir"] = [0, 0, 1]

cfg["maxdetphoton"] = 1500000
cfg["gpuid"] = 1
cfg["autopilot"] = 1

cfg["prop"] = np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37]])

cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

cfg["issrcfrom0"] = 1
cfg["isreflect"] = 1
cfg["issaveexit"] = 1

cfg["detpos"] = np.array([[25, 30, 0, 2]])

# cfg['prop']=np.array([[0, 0, 1, 1], [0.005, 0.01, 0, 1.37]])  # try this low-scattering case

# calculate the flux distribution with the given config
res = pmcx.mcxlab(cfg)
detp1 = res["detp"]

# it took me a while to figure out, but the key is to divide the area :)

el = np.arcsin(detp1["v"][:, 2])  # elevation angle of v, el=0 parallel to surface, el=pi/2 at normal dir

edges = np.linspace(el.min(), el.max(), 100)  # angle bins for hisotogram

bin = np.digitize(el, edges)  # angle bin of each photon, 1-based like histc

R = cfg["detpos"][0, 3]  # radius of the det
hedges = np.abs(R * np.sin(edges))  # height of each spherical segment for each bin
zonearea = 2 * np.pi * R * np.diff(hedges)  # area of each spherical segment

detweight = pmcx.detweight(detp1, cfg["prop"])  # get detected photon weight
angularweight = np.bincount(bin, weights=detweight, minlength=len(edges) + 1)[1:]  # sum total weight per zone

angularflux = angularweight[:-1] / zonearea  # calculate flux per angle

theta = (edges[:-1] + edges[1:]) * 0.5
ax = plt.figure().add_subplot(projection="polar")
ax.plot(theta, angularflux)  # plot flux vs angle
ax.plot(np.pi - theta, angularflux)  # mirror to form a circle
plt.show()
