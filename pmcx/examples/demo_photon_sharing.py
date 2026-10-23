"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_photon_sharing.m

Photon sharing: simulate 3 illumination patterns in a single run; each
detected/deposited photon carries its weight in all patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

mcximg = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

cfg = {}
cfg["nphoton"] = 1e7
vol = np.ones([60, 60, 60], dtype="uint8")
vol[:, :, 0] = 0
cfg["vol"] = vol
cfg["issaveref"] = 1
cfg["srctype"] = "pattern"
# split the 18x11 image into 3 patterns of 6x11, stacked along the 1st dimension: [3, 6, 11]
cfg["srcpattern"] = np.reshape(mcximg, [6, 3, mcximg.shape[1]], order="F").transpose(1, 0, 2)
cfg["srcnum"] = 3
cfg["srcpos"] = [0, 0, 0]
cfg["issrcfrom0"] = 1
cfg["srcdir"] = [0, 0, 1]
cfg["srcparam1"] = [60, 0, 0, cfg["srcpattern"].shape[1]]
cfg["srcparam2"] = [0, 60, 0, cfg["srcpattern"].shape[2]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["voidtime"] = 0
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["seed"] = 99999

# when photon sharing is used, the detpt['w0'] output of detected photon stores the
# 1-D index of the pixel within a 2D pattern (col-major, starting from 0)
# where the photon was initially launched. using detpt['w0'], one can look up
# the initial launch weight of each photon in all patterns

cfg["detpos"] = [[25, 30, 0, 2]]
cfg["savedetflag"] = "dpw"

res = pmcx.mcxlab(cfg)
detpt = res["detp"]

# the last dimension of flux and dref holds the solution of each pattern
fcw = np.squeeze(res["flux"]) * cfg["tstep"]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    im = axs[i].imshow(np.log10(np.abs(fcw[:, :, 1, i])))
    fig.colorbar(im, ax=axs[i])
    axs[i].set_title("pattern #%d" % (i + 1))

fcw = np.squeeze(res["dref"]) * cfg["tstep"]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    im = axs[i].imshow(np.abs(fcw[:, :, 0, i]))
    fig.colorbar(im, ax=axs[i])
    axs[i].set_title("pattern #%d" % (i + 1))

plt.figure()
plt.hist(np.ravel(detpt["w0"]))
plt.xlabel("1D pixel index of the launch position")
plt.show()
