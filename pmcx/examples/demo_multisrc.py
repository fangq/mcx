"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_multisrc.m

In this example, we show how to simultaneously simulate multiple sources
of the same type.

Specifically, we create a complex source made of an array of 9 planar
sources - where 8 of the sources form a circular array by rotating one
of the sources around the center of the bottom-face center; the 9th
source is manually placed at the center of the array.

One can individually control the settings for each of the sources,
including the position (srcpos), launch direction (srcdir), and
additional parameters (srcparam1/srcparam2).

By default, each source entry launches an equal fraction of photons,
determined by cfg['nphoton']/len(cfg['srcpos']). However, one can set the
4-th element of the srcpos for each source to create different weights
for each source. Here, we set the central source 4x of the intensity
compared to other circular-array sources (weight of 1). This allows mcx
to allocate 4/(8+4) = 1/3 of the total photons (cfg['nphoton']) to the central
source while each other source launches 1/12 of the total photon packets.

Worth to mention that we also set the 4th element of cfg['srcdir'], i.e. the
focal length, to create a convergent beam for each planar source.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 40], dtype="uint8")
cfg["srctype"] = "planar"

Rs = 20
delta = np.pi / 4

ang = np.arange(0, 2 * np.pi - delta + 1e-9, delta)
rotmat2d = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
offset2d = [30, 30]

# first create an array of planar sources by rotating around origin

# the 4th element of srcpos sets the initial weight
srcpos = np.tile([Rs, 0, 0, 1.0], (len(ang), 1))

# the 4th element of srcdir sets the focal length (positive: convergent)
srcdir = np.tile([-0.5, 0, np.sqrt(3) / 2, 20], (len(ang), 1))
srcparam1 = np.tile([-8, 3, 0, 0.0], (len(ang), 1))
srcparam2 = np.tile([3, 3, 0, 0.0], (len(ang), 1))

for i in range(1, len(ang)):
    for v in (srcpos, srcdir, srcparam1, srcparam2):
        v[i, :2] = rotmat2d @ v[i - 1, :2]

# translate the circular array to the desired offset
srcpos[:, 0] += offset2d[0]
srcpos[:, 1] += offset2d[1]

# manually add the 9th source to the center
cfg["srcpos"] = np.vstack([srcpos, [offset2d[0] - Rs / 4, offset2d[1] - Rs / 4, 0, 4]])  # initial weight is 4
cfg["srcdir"] = np.vstack([srcdir, [0, 0, 1, 0]])
cfg["srcparam1"] = np.vstack([srcparam1, [Rs / 2, 0, 0, 0]])
cfg["srcparam2"] = np.vstack([srcparam2, [0, Rs / 2, 0, 0]])

cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 0.2, 0, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

flux = pmcx.mcxlab(cfg)["flux"]
pmcx.plotvol(np.log10(flux[:, :, :, 0]))

## setting cfg['srcid'] to a positive number 1,2,3,.. specifies which src to simulate
cfg["srcid"] = 8

flux = pmcx.mcxlab(cfg)["flux"]
pmcx.plotvol(np.log10(flux[:, :, :, 0]))

## setting cfg['srcid'] to -1, solution of each src is stored separately

cfg["srcid"] = -1

flux = pmcx.mcxlab(cfg)["flux"]

# the last dimension holds the solution of each of the 9 sources
pmcx.plotvol(np.log10(np.squeeze(flux)))
plt.show()
