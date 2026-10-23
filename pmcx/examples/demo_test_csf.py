"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_test_csf.m

Error evaluation when assuming musp(CSF) with 1/average(thickness)
   from Custo Applied Optics 2006

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## preparing the input data
cfg = {}
# set seed to make the simulation repeatible
cfg["seed"] = 0x623F9A9E

cfg["nphoton"] = 2e8

# define a 4 layer structure
vol = np.ones([100, 100, 50], dtype="uint8")
vol[:, :, 10:16] = 2
vol[:, :, 16:18] = 3
vol[:, :, 18:] = 4
cfg["vol"] = vol

# define the source position
cfg["srcpos"] = [51, 51, 1]
cfg["srcdir"] = [0, 0, 1]

# use the brain optical properties defined at
# http://mcx.sourceforge.net/cgi-bin/index.cgi?MMC/CollinsAtlasMesh
# format: [mua(1/mm) mus(1/mm) g n]

cfg["prop"] = np.array(
    [
        [0, 0, 1, 1],  # medium 0: the environment
        [0.019, 7.8, 0.89, 1.37],  # medium 1: scalp
        [0.019, 7.8, 0.89, 1.37],  # medium 2: skull
        [0.0004, 0.009, 0.89, 1.37],  # medium 3: CSF
        [0.02, 9.0, 0.89, 1.37],  # medium 4: gray matter
        [0.08, 40.9, 0.84, 1.37],  # medium 5: white matter (not used)
    ]
)

# time-domain simulation parameters
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

cfg["gscatter"] = 50  # make scattering faster, with some approximations

# GPU thread configuration
cfg["autopilot"] = 1
cfg["gpuid"] = 1

cfg["isreflect"] = 1  # enable reflection at exterior boundary

# use literature CSF optical properties
tic = time.time()
f1 = pmcx.mcxlab(cfg)["flux"][:, :, :, 0]
print("elapsed %.2f s" % (time.time() - tic))

# use approximated CSF mus', from Custo Applied Optics 2006
cfg["prop"][3, 1] = 1 / 2
cfg["prop"][3, 2] = 0
tic = time.time()
f2 = pmcx.mcxlab(cfg)["flux"][:, :, :, 0]
print("elapsed %.2f s" % (time.time() - tic))

dd = f1 / f2

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, title in [
    (axs[0], f1, "with csf"),
    (axs[1], f2, "csf mus'=1/thickness"),
    (axs[2], dd, "difference log10(with/without)"),
]:
    im = ax.imshow(np.log10(np.abs(data[:, 50, :])))
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    for z in (10, 16, 18):
        ax.plot([z - 1, z - 1], [0, 100], "--")
plt.show()
