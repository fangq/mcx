"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_phasefun.m

In this example, we show how to customize phase function using cfg['invcdf']

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

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
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

# use built-in Henyey-Greenstein phase function
flux_hg = pmcx.mcxlab(cfg)["flux"]


# define Henyey-Greenstein phase function using cfg['invcdf']
def invhg(u, g):
    return (1 + g * g - ((1 - g * g) / (1 - g + 2 * g * u)) ** 2) / (2 * g)


u = np.arange(0.01, 1 - 0.01 + 1e-9, 0.01)
cfg["invcdf"] = invhg(u, 0.8)
flux_invhg = pmcx.mcxlab(cfg)["flux"]


# now use Rayleigh scattering phase function P(cos(theta))=P(u)=3/4*(1+u.^2)
def invrayleigh(u):
    a = 4 * u + ((4 * u - 2) ** 2 + 1) ** 0.5 - 2
    return np.cbrt(a) - 1 / np.cbrt(a)


cfg["invcdf"] = invrayleigh(u)

# calculate the flux distribution using Rayleigh scattering phase function
flux_rayleigh = pmcx.mcxlab(cfg)["flux"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, flux, title in zip(
    axs, (flux_hg, flux_invhg, flux_rayleigh), ("built-in Henyey-Greenstein", "invcdf Henyey-Greenstein", "invcdf Rayleigh")
):
    ax.imshow(np.log10(flux[:, 29, :, 0]).T)
    ax.set_title(title)
plt.show()
