"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_launchangle.m

In this example, we show how to customize photon launch angle distrbution
using cfg['angleinvcdf']

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 15]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
# cfg['gpuid']='11'  # use two GPUs together
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0, 0, 1, 1]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["isreflect"] = 0


def show(flux, title):
    plt.figure()
    plt.imshow(np.log10(np.abs(flux[:, 29, :, 0])).T)
    plt.colorbar()
    plt.title(title)


# define angleinvcdf in launch angle using cfg['angleinvcdf']
# launch angle is uniformly distributed between [pi/6 and pi/5] with interpolation (odd-number length)
cfg["angleinvcdf"] = np.linspace(1 / 6, 1 / 5, 5)
flux = pmcx.mcxlab(cfg)["flux"]
show(flux, "uniform launch angle between pi/6 and pi/5")

## test Lambertian/cosine distribution
# Lambertian distribution pdf: PDF(theta)=cos(theta)/pi, CDF=sin(theta)/pi, invcdf=asin(0:1)/pi,
# with interpolation (cfg['srcdir'][3] is not specified)
cfg["angleinvcdf"] = np.arcsin(np.arange(0, 1.0001, 0.01)) / np.pi
flux = pmcx.mcxlab(cfg)["flux"]
show(flux, "Lambertian launch angle")

## discrete angles with interpolation
cfg["angleinvcdf"] = [0, 0, 0, 0, 1 / 6, 1 / 6, 1 / 4]  # interpolatation is used between discrete angular values
flux = pmcx.mcxlab(cfg)["flux"]
show(flux, "discrete launch angles, interpolated")

## discrete angles without interpolation (by setting focal-length cfg['srcdir'][3] to 1)
cfg["angleinvcdf"] = [0, 0, 0, 0, 1 / 6, 1 / 6, 1 / 4]
cfg["srcdir"] = [0, 0, 1, 1]  # disable interpolation, use the discrete angles in angleinvcdf elements only
flux = pmcx.mcxlab(cfg)["flux"]
show(flux, "discrete launch angles, not interpolated")

## can be applied to area source - convolve with the area
cfg["srctype"] = "disk"
cfg["srcparam1"] = [10, 0, 0, 0]
cfg["srcdir"] = [0, 0, 1, 0]
flux = pmcx.mcxlab(cfg)["flux"]
show(flux, "disk source with discrete launch angles")
plt.show()
