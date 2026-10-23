"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_diffuse_reflectance_validation.m

In this example, we validate surface diffuse reflectance extracted from MC
detected photon profiles against analytical solutions computed from
diffusion equations (semi-infinite domain).

[Yao2018] Ruoyang Yao, Xavier Intes, and Qianqian Fang, "Direct approach
to compute Jacobians for diffuse optical tomography using perturbation Monte
Carlo-based photon 'replay'," Biomed. Opt. Express 9, 4588-4603 (2018)

[Kienle1997] Alwin Kienle and Michael S. Patterson, "Improved solutions
of the steady-state and the time-resolved diffusion equations for reflectance
from a semi-infinite turbid medium," J. Opt. Soc. Am. A 14, 246-254 (1997)

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

try:
    from redbirdpy.utility import getreff
except ImportError:  # effective reflection coefficient of the diffusion boundary condition

    def getreff(n_in, n_out=1.0):
        oc = np.arcsin(n_out / n_in)
        ostep = np.pi / 2000
        o = np.arange(0, oc, ostep)
        cosop = (n_in / n_out) * np.cos(o)
        cosop[cosop > 1] = 1
        sinop = np.sqrt(1 - cosop**2)
        cos_o, sin_o = np.cos(o), np.sin(o)
        r_fres = 0.5 * ((cos_o - (n_out / n_in) * cosop) / (cos_o + (n_out / n_in) * cosop)) ** 2 + 0.5 * (
            (cosop - (n_out / n_in) * cos_o) / (cosop + (n_out / n_in) * cos_o)
        ) ** 2
        r_phi = 2 * (np.sum(sin_o * cos_o * r_fres) * ostep + (1 - np.sin(oc) ** 2))
        r_j = 3 * (np.sum(sin_o * cos_o**2 * r_fres) * ostep + (2 / 3) * (1 - np.sin(oc) ** 3))
        return (r_phi + r_j) / (2 - r_phi + r_j)


## Monte Carlo
cfg = {}
cfg["nphoton"] = 1e8  # photon number
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")  # size in mm (length,width,height)

cfg["prop"] = np.array([[0.00, 0.0, 1.0, 1.0], [0.01, 10.0, 0.9, 1.4]])  # first row: ambient medium by default

cfg["srcpos"] = [30, 30, 0]  # (x,y,z) co-ordinates of light source
cfg["srcdir"] = [0, 0, 1]  # Direction - positive z-direction
cfg["issrcfrom0"] = 1

cfg["gpuid"] = 1
cfg["autopilot"] = 1

# time window: for time-solved results
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9

# save positions & directions of escaping photons
cfg["issaveexit"] = 1

# set up detectors: non-overlapping fiber detectors along x = 30
detpy = np.arange(31, 56)  # y changes from 31 to 55
r = 0.25  # radius 0.25mm
cfg["detpos"] = np.array([[30, y, 0, r] for y in detpy], dtype=float)

# visualize domain settings
pmcx.preview(cfg)

# launch simulations
res = pmcx.mcxlab(cfg)
detp = res["detp"]

# compute diffuse reflectance at detectors
drefmc = pmcx.cwdref(detp, cfg)

## Diffusion
mua, musp = cfg["prop"][1, 0], cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2])

# flux at detectors
fluxdiffusion = pmcx.cwfluxdiffusion(mua, musp, getreff(1.4, 1.0), cfg["srcpos"], cfg["detpos"][:, :3])

# fluence at detectors
fluencediffusion = pmcx.cwfluencediffusion(mua, musp, getreff(1.4, 1.0), cfg["srcpos"], cfg["detpos"][:, :3])
fluencediffusion = fluencediffusion[0] if isinstance(fluencediffusion, tuple) else fluencediffusion

# surface diffuse reflectance
drefdiffusion = 0.118 * np.ravel(fluencediffusion) + 0.306 * np.ravel(fluxdiffusion)  # Eq. 8 of Kienle1997

## compare diffuse reflectance
plt.figure()
idx = np.arange(1, len(detpy) + 1)
plt.semilogy(idx, drefdiffusion, "r--", idx, np.ravel(drefmc), "b+")
plt.legend(["Diffusion", "MC"])
plt.title("log10(Diffuse reflectance) [1/mm^2]")
plt.xlabel("Source-Detector seperation [mm]")
plt.show()
