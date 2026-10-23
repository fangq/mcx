"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validate_mcx_rf_redbird.m

Frequency-Domain comparison: Redbird (FEM diffusion) vs MCX (RF forward MC)

Redbird solves the frequency-domain diffusion equation:
  -div( D * grad(phi) ) + (mua + i*omega/(c/n)) * phi = q
MCX RF forward uses complex photon weights per Eq.9:
  w <- w * exp[ -(mua + i*omega*n/c0) * ds ]
Both produce complex-valued fluence. We compare amplitude and phase.

This demo requires redbirdpy (pip install redbirdpy) and iso2mesh.
"""

import copy
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import redbirdpy as rb
import pmcx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbfemhelper import slabmesh, cut2grid

## Parameters

freq = 200e6  # modulation frequency in Hz
omega = 2 * np.pi * freq  # angular frequency in rad/s

mua = 0.005  # absorption coeff [1/mm]
musp = 1.0  # reduced scattering coeff [1/mm]
g = 0  # anisotropy (isotropic for diffusion comparison)
nmed = 1.37  # refractive index
prop = [[0, 0, 1, 1], [mua, musp, g, nmed]]

## Redbird FEM setup (frequency domain)

cfg, _ = slabmesh([29.5, 29.5, 0], [40, 30, 0], prop, omega)

## Redbird: Build and solve FD system
print("Building Redbird FD system (f = %.0f MHz) ..." % (freq / 1e6))
tic = time.time()
detval_rb, phi_rb = rb.runforward(cfg)[:2]
print("elapsed %.2f s" % (time.time() - tic))
print("Redbird output is complex: %d" % np.iscomplexobj(phi_rb))

## Redbird CW for reference
cfg0, _ = slabmesh([29.5, 29.5, 0], [40, 30, 0], prop, 0)
phi_rb0 = np.real(rb.runforward(cfg0)[1])
phi_rb0[phi_rb0 < 0] = 0

## Extract Redbird detector readings
detval_rb = np.ravel(detval_rb)[0]
print("Redbird detector: amplitude = %.6e, phase = %.2f deg" % (abs(detval_rb), np.angle(detval_rb) * 180 / np.pi))

## MCX RF Forward setup

xcfg = {}
xcfg["nphoton"] = 1e8
xcfg["vol"] = np.ones([60, 60, 30], dtype="uint8")
xcfg["srcdir"] = [0, 0, 1, 0]
xcfg["autopilot"] = 1
xcfg["prop"] = prop
xcfg["tstart"] = 0
xcfg["tend"] = 5e-9
xcfg["tstep"] = 5e-9
xcfg["seed"] = 99999
xcfg["issrcfrom0"] = 1
xcfg["isreflect"] = 1
xcfg["srctype"] = "pencil"
xcfg["srcpos"] = [29.5, 29.5, 0]  # match Redbird source; issrcfrom0=1 so coords are physical mm

xcfg["omega"] = omega  # modulation angular frequency (omega>0 triggers RF forward)

print("Running MCX RF forward (f = %.0f MHz, %.0e photons) ..." % (freq / 1e6, xcfg["nphoton"]))
tic = time.time()
flux_rf = pmcx.mcxlab(copy.deepcopy(xcfg))
print("elapsed %.2f s" % (time.time() - tic))

# RF forward outputs complex flux rate (same unit as CW 'flux'); multiply by tstep for fluence
phi_mcx = np.squeeze(flux_rf["flux"]) * xcfg["tstep"]
print("MCX output is complex: %d, size: %s" % (np.iscomplexobj(phi_mcx), str(phi_mcx.shape)))

## MCX CW for reference
xcfg_cw = copy.deepcopy(xcfg)
xcfg_cw["outputtype"] = "flux"
del xcfg_cw["omega"]

print("Running MCX CW (%.0e photons) ..." % xcfg_cw["nphoton"])
flux_cw = pmcx.mcxlab(xcfg_cw)
phi_cw = np.squeeze(flux_cw["flux"]) * xcfg_cw["tstep"]

## Extract cross-sections at y=29.5 (voxel index 29, 0-based)
mcx_amp_yz = np.abs(phi_mcx[:, 29, :])  # [60 x 30]
mcx_phase_yz = np.angle(phi_mcx[:, 29, :]) * 180 / np.pi
cw_yz = np.real(phi_cw[:, 29, :])

x_mcx = np.arange(0.5, 60)
z_mcx = np.arange(0.5, 30)

# Redbird: cut at y=29.5 and grid onto the MCX voxel centers, [Nz x Nx]
rb_amp_grid = cut2grid(cfg, np.abs(phi_rb[:, 0]), 29.5, x_mcx, z_mcx)
rb_phase_grid = cut2grid(cfg, np.angle(phi_rb[:, 0]) * 180 / np.pi, 29.5, x_mcx, z_mcx)
rb_cw_grid = cut2grid(cfg0, phi_rb0[:, 0], 29.5, x_mcx, z_mcx)

extent = [x_mcx[0], x_mcx[-1], z_mcx[-1], z_mcx[0]]
eps = np.finfo(float).eps

## Figure 1: side-by-side 2D maps
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ax, img, clim, title in [
    (axs[0, 0], np.log10(cw_yz.T + eps), (-6, 0), "MCX CW log10(fluence)"),
    (axs[0, 1], np.log10(mcx_amp_yz.T + eps), (-6, 0), "MCX RF Amp log10 (f=%g MHz)" % (freq / 1e6)),
    (axs[0, 2], np.log10(rb_amp_grid + eps), (-6, 0), "Redbird FD Amp log10 (f=%g MHz)" % (freq / 1e6)),
    (axs[1, 0], np.zeros([len(z_mcx), len(x_mcx)]), (-60, 5), "MCX CW phase (0 deg)"),
    (axs[1, 1], mcx_phase_yz.T, (-60, 5), "MCX RF Phase (deg, f=%g MHz)" % (freq / 1e6)),
    (axs[1, 2], rb_phase_grid, (-60, 5), "Redbird FD Phase (deg, f=%g MHz)" % (freq / 1e6)),
]:
    im = ax.imshow(img, extent=extent, vmin=clim[0], vmax=clim[1])
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)
fig.suptitle("Frequency-Domain: Redbird (FEM) vs MCX RF Forward (MC)  |  f = %g MHz" % (freq / 1e6))

## Figure 2: contour overlay comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].contour(x_mcx, z_mcx, np.log10(rb_amp_grid + eps), np.arange(-7, 0.1, 0.5), colors="r", linewidths=2)
axs[0].contour(x_mcx, z_mcx, np.log10(mcx_amp_yz.T + eps), np.arange(-7, 0.1, 0.5), colors="b", linestyles="--", linewidths=2)
axs[0].set_title("log10(Amplitude), red: Redbird FD, blue: MCX RF")
axs[1].contour(x_mcx, z_mcx, rb_phase_grid, np.arange(-60, 0.1, 5), colors="r", linewidths=2)
axs[1].contour(x_mcx, z_mcx, mcx_phase_yz.T, np.arange(-60, 0.1, 5), colors="b", linestyles="--", linewidths=2)
axs[1].set_title("Phase (deg)")
for ax in axs:
    ax.invert_yaxis()
    ax.set_aspect("equal")

## Figure 3: depth profiles at (x=30, y=30)
z_line = np.arange(1, 31) - 0.5
mcx_amp_line = np.abs(phi_mcx[29, 29, :])
mcx_phase_line = np.angle(phi_mcx[29, 29, :]) * 180 / np.pi
cw_mcx_line = np.real(phi_cw[29, 29, :])
rb_amp_line, rb_phase_line, rb_cw_line = rb_amp_grid[:, 29], rb_phase_grid[:, 29], rb_cw_grid[:, 29]

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs[0, 0].semilogy(z_line, cw_mcx_line, "b-", z_mcx, rb_cw_line, "r--")
axs[0, 0].legend(["MCX CW", "Redbird CW"])
axs[0, 0].set_title("CW Fluence")
axs[0, 1].semilogy(z_line, mcx_amp_line, "b-", z_mcx, rb_amp_line, "r--")
axs[0, 1].legend(["MCX RF", "Redbird FD"])
axs[0, 1].set_title("FD Amplitude (f=%g MHz)" % (freq / 1e6))
axs[1, 0].plot(z_line, mcx_phase_line, "b-", z_mcx, rb_phase_line, "r--")
axs[1, 0].legend(["MCX RF", "Redbird FD"])
axs[1, 0].set_title("FD Phase (f=%g MHz)" % (freq / 1e6))
axs[1, 1].plot(z_line, mcx_amp_line / np.maximum(cw_mcx_line, eps), "b-", z_mcx, rb_amp_line / np.maximum(rb_cw_line, eps), "r--")
axs[1, 1].legend(["MCX", "Redbird"])
axs[1, 1].set_title("Amplitude demodulation ratio")
axs[1, 1].set_ylim([0, 1.1])

## Print summary
print("\n========== Summary ==========")
print("Frequency:     %.0f MHz" % (freq / 1e6))
print("mua = %.4f/mm, musp = %.1f/mm, n = %.2f" % (mua, musp, nmed))
print("\nAt detector (x=40, y=30, z=0):")
print("  Redbird: amp = %.4e, phase = %.2f deg" % (abs(detval_rb), np.angle(detval_rb) * 180 / np.pi))
mcx_det = phi_mcx[39, 29, 0]
print("  MCX:     amp = %.4e, phase = %.2f deg" % (abs(mcx_det), np.angle(mcx_det) * 180 / np.pi))
print("  Amp ratio (MCX/Redbird):   %.4f" % (abs(mcx_det) / abs(detval_rb)))
print("  Phase diff (MCX-Redbird):  %.2f deg" % ((np.angle(mcx_det) - np.angle(detval_rb)) * 180 / np.pi))
print("================================")
plt.show()
