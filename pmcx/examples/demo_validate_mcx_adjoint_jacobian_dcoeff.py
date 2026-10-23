"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validate_mcx_adjoint_jacobian_dcoeff.m

Frequency-Domain diffusion-coefficient Jacobian comparison: Redbird (FEM diffusion) vs
MCX (RF adjoint MC, outputtype='adjoint_dcoeff')

  J_D(r) = -grad(phi_src) . grad(phi_det) * dV

  Redbird: nodal J_D from the FEM forward solutions (rbjac in redbird-m), complex [1 x nnodes]
  MCX:     flux['jd'] from outputtype='adjoint_dcoeff' + omega>0, complex [60,60,30]

Agreement is expected in the diffusive regime (away from source/boundary).
This demo requires redbirdpy (pip install redbirdpy) and iso2mesh.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import redbirdpy as rb
import pmcx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbfemhelper import slabmesh, jacobians, cut2grid

## Shared optical parameters

freq = 200e6  # modulation frequency [Hz]
omega = 2 * np.pi * freq  # angular frequency [rad/s]

mua = 0.005  # absorption coeff [1/mm]
musp = 1.0  # reduced scattering [1/mm]
g = 0  # anisotropy (isotropic)
nmed = 1.37  # refractive index

## Redbird FEM setup  (domain [0,60]x[0,60]x[0,30] mm)

cfg, sd = slabmesh([20, 29.5, 0], [40, 30, 0], [[0, 0, 1, 1], [mua, musp, g, nmed]], omega)

## Redbird: solve FD forward from both source and detector, then compute J_D
print("Building Redbird FD system (f = %.0f MHz)..." % (freq / 1e6))
tic = time.time()
detphi_rb, phi_rb = rb.runforward(cfg)[:2]
detphi_rb = np.ravel(detphi_rb)
print("elapsed %.2f s" % (time.time() - tic))

# J_rb: node-based D-coefficient Jacobian, [1 x nnodes] complex
J_rb = jacobians(sd, phi_rb, cfg)[1]

print(
    "Redbird: detphi = %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)"
    % (detphi_rb[0].real, detphi_rb[0].imag, abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi)
)

## MCX RF adjoint setup  (same physical domain)

xcfg = {}
xcfg["nphoton"] = 1e8
xcfg["vol"] = np.ones([60, 60, 30], dtype="uint8")
xcfg["prop"] = [[0, 0, 1, 1], [mua, musp, g, nmed]]
xcfg["tstart"] = 0
xcfg["tend"] = 5e-9
xcfg["tstep"] = 5e-9
xcfg["autopilot"] = 1
xcfg["issrcfrom0"] = 1  # physical-mm coords; domain [0,60]x[0,60]x[0,30]
xcfg["isreflect"] = 1
xcfg["srctype"] = "pencil"
xcfg["srcpos"] = [20, 29.5, 0]  # matches Redbird srcpos
xcfg["srcdir"] = [0, 0, 1, 0]  # +z direction, focallength=0
xcfg["detpos"] = [[40, 30, 0, 1]]  # matches Redbird detpos, radius=1 mm
xcfg["detdir"] = [[0, 0, 1, 0]]  # +z (reversed photons enter medium)
xcfg["omega"] = omega  # RF modulation frequency
xcfg["outputtype"] = "adjoint_dcoeff"  # compute J_D = -grad(phi_src).grad(phi_det) per voxel
xcfg["issavedet"] = 0  # the detector only serves as an adjoint source, as mcxlab with one output

print("Running MCX RF adjoint (f = %.0f MHz, %.0e photons)..." % (freq / 1e6, xcfg["nphoton"]))
tic = time.time()
flux_adj = pmcx.mcxlab(xcfg)
print("elapsed %.2f s" % (time.time() - tic))

J_mcx = np.squeeze(flux_adj["jd"])  # [60, 60, 30] complex

## Extract xz cross-section at y = 29.5 mm
ym_idx = 29
x_mcx = np.arange(0.5, 60)
z_mcx = np.arange(0.5, 30)

J_mcx_grid = J_mcx[:, ym_idx, :].T  # [Nz=30, Nx=60] complex
J_rb_grid = cut2grid(cfg, J_rb[0], 29.5, x_mcx, z_mcx)

extent = [x_mcx[0], x_mcx[-1], z_mcx[-1], z_mcx[0]]
eps = np.finfo(float).eps


def panel(ax, img, title):
    im = ax.imshow(img, extent=extent)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)


## Figure 1/2: amplitude and phase maps
for func, name in ((lambda J: np.log10(np.abs(J) + eps), "log10|J_D|"), (lambda J: np.angle(J) * 180 / np.pi, "angle(J_D) (deg)")):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    panel(axs[0], func(J_mcx_grid), "MCX adjoint " + name)
    panel(axs[1], func(J_rb_grid), "Redbird FEM " + name)
    panel(axs[2], func(J_mcx_grid) - func(J_rb_grid), name + " difference (MCX - Redbird)")
    fig.suptitle("RF adjoint J_D comparison  |  f=%g MHz, mua=%.3f/mm, musp=%.1f/mm" % (freq / 1e6, mua, musp))

## Figure 3: contour overlay
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
clines_amp = np.concatenate([np.arange(-12, -5.9, 0.5), np.arange(-5.5, -3.9, 0.25)])
axs[0].contour(x_mcx, z_mcx, np.log10(np.abs(J_rb_grid) + eps), clines_amp, colors="r", linewidths=2)
axs[0].contour(x_mcx, z_mcx, np.log10(np.abs(J_mcx_grid) + eps), clines_amp, colors="b", linestyles="--", linewidths=2)
axs[0].set_title("log10|J_D| contours, red: Redbird FEM, blue: MCX adjoint")
clines_phs = np.arange(-180, 181, 10)
axs[1].contour(x_mcx, z_mcx, np.angle(J_rb_grid) * 180 / np.pi, clines_phs, colors="r", linewidths=2)
axs[1].contour(x_mcx, z_mcx, np.angle(J_mcx_grid) * 180 / np.pi, clines_phs, colors="b", linestyles="--", linewidths=2)
axs[1].set_title("angle(J_D) contours (deg)")
for ax in axs:
    ax.invert_yaxis()
    ax.set_aspect("equal")

## Figure 4: depth profiles at x = 29.5 mm
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
J_mcx_line, J_rb_line = J_mcx_grid[:, 29], J_rb_grid[:, 29]
axs[0].semilogy(z_mcx, np.abs(J_mcx_line) / np.nanmax(np.abs(J_mcx_line)), "b-", z_mcx, np.abs(J_rb_line) / np.nanmax(np.abs(J_rb_line)), "r--")
axs[0].legend(["MCX RF adjoint", "Redbird FEM"])
axs[0].set_title("|J_D| depth profile at x=29.5")
axs[1].plot(z_mcx, np.angle(J_mcx_line) * 180 / np.pi, "b-", z_mcx, np.angle(J_rb_line) * 180 / np.pi, "r--")
axs[1].set_title("angle(J_D) depth profile at x=29.5")

## Print summary
print("\n========== Summary ==========")
print("Frequency:  %.0f MHz" % (freq / 1e6))
print("mua=%.4f/mm,  musp=%.1f/mm,  n=%.2f" % (mua, musp, nmed))
print("\nRedbird detector value (src->det):  %.4e + %.4ei" % (detphi_rb[0].real, detphi_rb[0].imag))
print("  |amp| = %.4e,  phase = %.2f deg" % (abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi))
print("\nJ_rb  sum(real) = %.4e" % J_rb[0].real.sum())
print("J_mcx sum(real) = %.4e" % J_mcx.real.sum())
print("================================")
plt.show()
