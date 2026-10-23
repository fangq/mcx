"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validate_mcx_adjoint_mua_d.m

Dual-Jacobian comparison: Redbird (FEM diffusion) vs MCX (RF adjoint_mua_d)

Demonstrates outputtype='adjoint_mua_d', which computes BOTH J_mua and J_D
in a single MCX session (one set of forward/adjoint photon propagations).

  J_mua(r) = -phi_src(r) * phi_det(r) * dV      [mua sensitivity]
  J_D(r)   = -grad(phi_src) . grad(phi_det) * dV [diffusion-coeff sensitivity]

Output: flux['jmua'] and flux['jd'] are each complex [Nx, Ny, Nz, Ns*Nd]

Both are validated against Redbird FEM diffusion.  Agreement is expected in
the diffusive regime (away from source/boundary).
Geometry: slab domain [0,60]x[0,60]x[0,30] mm, 1 source, 1 detector.
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
g = 0  # anisotropy (isotropic, for diffusion comparison)
nmed = 1.37  # refractive index

## Redbird FEM setup  (domain [0,60]x[0,60]x[0,30] mm)

cfg, sd = slabmesh([20, 29.5, 0], [40, 30, 0], [[0, 0, 1, 1], [mua, musp, g, nmed]], omega)

## Redbird: one FD solve -> extract both J_mua and J_D
print("Building Redbird FD system (f = %.0f MHz)..." % (freq / 1e6))
tic = time.time()
detphi_rb, phi_rb = rb.runforward(cfg)[:2]
detphi_rb = np.ravel(detphi_rb)
print("elapsed %.2f s" % (time.time() - tic))

J_mua_rb, J_D_rb = jacobians(sd, phi_rb, cfg)  # each [1 x nnodes] complex

print(
    "Redbird: detphi = %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)"
    % (detphi_rb[0].real, detphi_rb[0].imag, abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi)
)

## MCX: single session with outputtype='adjoint_mua_d'

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
xcfg["outputtype"] = "adjoint_mua_d"  # single session: output J_mua + J_D together
xcfg["issavedet"] = 0  # the detector only serves as an adjoint source, as mcxlab with one output

print("Running MCX RF adjoint_mua_d (f = %.0f MHz, %.0e photons)..." % (freq / 1e6, xcfg["nphoton"]))
tic = time.time()
flux = pmcx.mcxlab(xcfg)
print("elapsed %.2f s" % (time.time() - tic))

J_mua_mcx = np.squeeze(flux["jmua"])  # [60, 60, 30] complex -- J_mua
J_D_mcx = np.squeeze(flux["jd"])  # [60, 60, 30] complex -- J_D

print("J_mua_mcx sum(real) = %.4e" % J_mua_mcx.real.sum())
print("J_D_mcx   sum(real) = %.4e" % J_D_mcx.real.sum())

## Interpolate Redbird results onto the MCX grid, xz cross-section at y = 29.5 mm
ym_idx = 29
x_mcx = np.arange(0.5, 60)
z_mcx = np.arange(0.5, 30)

J_mua_mcx_grid = J_mua_mcx[:, ym_idx, :].T  # [30 x 60]
J_D_mcx_grid = J_D_mcx[:, ym_idx, :].T
J_mua_rb_grid = cut2grid(cfg, J_mua_rb[0], 29.5, x_mcx, z_mcx)
J_D_rb_grid = cut2grid(cfg, J_D_rb[0], 29.5, x_mcx, z_mcx)

extent = [x_mcx[0], x_mcx[-1], z_mcx[-1], z_mcx[0]]
eps = np.finfo(float).eps
logamp = lambda J: np.log10(np.abs(J) + eps)
phase = lambda J: np.angle(J) * 180 / np.pi

## Figure 1/2: amplitude and phase maps -- J_mua (top row) and J_D (bottom row)
for func, kind in ((logamp, "log10|J|"), (phase, "angle(J) (deg)")):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for row, (Jm, Jr, name) in enumerate(((J_mua_mcx_grid, J_mua_rb_grid, "J_mua"), (J_D_mcx_grid, J_D_rb_grid, "J_D"))):
        for col, (img, title) in enumerate(
            ((func(Jm), "MCX " + name), (func(Jr), "Redbird " + name), (func(Jm) - func(Jr), name + " diff (MCX - Redbird)"))
        ):
            im = axs[row, col].imshow(img, extent=extent)
            axs[row, col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axs[row, col])
    fig.suptitle("Dual-Jacobian %s  |  f=%g MHz, mua=%.3f/mm, musp=%.1f/mm" % (kind, freq / 1e6, mua, musp))

## Figure 3: contour overlay -- banana shapes
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
clines_amp = np.concatenate([np.arange(-12, -5.9, 0.5), np.arange(-5.5, -3.9, 0.25)])
clines_phs = np.arange(-180, 181, 10)
for col, (Jm, Jr, name) in enumerate(((J_mua_mcx_grid, J_mua_rb_grid, "J_mua"), (J_D_mcx_grid, J_D_rb_grid, "J_D"))):
    axs[0, col].contour(x_mcx, z_mcx, logamp(Jr), clines_amp, colors="r", linewidths=2)
    axs[0, col].contour(x_mcx, z_mcx, logamp(Jm), clines_amp, colors="b", linestyles="--", linewidths=2)
    axs[0, col].set_title("log10|%s| contour (red: Redbird, blue: MCX)" % name, fontsize=9)
    axs[1, col].contour(x_mcx, z_mcx, phase(Jr), clines_phs, colors="r", linewidths=2)
    axs[1, col].contour(x_mcx, z_mcx, phase(Jm), clines_phs, colors="b", linestyles="--", linewidths=2)
    axs[1, col].set_title("angle(%s) contour (deg)" % name, fontsize=9)
for ax in axs.flat:
    ax.invert_yaxis()
    ax.set_aspect("equal")

## Figure 4: depth profiles at x = 29.5 mm (column 29)
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for row, (Jm, Jr, name) in enumerate(((J_mua_mcx_grid, J_mua_rb_grid, "J_mua"), (J_D_mcx_grid, J_D_rb_grid, "J_D"))):
    lm, lr = Jm[:, 29], Jr[:, 29]
    axs[row, 0].semilogy(z_mcx, np.abs(lm) / np.nanmax(np.abs(lm)), "b-", z_mcx, np.abs(lr) / np.nanmax(np.abs(lr)), "r--")
    axs[row, 0].legend(["MCX adjoint_mua_d", "Redbird FEM"])
    axs[row, 0].set_title("%s amplitude profile at x=29.5 mm" % name)
    axs[row, 1].plot(z_mcx, phase(lm), "b-", z_mcx, phase(lr), "r--")
    axs[row, 1].set_title("%s phase profile at x=29.5 mm" % name)

## Print summary
print("\n========== Summary ==========")
print("Frequency:  %.0f MHz" % (freq / 1e6))
print("mua=%.4f/mm,  musp=%.1f/mm,  n=%.2f" % (mua, musp, nmed))
print(
    "\nRedbird detector value:  %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)"
    % (detphi_rb[0].real, detphi_rb[0].imag, abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi)
)
print("\nJ_mua_rb  sum(real) = %.4e" % J_mua_rb[0].real.sum())
print("J_mua_mcx sum(real) = %.4e" % J_mua_mcx.real.sum())
print("\nJ_D_rb    sum(real) = %.4e" % J_D_rb[0].real.sum())
print("J_D_mcx   sum(real) = %.4e" % J_D_mcx.real.sum())
print("================================")
plt.show()
