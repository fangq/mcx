"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_validate_mcx_adjoint_jacobian.m

Frequency-Domain Jacobian comparison: Redbird (FEM diffusion) vs MCX (RF adjoint MC)

Both compute the adjoint Jacobian J(r) = phi_src(r) * phi_det(r) per voxel,
where phi_src is the forward fluence from the source and phi_det is the
(adjoint) forward fluence launched from the detector position.

  Redbird: J_rb(r) from the FEM forward solutions, complex [1 x nnodes]
  MCX:     J_mcx(r) from outputtype='adjoint' + omega>0,
           output is complex [Nx,Ny,Nz] = [60,60,30]

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

## Redbird: solve FD forward + compute Jacobian
##  runforward solves from BOTH source and detector positions.
##  phi_rb is [nnodes x 2]: col 0 = phi from src, col 1 = phi from det.

print("Building Redbird FD system (f = %.0f MHz)..." % (freq / 1e6))
tic = time.time()
detphi_rb, phi_rb = rb.runforward(cfg)[:2]
detphi_rb = np.ravel(detphi_rb)
print("elapsed %.2f s" % (time.time() - tic))

# J_rb: node-based Jacobian, size [nsd_pairs x nnodes] = [1 x nnodes], complex
J_rb = jacobians(sd, phi_rb, cfg)[0]

print(
    "Redbird: detphi = %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)"
    % (detphi_rb[0].real, detphi_rb[0].imag, abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi)
)

## MCX RF adjoint setup  (same physical domain)
##  outputtype='adjoint' + omega>0:
##    - all Ns+Nd sources (source + detector acting as reversed source)
##      are simulated together (srcid=-1 internally)
##    - adjoint kernel runs after normalization:
##        J(r) = phi_src(r) * phi_det(r)   (complex product for RF)

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
xcfg["detdir"] = [[0, 0, 1, 0]]  # +z (reversed photons enter medium), focal-length similar to srcdir[3]
# xcfg['detdir'][0][3] = -np.inf  # det launches photon in a Lambertian profile, more realistic
# xcfg['detdir'][0][3] = np.nan   # det launches photon isotropically, receiving from all angles
xcfg["omega"] = omega  # RF modulation frequency
xcfg["outputtype"] = "adjoint"  # compute J = phi_src * phi_det per voxel
xcfg["issavedet"] = 0  # the detector only serves as an adjoint source, as mcxlab with one output

print("Running MCX RF adjoint (f = %.0f MHz, %.0e photons)..." % (freq / 1e6, xcfg["nphoton"]))
tic = time.time()
flux_adj = pmcx.mcxlab(xcfg)
print("elapsed %.2f s" % (time.time() - tic))

# flux_adj['jmua'] is complex [60, 60, 30, 1] (Ns*Nd=1 source-detector pair)
J_mcx = np.squeeze(flux_adj["jmua"])  # [60, 60, 30] complex

## Extract xz cross-section at y = 29.5 mm (voxel index 29, 0-based, with issrcfrom0=1)

ym_idx = 29
x_mcx = np.arange(0.5, 60)  # voxel center x
z_mcx = np.arange(0.5, 30)  # voxel center z

J_mcx_grid = J_mcx[:, ym_idx, :].T  # [Nz=30, Nx=60] complex
J_rb_grid = cut2grid(cfg, J_rb[0], 29.5, x_mcx, z_mcx)  # [Nz=30, Nx=60] complex

extent = [x_mcx[0], x_mcx[-1], z_mcx[-1], z_mcx[0]]
eps = np.finfo(float).eps


def panel(ax, img, title):
    im = ax.imshow(img, extent=extent)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)


## Figure 1: Jacobian amplitude maps
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
panel(axs[0], np.log10(np.abs(J_mcx_grid) + eps), "MCX adj |J| log10 (f=%g MHz)" % (freq / 1e6))
panel(axs[1], np.log10(np.abs(J_rb_grid) + eps), "Redbird FEM |J| log10 (f=%g MHz)" % (freq / 1e6))
panel(axs[2], np.log10(np.abs(J_mcx_grid) + eps) - np.log10(np.abs(J_rb_grid) + eps), "log10|J| difference (MCX - Redbird)")
fig.suptitle("RF adjoint |J| comparison  |  f=%g MHz, mua=%.3f/mm, musp=%.1f/mm" % (freq / 1e6, mua, musp))

## Figure 2: Jacobian phase maps
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
panel(axs[0], np.angle(J_mcx_grid) * 180 / np.pi, "MCX adj angle(J) (deg)")
panel(axs[1], np.angle(J_rb_grid) * 180 / np.pi, "Redbird FEM angle(J) (deg)")
panel(axs[2], (np.angle(J_mcx_grid) - np.angle(J_rb_grid)) * 180 / np.pi, "Phase difference MCX - Redbird (deg)")
fig.suptitle("RF adjoint phase(J) comparison  |  f=%g MHz" % (freq / 1e6))

## Figure 3: Contour overlay -- banana shape
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
clines_amp = np.concatenate([np.arange(-12, -5.9, 0.5), np.arange(-5.5, -3.9, 0.25)])
axs[0].contour(x_mcx, z_mcx, np.log10(np.abs(J_rb_grid) + eps), clines_amp, colors="r", linewidths=2)
axs[0].contour(x_mcx, z_mcx, np.log10(np.abs(J_mcx_grid) + eps), clines_amp, colors="b", linestyles="--", linewidths=2)
axs[0].set_title("log10|J| contours, red: Redbird FEM, blue: MCX adjoint")
clines_phs = np.arange(0, 271, 10)
axs[1].contour(x_mcx, z_mcx, np.angle(J_rb_grid) * 180 / np.pi, clines_phs, colors="r", linewidths=2)
axs[1].contour(x_mcx, z_mcx, np.angle(J_mcx_grid) * 180 / np.pi, clines_phs, colors="b", linestyles="--", linewidths=2)
axs[1].set_title("angle(J) contours (deg)")
for ax in axs:
    ax.invert_yaxis()
    ax.set_aspect("equal")

## Figure 4: Depth profiles at x = 29.5 mm (source x), grid column 29
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
J_mcx_line, J_rb_line = J_mcx_grid[:, 29], J_rb_grid[:, 29]
axs[0].semilogy(z_mcx, np.abs(J_mcx_line) / np.nanmax(np.abs(J_mcx_line)), "b-", z_mcx, np.abs(J_rb_line) / np.nanmax(np.abs(J_rb_line)), "r--")
axs[0].legend(["MCX RF adjoint", "Redbird FEM"])
axs[0].set_xlabel("Depth z (mm)")
axs[0].set_title("Amplitude depth profile at x=29.5")
axs[1].plot(z_mcx, np.angle(J_mcx_line) * 180 / np.pi, "b-", z_mcx, np.angle(J_rb_line) * 180 / np.pi, "r--")
axs[1].set_xlabel("Depth z (mm)")
axs[1].set_ylabel("Phase (deg)")
axs[1].set_title("Phase depth profile at x=29.5")

## Print summary
print("\n========== Summary ==========")
print("Frequency:  %.0f MHz" % (freq / 1e6))
print("mua=%.4f/mm,  musp=%.1f/mm,  n=%.2f" % (mua, musp, nmed))
print("\nRedbird detector value (src->det):  %.4e + %.4ei" % (detphi_rb[0].real, detphi_rb[0].imag))
print("  |amp| = %.4e,  phase = %.2f deg" % (abs(detphi_rb[0]), np.angle(detphi_rb[0]) * 180 / np.pi))
print("\nJ_rb  sum(real) = %.4e" % J_rb[0].real.sum())
print("J_mcx sum(real) = %.4e" % J_mcx.real.sum())
print("================================")
print("Note: MCX (transport, RTE) and Redbird (diffusion equation)")
print("agree well in the diffusive regime (far from source/boundary).")
print("Deviations near the source are expected.")
plt.show()
