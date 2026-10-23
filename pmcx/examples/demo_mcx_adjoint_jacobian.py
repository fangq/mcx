"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcx_adjoint_jacobian.m

Demonstrates adjoint Jacobian (banana-shaped sensitivity profiles) with
  outputtype='adjoint'  -- Jacobian J(r) = phi_src(r) * phi_det(r)
  srcid=-2              -- raw forward fluence for merged src+det set
Both CW (omega=0) and RF frequency-domain (omega>0) cases are shown.

Geometry (issrcfrom0=1, all positions in mm, voxel size 1 mm):
  S1 at x=20, S2 at x=80  (pencil beam, pointing +z into slab)
  D1 at x=38, D2 at x=62  (disk r=3 mm, same orientation)
  Cross-section to plot: y = Ny/2, showing the x-z plane.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## ---- Common geometry ------------------------------------------------
Nx, Ny, Nz = 100, 60, 50
ym = round(Ny / 2)  # mid-plane y position (0-based)

cfg = {}
cfg["nphoton"] = 3e7
cfg["vol"] = np.ones([Nx, Ny, Nz], dtype="uint8")
cfg["issrcfrom0"] = 1  # positions are 0-based
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["unitinmm"] = 1
# [mua  mus   g    n]  (medium 0 = background, 1 = tissue)
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1.0, 0.9, 1.37]]

# Two pencil sources as rows: S1 (row 1) and S2 (row 2), starting at z=0.5 inside voxel 0
cfg["srctype"] = "pencil"
cfg["srcpos"] = [[20, ym, 0.5], [80, ym, 0.5]]
cfg["srcdir"] = [[0, 0, 1, 0], [0, 0, 1, 0]]

# Detectors: [x  y  z  radius_mm]  and matching directions
cfg["detpos"] = [[38, ym, 0.5, 3], [62, ym, 0.5, 3]]
cfg["detdir"] = [[0, 0, 1, 0], [0, 0, 1, 0]]
cfg["issavedet"] = 0  # the detectors only serve as adjoint sources here; mcxlab saves no photons with one output

## ---- Part 1: CW Adjoint Jacobian ------------------------------------
cfg_adj = copy.deepcopy(cfg)
cfg_adj["outputtype"] = "adjoint"

print("Running CW adjoint (Ns=2, Nd=2 -> 4 Jacobians) ...")
f_cw = pmcx.mcxlab(cfg_adj)

# Output: f_cw['jmua'] is real [Nx, Ny, Nz, Ns*Nd]
#   [...,0] = J(S1,D1)   [...,1] = J(S1,D2)
#   [...,2] = J(S2,D1)   [...,3] = J(S2,D2)
J_cw = np.asarray(f_cw["jmua"], dtype=np.float64)
print("  CW adjoint size: %s   range: [%.2e, %.2e]" % (J_cw.shape, J_cw.min(), J_cw.max()))

## ---- Part 2: RF Adjoint Jacobian (omega>0 triggers RF forward) ------
cfg_rf = copy.deepcopy(cfg)
cfg_rf["outputtype"] = "adjoint"
cfg_rf["omega"] = 2 * np.pi * 100e6  # 100 MHz

print("Running RF adjoint at 100 MHz ...")
f_rf = pmcx.mcxlab(cfg_rf)

# Output: f_rf['jmua'] is complex [Nx, Ny, Nz, Ns*Nd]
J_rf = np.asarray(f_rf["jmua"])
print(
    "  RF adjoint size: %s   complex: %d   |range|: [%.2e, %.2e]"
    % (J_rf.shape, np.iscomplexobj(J_rf), np.abs(J_rf).min(), np.abs(J_rf).max())
)

## ---- Part 3: srcid=-2 merged forward fluences -----------------------
cfg_fwd = copy.deepcopy(cfg)
cfg_fwd["srcid"] = -2

print("Running srcid=-2 forward (Ns+Nd=4 merged sources) ...")
f_fwd = pmcx.mcxlab(cfg_fwd)

# Output: [Nx, Ny, Nz, maxgate, 1, Ns+Nd]  -- reshape to 4-D
phi = np.reshape(np.asarray(f_fwd["flux"], dtype=np.float64), [Nx, Ny, Nz, -1], order="F")
print("  srcid=-2 fluence size (after reshape): %s   range: [%.2e, %.2e]" % (phi.shape, phi.min(), phi.max()))
# phi[...,0] = S1 fluence    phi[...,1] = S2 fluence
# phi[...,2] = D1 fluence    phi[...,3] = D2 fluence

## ---- Pair labels and source/detector x positions --------------------
pair_labels = ["J(S1,D1)", "J(S1,D2)", "J(S2,D1)", "J(S2,D2)"]
sx = [20, 80]  # source x positions (0-based)
dx = [38, 62]  # detector x positions (0-based)
clim_log = [-10, 0]


def plot_xz_slice(ax, vol3d, ys, ttl, src_x, det_x):
    """plot log10 of an xz slice with source/detector markers"""
    sl = np.abs(vol3d[:, ys, :]).T  # [Nz, Nx]
    im = ax.imshow(np.log10(np.maximum(sl, 10 ** clim_log[0])), origin="lower", cmap="hot", aspect="auto")
    for x in np.atleast_1d(src_x):
        ax.plot(x, 0, "r^", markersize=9)
    for x in np.atleast_1d(det_x):
        ax.plot(x, 0, "bs", markersize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(ttl)
    plt.colorbar(im, ax=ax)


## ---- Figure 1: all four CW banana Jacobians -------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for k in range(4):
    plot_xz_slice(axs.flat[k], J_cw[..., k], ym, pair_labels[k], sx[k // 2], dx[k % 2])
fig.suptitle("CW Adjoint Jacobians (log10)")

## ---- Figure 2: CW vs RF for pair S1-D1 -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plot_xz_slice(axs[0], J_cw[..., 0], ym, "CW  J(S1,D1)", sx[0], dx[0])
plot_xz_slice(axs[1], np.abs(J_rf[..., 0]), ym, "|RF J(S1,D1)|  100 MHz", sx[0], dx[0])
ph = np.angle(J_rf[:, ym, :, 0]).T * 180 / np.pi
im = axs[2].imshow(ph, origin="lower", cmap="hsv", aspect="auto")
axs[2].plot(sx[0], 0, "r^", markersize=9)
axs[2].plot(dx[0], 0, "bs", markersize=9)
axs[2].set_title("Phase RF J(S1,D1)  (deg)")
plt.colorbar(im, ax=axs[2])
fig.suptitle("CW vs RF (100 MHz) Adjoint Jacobian -- Pair S1-D1")

## ---- Figure 3: individual fluences from srcid=-2 --------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
vol_labels = ["S1 fluence", "S2 fluence", "D1 adj-src fluence", "D2 adj-src fluence"]
markers = [(sx[0], "r^"), (sx[1], "r^"), (dx[0], "bs"), (dx[1], "bs")]
for k in range(4):
    ax = axs.flat[k]
    im = ax.imshow(np.log10(np.maximum(phi[:, ym, :, k].T, 10 ** clim_log[0])), origin="lower", cmap="hot", aspect="auto")
    ax.plot(markers[k][0], 0, markers[k][1], markersize=9)
    ax.set_title(vol_labels[k])
    plt.colorbar(im, ax=ax)
fig.suptitle("srcid=-2: Forward fluence for each merged source/detector")

## ---- Figure 4: consistency check  J_CW ~ phi_S1 * phi_D1 -----------
Jprod = phi[..., 0] * phi[..., 2]  # S1 x D1 direct product

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plot_xz_slice(axs[0], J_cw[..., 0], ym, "J(S1,D1)  adjoint mode", sx[0], dx[0])
plot_xz_slice(axs[1], Jprod, ym, "phi_S1 * phi_D1  srcid=-2 product", sx[0], dx[0])
fig.suptitle("Adjoint consistency: J(S1,D1) vs phi_S1 * phi_D1")
print("\nDone.")
plt.show()
