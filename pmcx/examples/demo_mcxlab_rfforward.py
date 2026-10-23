"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_rfforward.m

Frequency-domain forward MC with MCX

Eq.9: w <- w * exp[-(mu_a + i*omega*n/c0) * ds]
Leino, Pulkkinen, and Tarvainen, OSA Continuum 2(3), 957-965 (2019)

Key setting: cfg['omega'] = 2*pi*freq  (no special outputtype needed for forward mode)
When seed is NOT from file and omega > 0, MCX automatically runs RF forward mode.
The output is a complex-valued fluence volume.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx

## Common setup
cfg = {}
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 1]
cfg["srcdir"] = [0, 0, 1]
cfg["issrcfrom0"] = 1
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["prop"] = np.array([[0, 0, 1, 1], [0.01, 10, 0.9, 1.37]])

## CW simulation
cfg_cw = copy.deepcopy(cfg)
cfg_cw["outputtype"] = "flux"
print("Running CW ...")
fluence_cw = np.squeeze(pmcx.mcxlab(cfg_cw)["flux"])

## RF forward at 100 MHz
cfg_rf = copy.deepcopy(cfg)
cfg_rf["omega"] = 2 * np.pi * 100e6
print("Running RF forward at %.0f MHz ..." % (cfg_rf["omega"] / (2 * np.pi) / 1e6))
fluence_rf = pmcx.mcxlab(cfg_rf)

## The output is complex
phi = np.squeeze(fluence_rf["flux"])
print("Complex output: %d, size: %s" % (np.iscomplexobj(phi), str(phi.shape)))

amp = np.abs(phi)
phs = np.angle(phi) * 180 / np.pi

## Plot
z = 29
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, img, title in [
    (axs[0], np.log10(np.maximum(fluence_cw[:, :, z], 1e-12)), "CW log10(fluence)"),
    (axs[1], np.log10(np.maximum(amp[:, :, z], 1e-12)), "log10-amp of J_FD(mua)"),
    (axs[2], phs[:, :, z], "Phase of J_FD(mua) (deg)"),
]:
    im = ax.imshow(img)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
fig.suptitle("MCX RF Forward  f=%g MHz  mua=%.3f/mm" % (cfg_rf["omega"] / (2 * np.pi) / 1e6, cfg["prop"][1, 0]))

## Depth profiles
fig, ax1 = plt.subplots()
zz = np.arange(1, 61)
ax1.semilogy(zz, fluence_cw[29, 29, :], "b-", linewidth=1.5, label="CW")
ax1.semilogy(zz, amp[29, 29, :], "r--", linewidth=1.5, label="FD Amp")
ax1.set_ylabel("Fluence / Amplitude")
ax2 = ax1.twinx()
ax2.plot(zz, phs[29, 29, :], "g-", linewidth=1.5, label="Phase")
ax2.set_ylabel("Phase (deg)")
ax1.set_xlabel("Depth (voxels)")
ax1.set_title("Depth profile x=30 y=30")
fig.legend()
plt.show()
