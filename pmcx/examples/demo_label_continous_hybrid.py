"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_label_continous_hybrid.m

Tissue-type based continuously varying media: each voxel stores a 3-element
float32 record [new value, property index (0-mua,1-mus,2-g,3-n), label], so
that one optical property of the voxel's tissue type is replaced voxel by voxel.

This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pmcx


def hybrid(replace_property, property_substitute, label):
    # concatenated as a 4-D single array, the 1st dimension holds the 3 components
    return np.stack([replace_property, property_substitute, label]).astype(np.float32)


def compare(conventional, new, tstep, clines, title):
    plt.figure()
    plt.contourf(np.log10(conventional["flux"][30, :, :, 0] * tstep), clines, cmap="Reds")
    plt.contour(np.log10(new["flux"][30, :, :, 0] * tstep), clines, colors="w", linestyles="--")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.title(title + " (filled: conventional, dashed: new)")


## case 1: homogenous media, update the refractive index from n_old to n_new for all voxels

n_old = 1.37
n_new = 1.55

cfg1 = {}
cfg1["nphoton"] = 1e8
cfg1["srcpos"] = [30, 30, 1]
cfg1["srcdir"] = [0, 0, 1]
cfg1["gpuid"] = 1
cfg1["autopilot"] = 1
cfg1["tstart"] = 0
cfg1["tend"] = 5e-9
cfg1["tstep"] = 5e-9
cfg1["isreflect"] = 1

# conventional way to define optical properties: label each voxel with
# tissue types
cfg1["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg1["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, n_new]]
flux1_conventional = pmcx.mcxlab(copy.deepcopy(cfg1))

# tissue-type based continuous varying media
cfg1["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, n_old]]  # n=1.37 will be updated during simulation
shape = cfg1["vol"].shape
property_substitute = 3 * np.ones(shape)  # 0-update mua, 1-update mus, 2-update g, 3-update n
label = np.ones(shape)  # label-based tissue type
replace_property = n_new * np.ones(shape)  # values to substitute corresponding optical properties
cfg1["vol"] = hybrid(replace_property, property_substitute, label)
flux1_new = pmcx.mcxlab(cfg1)

compare(flux1_conventional, flux1_new, cfg1["tstep"], np.arange(-7, 0, 1), "homogeneous cube")

## case 2: Two-layer slab: The refractive indices of tissue type 1 and 2 will be updated to 1.45 and 1.6 respectively

n_old_1 = 1.37
n_new_1 = 1.45

n_old = 1.37
n_new_2 = 1.6

cfg2 = copy.deepcopy(cfg1)

# conventional way to define optical properties: label each voxel with
# tissue types
vol = np.ones([60, 60, 60], dtype="uint8")
vol[:, :, 30:60] = 2
cfg2["vol"] = vol
cfg2["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, n_new_1], [0.01, 2, 0, n_new_2]]
flux2_conventional = pmcx.mcxlab(copy.deepcopy(cfg2))

# tissue-type based continuous varying media
cfg2["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, n_old_1], [0.01, 2, 0, n_old]]  # n=1.37 will be updated during the simulation
property_substitute = 3 * np.ones(shape)  # 0-update mua, 1-update mus, 2-update g, 3-update n
label = np.ones(shape)  # label-based tissue type
label[:, :, 30:60] = 2
replace_property = n_new_1 * np.ones(shape)  # values to substitute corresponding optical properties
replace_property[:, :, 30:60] = n_new_2
cfg2["vol"] = hybrid(replace_property, property_substitute, label)
flux2_new = pmcx.mcxlab(cfg2)

compare(flux2_conventional, flux2_new, cfg2["tstep"], np.arange(-10, -1.5, 0.5), "two layer slab")

## case 3: multi-layer slab: mua of tissue type 1 will be updated to 0.05 while the refractive indices
## of tissue type 2 will be updated to [1.30,1.35,1.40,1.45,1.50,1.55]

mua_old_1 = 0.005
mua_new_1 = 0.05

n_old = 1.37
n_new = [1.30, 1.35, 1.40, 1.45, 1.50, 1.55]

cfg3 = copy.deepcopy(cfg1)

# conventional way to define optical properties: label each voxel with
# tissue types
vol = np.ones([60, 60, 60], dtype="uint8")
for i in range(1, 7):  # each layer
    vol[:, :, 30 + (i - 1) * 5 : 30 + i * 5] = i + 1
cfg3["vol"] = vol
cfg3["prop"] = [[0, 0, 1, 1], [mua_new_1, 1, 0, 1.37]] + [[0.01, 2, 0, n] for n in n_new]
flux3_conventional = pmcx.mcxlab(copy.deepcopy(cfg3))

# tissue-type based continuous varying media
cfg3["prop"] = [[0, 0, 1, 1], [0.005, 1, 0, 1.37], [0.01, 2, 0, n_old]]  # n=1.37 will be updated during the simulation
property_substitute = np.zeros(shape)  # 0-update mua, 1-update mus, 2-update g, 3-update n
property_substitute[:, :, 30:60] = 3
label = np.ones(shape)  # label-based tissue type 0-255
label[:, :, 30:60] = 2
replace_property = mua_new_1 * np.ones(shape)  # values to substitute corresponding optical properties
for i, n in enumerate(n_new):
    replace_property[:, :, 30 + i * 5 : 35 + i * 5] = n
cfg3["vol"] = hybrid(replace_property, property_substitute, label)
flux3_new = pmcx.mcxlab(cfg3)

compare(flux3_conventional, flux3_new, cfg3["tstep"], np.arange(-12, -1, 1), "multi-layered slab")
plt.show()
