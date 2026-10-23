"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_srctype.m

 wide-field source tests

The 3-D slice views of the MATLAB demo are shown here as 2-D cross-sections.
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx


def run(cfg):
    """run a simulation and return the time-integrated fluence (1/mm^2)"""
    return pmcx.mcxlab(cfg)["flux"] * cfg["tstep"]


def show(ax, img, title):
    with np.errstate(divide="ignore"):
        im = ax.imshow(np.log10(np.abs(img)))
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)


## test group 1

# a regular pencil beam at the center of the volume
cfg = {}
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcpos"] = [30, 30, 30]
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
# cfg['printnum']=10
cfg["seed"] = 99999
cfg["srctype"] = "pencil"
fcw = run(cfg)

fig, axs = plt.subplots(2, 2, figsize=(10, 9))
show(axs[0, 0], fcw[:, 29, :, 0], "pencil beam at volume center")

# an isotropic source at the center of the volume
cfg["srctype"] = "isotropic"
fcw = run(cfg)
show(axs[0, 1], fcw[:, 29, :, 0], "isotropic source at volume center")

# a pencil beam outside the volume
cfg["srctype"] = "pencil"
cfg["srcpos"] = [30, 30, -10]
fcw = run(cfg)
show(axs[1, 0], fcw[:, 29, :, 0], "pencil beam launched from outside the volume")

# an isotropic source outside the volume
cfg["srctype"] = "isotropic"
cfg["srcpos"] = [30, 30, -10]
cfg["tend"] = 1e-9
cfg["tstep"] = 1e-9
fcw = run(cfg)
show(axs[1, 1], fcw[:, 29, :, 0], "isotrpoic source at [30 30 -10]")

## test group 2

cfg = {}
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcdir"] = [0, 0, 1, 0]
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999

# a uniform planar source outside the volume
cfg["srctype"] = "planar"
cfg["srcpos"] = [10, 10, 0]
cfg["srcparam1"] = [40, 0, 0, 0]
cfg["srcparam2"] = [0, 40, 0, 0]
cfg["tend"] = 0.4e-10
cfg["tstep"] = 0.4e-10
fcw = run(cfg)
show(axs[0, 0], fcw[:, :, 0, 0], "a uniform planar source")

cfg["srctype"] = "fourier"
cfg["srcparam1"] = [40, 10, 0, 2]
cfg["srcparam2"] = [0, 40, 0, 2]
fcw = run(cfg)
show(axs[0, 1], fcw[:, :, 0, 0], "an SFDI source (2,2) in a quadrilateral")

cfg["srctype"] = "fourier"
cfg["srcpos"] = [0, 0, 70]
cfg["srcdir"] = [0, 0, -1]
cfg["srcparam1"] = [60, 0, 0, 2]
cfg["srcparam2"] = [0, 60, 0, 1]
cfg["tend"] = 1e-9
cfg["tstep"] = 1e-9
fcw = run(cfg)
show(axs[1, 0], fcw[:, :, 59, 0], "a spatial frequency domain source (2,1)")

mcximg = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

cfg["nphoton"] = 1e7
cfg["srctype"] = "pattern"
cfg["srcpattern"] = mcximg
cfg["srcpos"] = [-10 * np.sqrt(2), 0, 40]
cfg["srcdir"] = list(np.array([1, 1, 0]) / np.sqrt(2))
cfg["srcparam1"] = [20 / np.sqrt(2), -20 / np.sqrt(2), 0, mcximg.shape[0]]
cfg["srcparam2"] = [0, 0, -15, mcximg.shape[1]]
cfg["tend"] = 2e-10
cfg["tstep"] = 2e-10
cfg["voidtime"] = 0
fcw = run(cfg)
show(axs[1, 1], fcw[:, :, 39, 0], "an arbitrary pattern source from an angle")

## test group 3

cfg = {}
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
cfg["nphoton"] = 1e8
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["srcdir"] = [0, 0, 1]
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999

# a uniform planar source outside the volume
cfg["srctype"] = "fourierx"
cfg["srcpos"] = [10, 10, -1]
cfg["srcparam1"] = [40, 0, 0, 40]
cfg["srcparam2"] = [2, 1.5, 0, 0]
cfg["tend"] = 0.4e-10
cfg["tstep"] = 0.4e-10
fcw = run(cfg)
show(axs[0, 0], fcw[:, :, 0, 0], "a general Fourier source (2,1.5)")

# a uniform planar source outside the volume
cfg["srctype"] = "slit"
cfg["srcpos"] = [10, 30, 0]
cfg["srcdir"] = list(np.array([0, 1, 1]) / np.sqrt(2))
cfg["srcparam1"] = [40, 0, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
cfg["prop"] = [[0, 0, 1, 1], [0.005, 0.1, 0.9, 1.37]]
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
fcw = run(cfg)
show(axs[0, 1], fcw[:, 44, :, 0], "a slit source")

cfg["nphoton"] = 1e7
vol = np.ones([60, 60, 60], dtype="uint8")
vol[24:35, 24:35, 24:35] = 2
cfg["vol"] = vol
cfg["prop"] = [[0, 0, 1, 1], [0.005, 0.01, 0.8, 1.37], [0.01, 1, 0.8, 1.37]]
cfg["srctype"] = "pattern"
cfg["srcpattern"] = mcximg[0:6, :]
cfg["srcpos"] = [-13, 13, 13]
cfg["srcdir"] = [1, 0, 0]
cfg["srcparam1"] = [0, 30, 0, cfg["srcpattern"].shape[0]]
cfg["srcparam2"] = [0, 0, 30, cfg["srcpattern"].shape[1]]
cfg["tend"] = 0.3e-9
cfg["tstep"] = 0.1e-10
cfg["voidtime"] = 0
fcw1 = run(cfg)

cfg["srcpattern"] = np.rot90(mcximg[6:12, :], 3)
cfg["srcpos"] = [17, 17, 60 + 1]
cfg["srcdir"] = [0, 0, -1]
cfg["srcparam1"] = [30, 0, 0, cfg["srcpattern"].shape[0]]
cfg["srcparam2"] = [0, 30, 0, cfg["srcpattern"].shape[1]]
fcw2 = run(cfg)

cfg["srcpattern"] = mcximg[12:, :]
cfg["srcpos"] = [60 - 15, -1, 60 - 15]
cfg["srcdir"] = [0, 1, 0]
cfg["srcparam1"] = [-30, 0, 0, cfg["srcpattern"].shape[0]]
cfg["srcparam2"] = [0, 0, -30, cfg["srcpattern"].shape[1]]
fcw3 = run(cfg)

fcw = np.sum(fcw1 + fcw2 + fcw3, axis=3)
show(axs[1, 0], fcw[:, :, 59], "three pattern sources from three sides")

## volumetric source
Rsrc = 20
xi, yi, zi = np.meshgrid(*([np.arange(-Rsrc, Rsrc + 1)] * 3), indexing="ij")
sphsrc = (xi * xi + yi * yi + zi * zi) <= Rsrc * Rsrc

dim = 60

# define source pattern
cfg = {}
# basic settings
cfg["nphoton"] = 1e7
cfg["tstart"] = 0
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
cfg["respin"] = 1
cfg["seed"] = 99999
cfg["outputtype"] = "energy"  # should get the energy deposits in each voxel

cfg["srctype"] = "pattern3d"
cfg["srcpattern"] = sphsrc.astype(np.float32)
cfg["srcparam1"] = list(sphsrc.shape) + [0]
cfg["srcpos"] = [10, 10, 20]
cfg["srcdir"] = [0, 0, 1, np.nan]
cfg["autopilot"] = 1

# define volume and inclusions
cfg["vol"] = np.ones([dim, dim, dim], dtype="uint8")
## you can use a JSON string to define cfg['shapes']
cfg["shapes"] = '{"Shapes":[{"Sphere":{"O":[25,21,10],"R":10,"Tag":2}}]}'
cfg["prop"] = [
    [0, 0, 1, 1],  # Boundary
    [0.003, 0.03, 0.8, 1],  # Tissue mua = 0.003 mus = 0.03 n = 1.85/1.37
    [0.006, 0.09, 0.8, 1],  # Cancer mus  = 1.09 n = 1.37
    [1.3, 0.8, 0.8, 1],  # TAM mua = 0.003, n = 0.095
]
cfg["unitinmm"] = 1e-3

flux = pmcx.mcxlab(cfg)["flux"].sum(axis=3)
show(axs[1, 1], flux[:, 39, :], "a volumetric spherical source")

## test group 4

cfg = {}
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999

# a cone beam
cfg["srctype"] = "cone"
cfg["srcpos"] = [30, 30, -10]
cfg["srcdir"] = [0, 0, 1]
cfg["tend"] = 5e-11
cfg["tstep"] = 5e-11
cfg["srcparam1"] = [np.pi / 6, 0, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
fcw = run(cfg)
show(axs[0, 0], fcw[:, :, 0, 0], "a uniform cone beam (uniform solid-angle)")

# a beam with arcsine distribution profile
cfg["srctype"] = "arcsine"
fcw = run(cfg)
show(axs[0, 1], fcw[:, :, 0, 0], "an arcsine-distribution beam")

# a uniform disk source
cfg["srctype"] = "disk"
cfg["srcparam1"] = [20, 0, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
fcw = run(cfg)
show(axs[1, 0], fcw[:, :, 0, 0], "a uniform disk source")

# a gaussian beam source
cfg["srctype"] = "gaussian"
cfg["srcparam1"] = [10, 0, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
cfg["tend"] = 5e-11
cfg["tstep"] = 5e-11
fcw = run(cfg)
show(axs[1, 1], fcw[:, :, 0, 0], "a gaussian beam source")

## test group 5

cfg = {}
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
cfg["nphoton"] = 1e6
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 1, 0.8, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999
cfg["tend"] = 5e-11
cfg["tstep"] = 5e-11

# a ring beam
cfg["srctype"] = "ring"
cfg["srcpos"] = [30, 30, -10]
cfg["srcdir"] = [0, 0, 1]
cfg["srcparam1"] = [15, 10, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
fcw = run(cfg)
show(axs[0, 0], fcw[:, :, 0, 0], "a ring beam")

# a diverging ring sector beam
cfg["srctype"] = "ring"
cfg["srcpos"] = [30, 30, -10]
cfg["srcdir"] = [0, 0, 1, -20]
cfg["srcparam1"] = [15, 10, 0, 2 * np.pi / 3]
cfg["srcparam2"] = [0, 0, 0, 0]
fcw = run(cfg)
show(axs[0, 1], fcw[:, :, 0, 0], "a ring-sector beam")

# a pencil beam array
cfg["srctype"] = "pencilarray"
cfg["srcpos"] = [10, 10, 0]
cfg["srcdir"] = [0, 0, 1]
cfg["srcparam1"] = [40, 0, 0, 4]
cfg["srcparam2"] = [0, 40, 0, 5]
fcw = run(cfg)
show(axs[1, 0], fcw[:, :, 0, 0], "a pencil beam array")

## test group 6

cfg = {}
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
cfg["nphoton"] = 1e7
cfg["vol"] = np.ones([60, 60, 60], dtype="uint8")
cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["prop"] = [[0, 0, 1, 1], [0.005, 0.01, 1, 1.37]]
cfg["tstart"] = 0
cfg["seed"] = 99999

# a ring beam
cfg["srctype"] = "disk"
cfg["srcpos"] = [30, 30, -10]
cfg["srcdir"] = [0, 0, 1]
cfg["tend"] = 5e-11
cfg["tstep"] = 5e-11
cfg["srcparam1"] = [20, 15, 0, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
fcw = run(cfg)
show(axs[0, 0], fcw[:, :, 0, 0], "a disk/ring source")

# a hyperboloid source
cfg["srctype"] = "hyperboloid"
cfg["srcpos"] = [30, 30, 5]
cfg["srcparam1"] = [2, 20, 4, 0]
cfg["srcparam2"] = [0, 0, 0, 0]
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
fcw = run(cfg)
show(axs[0, 1], fcw[:, 29, :, 0], "a hyperboloid source")

# x-direction broadened slit source
cfg["srctype"] = "slit"
cfg["srcpos"] = [10, 30, 0]
cfg["srcdir"] = list(np.array([0, 1, 1]) / np.sqrt(2))
cfg["srcparam1"] = [40, 0, 0, 0]
cfg["srcparam2"] = [0.2, 0, 0, 0]
cfg["prop"] = [[0, 0, 1, 1], [0.005, 0.01, 0.9, 1.37]]
cfg["tend"] = 5e-9
cfg["tstep"] = 5e-9
fcw = run(cfg)
show(axs[1, 0], fcw[:, 44, :, 0], "x-axis brodened slit source")

# x/y-direction broadened slit source
cfg["srctype"] = "slit"
cfg["srcpos"] = [10, 30, 0]
cfg["srcdir"] = list(np.array([0, 1, 1]) / np.sqrt(2))
cfg["srcparam1"] = [40, 0, 0, 0]
cfg["srcparam2"] = [0.1, 0.3, 0, 0]
fcw = run(cfg)
show(axs[1, 1], fcw[:, 44, :, 0], "x/y brodened slit source")

## test group 7

# debug flag to retrieve/test build-in RNG
cfg["vol"] = np.ones([100, 100, 100], dtype="uint8")
cfg["debuglevel"] = "R"
cfg["isnormalized"] = 0
rng = pmcx.mcxlab(cfg)["flux"].ravel()
plt.figure()
plt.hist(rng, 1000)
plt.title("raw RNG distribution")
del cfg["debuglevel"]
plt.show()
