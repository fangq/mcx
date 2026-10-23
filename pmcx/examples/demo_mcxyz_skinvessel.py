"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxyz_skinvessel.m

 mcxyz skinvessel benchmark

 must change mcxyz maketissue.m boundaryflag variable from 2 to 1 to get
 comparable absorption fraction (40%), otherwise, mcxyz obtains slightly
 higher absorption (~42%) with boundaryflag=2
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

cfg = {}
cfg["vol"] = np.zeros([200, 200, 200], dtype="uint8")
cfg["shapes"] = (
    '{"Shapes":[{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]},'
    '{"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}]}'
)
cfg["unitinmm"] = 0.005
cfg["prop"] = [
    [0.0000, 0.0, 1.0000, 1],
    [3.5640e-05, 1.0000, 1.0000, 1.3700],
    [23.0543, 9.3985, 0.9000, 1.3700],
    [0.0458, 35.6541, 0.9000, 1.3700],
    [1.6572, 37.5940, 0.9000, 1.3700],
]

cfg["nphoton"] = 1e8
cfg["issrcfrom0"] = 1
cfg["srcpos"] = [100, 100, 20]
cfg["tstart"] = 0
cfg["tend"] = 5e-8
cfg["tstep"] = 5e-8
cfg["srcdir"] = [0, 0, 1]
cfg["srctype"] = "disk"
cfg["srcparam1"] = [0.3 / cfg["unitinmm"], 0, 0, 0]
cfg["isreflect"] = 0
cfg["autopilot"] = 1
cfg["gpuid"] = 1
cfg["debuglevel"] = "P"

# cfg['outputtype']='energy'
cfg["outputtype"] = "flux"
flux = pmcx.mcxlab(cfg)["flux"][:, :, :, 0]

# convert mcx solution to mcxyz's output
# 'energy': mcx outputs normalized energy deposition, must convert
# it to normalized energy density (1/cm^3) as in mcxyz
# 'flux': cfg['tstep'] is used in mcx's fluence normalization, must
# undo 100 converts 1/mm^2 from mcx output to 1/cm^2 as in mcxyz
if cfg["outputtype"] == "energy":
    mcxdata = flux / ((cfg["unitinmm"] / 10) ** 3)
else:
    mcxdata = flux * 100

if cfg["outputtype"] == "flux":
    mcxdata = mcxdata * cfg["tstep"]

plt.figure()
dim = cfg["vol"].shape
yi = (np.arange(1, dim[1] + 1) - dim[1] // 2) * cfg["unitinmm"]
zi = np.arange(1, dim[2] + 1) * cfg["unitinmm"]

clim = [-2.4429, 4.7581] if cfg["outputtype"] == "energy" else [0.5, 2.8]
plt.imshow(
    np.log10(np.abs(mcxdata[99, :, :])).T, extent=[yi[0], yi[-1], zi[-1], zi[0]], cmap="jet", vmin=clim[0], vmax=clim[1]
)
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()
