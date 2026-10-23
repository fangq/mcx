"""
pmcx - Monte Carlo eXtreme for Python, ported from mcxlab/examples/demo_mcxlab_2d.m

A sample 2D MCX simulation

Author: Qianqian Fang <q.fang at neu.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
import pmcx

# you must define a 3D array with one singleton dimension (with length 1);
# here the 1st dimension is the singleton one, as in the MATLAB demo

cfg = {}
cfg["nphoton"] = 1e6

vol = np.ones([1, 100, 100], dtype="uint8")  # from 2d to 3d
vol[0, 29:70, 9:50] = 2
cfg["vol"] = vol
cfg["issrcfrom0"] = 1
cfg["srctype"] = "pencil"

cfg["srcpos"] = [0, 50, 0]  # src position must be located in the 2D plane
cfg["srcdir"] = [0, 0, 1]  # src dir must align in the plan (y-z in this case)

cfg["gpuid"] = 1
cfg["autopilot"] = 1
cfg["gscatter"] = 100

myprop = [[0.02, 0.1, 0.9, 1.37], [0.02, 10, 0.9, 1.37 * 5]]

cfg["prop"] = [[0, 0, 1, 1]] + myprop
cfg["tstart"] = 0
cfg["tend"] = 1e-8
cfg["tstep"] = 1e-8
res = pmcx.mcxlab(cfg)

cw = np.squeeze(np.sum(res["flux"], axis=3))
plt.figure()
plt.imshow(np.log10(np.abs(cw)))
plt.colorbar()
plt.show()
