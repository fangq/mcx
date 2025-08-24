# Copyright (c) 2022-2023 Matin Raayai Ardakani <raayaiardakani.m at northeastern.edu>. All rights reserved.
# Copyright (c) 2022-2023 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""PMCX - Python bindings for Monte Carlo eXtreme photon transport simulator

Example usage:

# To list available GPUs
import pmcx
pmcx.gpuinfo()

# To run a simulation
res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype='uint8'),
               tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30,30,0],
               srcdir=[0,0,1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
"""

try:
    from _pmcx import gpuinfo, run, version
except ImportError:  # pragma: no cover
    print("the pmcx binary extension (_pmcx) is not compiled! please compile first")

from .utils import (
    detweight,
    cwdref,
    meanpath,
    meanscat,
    dettpsf,
    dettime,
    tddiffusion,
    getdistance,
    detphoton,
    mcxlab,
)

# from .files import loadmc2, loadmch, load, save
from .bench import bench

__version__ = "0.4.3"

__all__ = (
    "gpuinfo",
    "run",
    "version",
    "bench",
    "detweight",
    "cwdref",
    "meanpath",
    "meanscat",
    "dettpsf",
    "dettime",
    "tddiffusion",
    "getdistance",
    "detphoton",
    "mcxlab",
)
