# Copyright (c) 2022-2023 Matin Raayai Ardakani <raayaiardakani.m at northeastern.edu>. All rights reserved.
# Copyright (c) 2022-2025 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
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

"""PMCX - Python bindings of Monte Carlo eXtreme (MCX), a GPU (CUDA) photon transport simulator.

Simulator (compiled module ``_pmcx``)

* ``run(cfg)`` or ``run(**cfg)``: run a simulation; see ``help(pmcx.run)`` for the
  ``cfg`` keys and the outputs
* ``gpuinfo()``: list the GPUs
* ``version()``: the MCX version string

Utilities

* :func:`mcxlab`: run a simulation and post-process the outputs as mcxlab does
* :func:`mcxcreate`, ``bench``: built-in benchmark configurations
* :func:`detweight`, :func:`meanpath`, :func:`meanscat`, :func:`dettpsf`,
  :func:`dettime`, :func:`detphoton`, :func:`getdistance`: process detected photons
* :func:`rfreplay`, :func:`rfmusreplay`: frequency-domain Jacobians by photon replay
* :func:`cwdiffusion`, :func:`cwfluxdiffusion`, :func:`cwfluencediffusion`,
  :func:`tddiffusion`, :func:`cwdref`, :func:`dcsg1`: analytical diffusion solutions
* :func:`svmc`: build split-voxel (SVMC) volumes
* :func:`loadmc2`, :func:`loadmch`, :func:`loadnii`, :func:`loadfile`,
  :func:`mcx2json`, :func:`json2mcx`: file I/O
* :func:`preview`, :func:`plotshapes`, :func:`plotphotons`, :func:`plotvol`: plotting
  (need ``iso2mesh``)

The utilities are pure Python; on platforms without CUDA (e.g. macOS), the package
installs without ``_pmcx`` and they are used by pmcxcl.

Examples
--------
>>> import numpy as np
>>> import pmcx
>>> pmcx.gpuinfo()
>>> res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype="uint8"),
...                tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30, 30, 0],
...                srcdir=[0, 0, 1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
>>> res["flux"].shape
(60, 60, 60, 1)
"""

try:
    from _pmcx import gpuinfo, run, version
except ImportError:  # pragma: no cover
    # the noarch wheel (e.g. on macOS, which has no CUDA) ships only the python utilities,
    # which pmcxcl also borrows; report the missing simulator only when it is called

    def _nobinary(*args, **kwargs):
        raise ImportError(
            "the pmcx binary extension (_pmcx) is not available on this platform; this "
            "installation only provides the utility functions - use pmcxcl to run simulations"
        )

    gpuinfo = run = version = _nobinary

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
    cwdiffusion,
    cwfluxdiffusion,
    cwfluencediffusion,
    dcsg1,
    mcxcreate,
    rfreplay,
    rfmusreplay,
    svmc,
)

from .io import (
    loadmc2,
    loadmch,
    loadfile,
    mcx2json,
    json2mcx,
    loadnii,
)

from .plot import (
    preview,
    plotshapes,
    plotphotons,
    plotvol,
)

from .bench import bench

__version__ = "0.7.3"

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
    "cwdiffusion",
    "cwfluxdiffusion",
    "cwfluencediffusion",
    "dcsg1",
    "mcxcreate",
    "rfreplay",
    "rfmusreplay",
    "svmc",
    "loadmc2",
    "loadmch",
    "loadfile",
    "mcx2json",
    "json2mcx",
    "loadnii",
    "preview",
    "plotshapes",
    "plotphotons",
    "plotvol",
)
