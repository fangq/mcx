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

"""Built-in benchmark configurations, the ``--bench`` inputs of the ``mcx`` command line.

``bench`` is a dict of ``cfg`` dicts that run with ``pmcx.run(bench[name])``:

* ``cube60``: a 60x60x60 mm homogeneous cube with a pencil beam and 4 detectors,
  absorbing boundary
* ``cube60b``: ``cube60`` with reflecting boundaries (``isreflect=1``)
* ``cube60planar``: ``cube60b`` with a 40x40 mm planar source

For more benchmarks, see :func:`pmcx.mcxcreate`.
"""

import numpy as np
import copy

bench = {}
bench["cube60"] = {
    "nphoton": 1000000,
    "vol": np.ones([60, 60, 60], dtype="uint8"),
    "tstart": 0,
    "tend": 5e-9,
    "tstep": 5e-9,
    "srcpos": [29, 29, 0],
    "srcdir": [0, 0, 1],
    "prop": [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37], [0.002, 5, 0.9, 1]],
    "isreflect": 0,
    "seed": 1648335518,
    "session": "cube60",
    "detpos": [[29, 19, 0, 1], [29, 39, 0, 1], [19, 29, 0, 1], [39, 29, 0, 1]],
    "issrcfrom0": 1,
}

bench["cube60b"] = copy.deepcopy(bench["cube60"])
bench["cube60b"]["isreflect"] = 1

bench["cube60planar"] = copy.deepcopy(bench["cube60b"])
bench["cube60planar"]["srctype"] = "planar"
bench["cube60planar"]["srcpos"] = [10, 10, -10]
bench["cube60planar"]["srcparam1"] = [40, 0, 0, 0]
bench["cube60planar"]["srcparam2"] = [0, 40, 0, 0]
