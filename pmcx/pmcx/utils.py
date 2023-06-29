# Copyright (c) 2023 Kuznetsov Ilya
# Copyright (c) 2023 Qianqian Fang (q.fang <at> neu.edu)
# Copyright (c) 2023 Shijie Yan (yan.shiji <at> northeastern.edu)
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


"""Utilities for processing MCX input and output data"""


import numpy as np


def detweight(detp, prop):
    """
    Recalculate the detected photon weight using partial path data and
    optical properties (for perturbation Monte Carlo or detector readings)

    author: Kuznetsov Ilya
    Python code was adapted from mcxdetweight.m MATLAB function written by Qianqian Fang (q.fang <at> neu.edu)

    input:
        detp: the 2nd output from mcxlab. detp must be a dict
        prop: optical property list, as defined in the cfg['prop'] field of mcxlab's input
        unitinmm: voxel edge-length in mm, should use cfg['unitinmm'] used to generate detp;
                  if ignored, assume to be 1 (mm)

    output:
        detw: re-caculated detected photon weight based on the partial path data and optical property table

    License: GPLv3, see https://mcx.space/ for details
    """

    if "prop" in detp:
        prop = detp["prop"]

    medianum = prop.shape[0]
    if medianum <= 1:
        raise ValueError("empty property list")

    if unitinmm is None:
        if "unitinmm" in detp:
            unitinmm = detp["unitinmm"]
        else:
            unitinmm = 1

    if isinstance(detp, dict):
        if "w0" not in detp:
            detw = np.ones(detp["ppath"].shape[0])
        else:
            detw = detp["w0"]

        for i in range(medianum - 1):
            detw *= np.exp(-prop[i + 1, 0] * detp["ppath"][:, i] * unitinmm)
    else:
        raise ValueError('the first input must be a dict with a key named "ppath"')

    return detw


def cwdref(detp, cfg):
    """
    Compute CW diffuse reflectance from MC detected photon profiles.

    author: Kuznetsov Ilya
    Python code was adapted from mcxcwdref.m MATLAB function written by Shijie Yan (yan.shiji <at> northeastern.edu)

    input:
        detp: profiles of detected photons
        cfg:  a dictionary. Each element of cfg defines
              the parameters associated with a MC simulation.

    output:
        dref: CW diffuse reflectance at detectors

    this file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see https://mcx.space for details
    see Yao2018
    """

    unitinmm = 1
    if "unitinmm" in cfg:
        unitinmm = cfg["unitinmm"]

    det_weight = detweight(detp, cfg["prop"])
    detnum = len(np.unique(detp["detid"]))
    detweightsum = np.zeros(detnum)

    for i in range(len(detp["detid"])):
        index = int(detp["detid"][i]) - 1
        detweightsum[index] += det_weight[i]

    area = np.pi * (cfg["detpos"][:, 3] * unitinmm) ** 2
    dref = detweightsum / area / cfg["nphoton"]

    return dref
