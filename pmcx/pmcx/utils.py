# Copyright (c) 2023-2025 Qianqian Fang (q.fang <at> neu.edu)
# Copyright (c) 2023 Fan-Yu (Ivy) Yen (yen.f at northeastern.edu)
# Copyright (c) 2023 Kuznetsov Ilya (for porting cwdref)
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

__all__ = (
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
)

##====================================================================================
## dependent libraries
##====================================================================================

import re
import sys
import os
import copy
import numpy as np

from .io import loadmch
from .plot import preview

##====================================================================================
## implementation
##====================================================================================


def cwdref(detp, cfg):
    """
    Compute CW diffuse reflectance from MC detected photon profiles.

    (Python code was adapted from mcxcwdref.m MATLAB function, ported by Kuznetsov Ilya)

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


# ---------------------------------------------
def meanpath(detp, prop=None):
    """
    Calculate the average pathlengths for each tissue type for a given source-detector pair

    (Python code was adapted from mcxmeanpath.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: the 2nd output from mcxlab. detp can be either a struct or an array (detp.data)
        prop: optical property list, as defined in the cfg.prop field of mcxlab's input

    output:
        avgpath: the average pathlength for each tissue type
    """
    if "unitinmm" in detp:
        unitinmm = detp["unitinmm"]
    else:
        unitinmm = 1

    if prop is None:
        if "prop" in detp:
            prop = detp["prop"]
        else:
            raise ValueError('Must provide input "prop"')

    detw = detweight(detp, prop)
    avgpath = np.sum(
        detp["ppath"] * unitinmm * np.tile(detw, (detp["ppath"].shape[1], 1)).T, 0
    ) / np.sum(detw)

    return avgpath


def detweight(detp, prop=None, unitinmm=None):
    """
    Recalculate the detected photon weight using partial path data and
    optical properties (for perturbation Monte Carlo or detector readings)

    (Python code was adapted from mcxdetweight.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: the 2nd output from mcxlab. detp must be a dict.
        prop: optical property list, as defined in the cfg.prop field of mcxlab's input.
        unitinmm: voxel edge-length in mm, should use cfg.unitinmm used to generate detp;
              if ignored, assume to be 1 (mm)

    output:
        detw: re-caculated detected photon weight based on the partial path data and optical property table
    """
    if prop is None:
        if "prop" in detp:
            prop = detp["prop"]
        else:
            raise ValueError('Must provide input "prop"')

    medianum = np.array(prop).shape[0]
    if medianum <= 1:
        raise ValueError("Empty property list")

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
            detw = detw * np.exp(-prop[i + 1, 0] * detp["ppath"][:, i] * unitinmm)
    else:
        raise ValueError('The first input must be a dict with a key named "ppath"')

    return detw


def meanscat(detp, prop):
    """
    Calculate the average scattering event counts for each tissue type for a given source-detector pair

    (Python code was adapted from mcxmeanscat.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: the 2nd output from mcxlab. detp can be either a struct or an array (detp.data)
        prop: optical property list, as defined in the cfg.prop field of mcxlab's input

    output:
        avgnscat: the average scattering event count for each tissue type

    Python code was ported from mcxmeanpath.m MATLAB function by Fan-Yu Yen
    """
    detw = detweight(detp, prop)
    avgnscat = np.sum(
        np.array(detp["nscat"], dtype=float) * detw[:, np.newaxis]
    ) / np.sum(detw)

    return avgnscat


def dettpsf(detp, detnum, prop, time):
    """
    Calculate the temporal point spread function curve of a specified detector
    given the partial path data, optical properties, and distribution of time bins

    (Python code was adapted from mcxdettpsf.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: the 2nd output from mcxlab. detp must be a struct with detid and ppath subfields
        detnum: specified detector number
        prop: optical property list, as defined in the cfg.prop field of mcxlab's input
        time: distribution of time bins, a 1*3 vector [tstart tend tstep]
              could be defined different from in the cfg of mcxlab's input
    output:
        tpsf: caculated temporal point spread function curve of the specified detector
    """
    detp = detp.copy()

    # select the photon data of the specified detector
    if "w0" in detp:
        detp["w0"] = detp["w0"][detp["detid"] == detnum]
    detp["ppath"] = detp["ppath"][detp["detid"] == detnum, :]
    detp["detid"] = detp["detid"][detp["detid"] == detnum]

    # calculate the detected photon weight and arrival time
    replayweight = detweight(detp, prop)
    replaytime = dettime(detp, prop)

    # define temporal point spread function vector
    nTG = int(np.round((time[1] - time[0]) / time[2]))
    tpsf = np.zeros((nTG, 1))

    # calculate the time bin, make sure not to exceed the boundary
    ntg = np.ceil((replaytime - time[0]) / time[2])
    ntg[ntg < 1] = 1
    ntg[ntg > nTG] = nTG

    ntg = ntg.astype(int)
    ntg = ntg[0]

    # add each photon weight to corresponding time bin
    for i in range(len(replayweight)):
        tpsf[ntg[i] - 1] = tpsf[ntg[i] - 1] + replayweight[i]

    return tpsf


def dettime(detp, prop=None, unitinmm=None):
    """
    Recalculate the detected photon time using partial path data and
    optical properties (for perturbation Monte Carlo or detector readings)

    (Python code was adapted from mcxdettime.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: the 2nd output from mcxlab. detp must be a struct
        prop: optical property list, as defined in the cfg.prop field of mcxlab's input
        unitinmm: voxel edge-length in mm, should use cfg.unitinmm used to generate detp;
              if ignored, assume to be 1 (mm)

    output:
        dett: re-caculated detected photon time based on the partial path data and optical property table
    """
    R_C0 = 3.335640951981520e-12  # inverse of light speed in vacuum
    if unitinmm is None:
        if "unitinmm" in detp:
            unitinmm = detp["unitinmm"]
        else:
            unitinmm = 1

    if prop is None:
        if "prop" in detp:
            prop = detp["prop"]
        else:
            raise ValueError('must provide input "prop"')

    medianum = prop.shape[0]
    if medianum <= 1:
        raise ValueError("empty property list")

    if isinstance(detp, dict):
        dett = np.zeros((1, detp["ppath"].shape[0]))
        for i in range(medianum - 1):
            dett = dett + prop[i + 1, 3] * detp["ppath"][:, i] * R_C0 * unitinmm
    else:
        raise ValueError(
            'the first input must be a struct with a subfield named "ppath"'
        )
    return dett


def tddiffusion(mua, musp, v, Reff, srcpos, detpos, t):
    """
    Semi-infinite medium analytical solution to diffusion model

    (Python code was adapted from tddiffusion.m MATLAB function, ported by Fan-Yu Yen)

    input:
        mua:   the absorption coefficients in 1/mm
        musp:  the reduced scattering coefficients in 1/mm
        v:     the speed of light
        Reff:  the effective reflection coeff.
        srcpos:array for the source positions (x,y,z)
        detpos:array for the detector positions (x,y,z)
        t:     a list of time in s at which to evaluate the
             analytical diffusion solution

    output:
        Phi:  the output fluence for all time points
    """
    srcpos = np.atleast_2d(srcpos)
    detpos = np.atleast_2d(detpos)

    D = 1 / (3 * (mua + musp))
    zb = (1 + Reff) / (1 - Reff) * 2 * D

    z0 = 1 / (musp + mua)
    r = getdistance(
        np.concatenate((srcpos[:, 0:2].flatten(), srcpos[:, 2].flatten() + z0)), detpos
    )
    r2 = getdistance(
        np.concatenate(
            (srcpos[:, 0:2].flatten(), srcpos[:, 2].flatten() - z0 - 2 * zb)
        ),
        detpos,
    )

    s = 4 * D * v * t

    # unit of phi:  1/(mm^2*s)
    Phi = (
        v
        / ((s * np.pi) ** (3 / 2))
        * np.exp(-mua * v * t)
        * (np.exp(-(r**2) / s) - np.exp(-(r2**2) / s))
    )

    return Phi


def getdistance(srcpos, detpos):
    """
    Compute the source/detector separation from the positions

    (Python code was adapted from getdistance.m MATLAB function, ported by Fan-Yu Yen)

    input:
        srcpos:array for the source positions (x,y,z)
        detpos:array for the detector positions (x,y,z)

    output:
        separation:  the distance matrix between all combinations
              of sources and detectors. separation has the number
              of source rows, and number of detector of columns.
    """
    srcpos = np.atleast_2d(srcpos)
    detpos = np.atleast_2d(detpos)

    srcnum = len(srcpos[:, 0])
    detnum = len(detpos[:, 0])

    separation = np.zeros((detnum, srcnum))

    for s in range(srcnum):
        for r in range(detnum):
            separation[r, s] = np.linalg.norm(srcpos[s, :] - detpos[r, :])
    return separation


def detphoton(detp, medianum, savedetflag, issaveref=None, srcnum=None):
    """
    Separating combined detected photon data into easy-to-read structure based on
    user-specified detected photon output format ("savedetflag")

    (Python code was adapted from mcxdetphoton.m MATLAB function, ported by Fan-Yu Yen)

    input:
        detp: a 2-D array defining the combined detected photon data, usually
              detp.data, where detp is the 2nd output from mcxlab
        medianum: the total number of non-zero tissue types (row number of cfg.prop minus 1)
        savedetflag: the cfg.savedetflag string, containing letters 'dspmxvwi' denoting different
              output data fields, please see mcxlab's help
        issaveref: the cfg.issaveref flag, 1 for saving diffuse reflectance, 0 not to save
        srcnum: the cfg.srcnum flag, denoting the number of source patterns in the photon-sharing mode

    output:
        newdetp: re-organized detected photon data as a dict; the mapping of the fields are
                 newdetp['detid']: the ID(>0) of the detector that captures the photon (1)
                 newdetp['nscat']: cummulative scattering event counts in each medium (#medium)
                 newdetp['ppath']: cummulative path lengths in each medium, i.e. partial pathlength (#medium)
                      one need to multiply cfg.unitinmm with ppath to convert it to mm.
                 newdetp['mom']: cummulative cos_theta for momentum transfer in each medium (#medium)
                 newdetp['p'] or ['v']: exit position and direction, when cfg.issaveexit=1 (3)
                 newdetp['w0']: photon initial weight at launch time (3)
                 newdetp['s']: exit Stokes parameters for polarized photon (4)
                 newdetp['srcid']: the ID of the source when multiple sources are defined (1)
    """
    newdetp = {}
    c0 = 0
    length = 0

    if re.search("[dD]", savedetflag):
        if issaveref is not None and issaveref > 1:
            newdetp["w0"] = detp[0, :].transpose()
        else:
            newdetp["detid"] = detp[0, :].astype(int).transpose()
            if np.any(newdetp["detid"] > 65535):
                newdetp["srcid"] = np.right_shift(newdetp["detid"], 16, dtype=np.int32)
                newdetp["detid"] = np.bitwise_and(
                    newdetp["detid"], 0xFFFF, dtype=np.int32
                )
        c0 = 1

    length = medianum
    if re.search("[sS]", savedetflag):
        newdetp["nscat"] = detp[c0 : c0 + length, :].astype(int).transpose()
        c0 = c0 + length

    if re.search("[pP]", savedetflag):
        newdetp["ppath"] = detp[c0 : c0 + length, :].transpose()
        c0 = c0 + length

    if re.search("[mM]", savedetflag):
        newdetp["mom"] = detp[c0 : c0 + length, :].transpose()
        c0 = c0 + length

    length = 3
    if re.search("[xX]", savedetflag):
        newdetp["p"] = detp[c0 : c0 + length, :].transpose()
        c0 = c0 + length

    if re.search("[vV]", savedetflag):
        newdetp["v"] = detp[c0 : c0 + length, :].transpose()
        c0 = c0 + length

    if re.search("[wW]", savedetflag):
        length = 1
        newdetp["w0"] = detp[c0 : c0 + length, :].transpose()
        if srcnum is not None and srcnum > 1:
            newdetp["w0"] = newdetp["w0"].view("uint32")
        c0 = c0 + length

    if re.search("[iI]", savedetflag):
        length = 4
        newdetp["s"] = detp[c0 : c0 + length, :].transpose()
        c0 = c0 + length

    return newdetp


def mcxlab(*args):
    """
    Python wrapper of mcxlab - please see the help information of mcxlab.m for details

    (Python code was adapted from mcxlab.m MATLAB function, ported by Fan-Yu Yen)

    Format:
       res = mcxlab(cfg);
          or
       res = mcxlab(cfg, option);

    Input:
       cfg: a dictionary defining the parameters associated with a simulation.
            if cfg='gpuinfo': return the supported GPUs and their parameters,
            if cfg='version': return the version of MCXLAB as a string,
       option: (optional), options is a string, specifying additional options
            option='opencl':  force using mcxcl.mex* instead of mcx.mex* on NVIDIA/AMD/Intel hardware
            option='cuda':    force using mcx.mex* instead of mcxcl.mex* on NVIDIA GPUs

       if one defines USE_MCXCL=1 in as Python global variable, all following
       pmcx.mcxlab calls will use _pmcxcl (OpenCL version of mcx); by setting option='cuda',
       one can force pmcx.mcxlab to use _pmcx (CUDA version). Similarly, if
       USE_MCXCL=0, all pmcx.mcxlab calls will use _pmcx by default, unless
       one sets option='opencl'.

    Output:
         res: a dictionary containing the following subfields
             res['flux'] is a 4D array with
                    dimensions specified by [size(vol) total-time-gates].
                    The content of the array is the normalized fluence at
                    each voxel of each time-gate.

                    when cfg.debuglevel contains 'T', fluence(i).data stores trajectory
                    output, see below
             res['dref'] is a 4D array with the same dimension as fluence(i).data
                    if cfg.issaveref is set to 1, containing only non-zero values in the
                    layer of voxels immediately next to the non-zero voxels in cfg.vol,
                    storing the normalized total diffuse reflectance (summation of the weights
                    of all escaped photon to the background regardless of their direction);
                    it is an empty array [] when if cfg.issaveref is 0.
             res['stat'] is a structure storing additional information, including
                    runtime: total simulation run-time in millisecond
                    nphoton: total simulated photon number
                    energytot: total initial weight/energy of all launched photons
                    energyabs: total absorbed weight/energy of all photons
                    normalizer: normalization factor
                    unitinmm: same as cfg.unitinmm, voxel edge-length in mm

             res['detp']: res['detp'] is a directionary object including the following field
                   res['detp']['detid']: the ID(>0) of the detector that captures the photon
                   res['detp']['nscat']: cummulative scattering event counts in each medium
                   res['detp']['ppath']: cummulative path lengths in each medium (partial pathlength)
                        one need to multiply cfg.unitinmm with ppath to convert it to mm.
                   res['detp']['mom']: cummulative cos_theta for momentum transfer in each medium
                   res['detp']['p'] or ['v']: exit position and direction, when cfg.issaveexit=1
                   res['detp']['nscat']: photon initial weight at launch time
                   res['detp']['s']: exit Stokes parameters for polarized photon
                   res['detp']['prop']: optical properties, a copy of cfg.prop
                   res['detp']['data']: a concatenated and transposed array in the order of
                        [detid nscat ppath mom p v w0]'

                   if returned by pmcx.run, res['det'] is a 2D numpy array in a format as
                   res['detp']['data'] described above

             res['vol']: (optional) a numpy array storing a preprocessed volume.
                   Each volume is a 3D int32 array.
             res['seeds']: (optional), if give, mcxlab returns the seeds, in the form of
                   a byte array (uint8) for each detected photon. The column number
                   of seed equals that of res['detp'].
             res['traj']: if given, mcxlab returns the trajectory data for
                   each simulated photon. The output has 6 rows, the meanings are
                      res['traj']['id']:    1:    index of the photon packet
                      res['traj']['pos']: 2-4:    x/y/z/ of each trajectory position
                                            5:    current photon packet weight
                                            6:    reserved
               By default, mcxlab only records the first 1e7 positions along all
               simulated photons; change cfg['maxjumpdebug'] to define a different limit.
    """
    try:
        defaultocl = eval("USE_MCXCL", globals())
    except:
        defaultocl = 0

    useopencl = defaultocl

    if useopencl == 0:
        try:
            from _pmcx import gpuinfo, run, version
        except ImportError:
            print(
                "the pmcx binary extension (_pmcx) is not compiled! please compile first"
            )
    else:
        try:
            from _pmcxcl import gpuinfo, run, version
        except ImportError:
            raise ImportError(
                'To call OpenCL based MCX, one must first run "pip install pmcxcl" to install pmcxcl'
            )

    if len(args) == 1 and isinstance(args[0], str):
        if args[0] == "gpuinfo":
            varargout = gpuinfo()
            return varargout
        elif args[0] == "version":
            varargout = version()
            return varargout

    if len(args) == 2 and isinstance(args[1], str):
        if args[1] == "preview":
            varargout = preview(args[0])
            return varargout
        elif args[1] == "opencl":
            useopencl = 1

    if isinstance(args[0], dict):
        castlist = [
            "srcpattern",
            "srcpos",
            "detpos",
            "prop",
            "workload",
            "srcdir",
            "srciquv",
        ]
        for j in range(len(castlist)):
            if castlist[j] in args[0]:
                args[0][castlist[j]] = np.double(args[0][castlist[j]])

        if "vol" in args[0] and np.ndim(args[0]["vol"]) == 4:
            if (
                isinstance(args[0]["vol"], np.float32)
                or isinstance(args[0]["vol"], np.float64)
            ) and "unitinmm" in args[0]:
                args[0]["vol"] = args[0]["vol"] * args[0]["unitinmm"]

        if "tstart" not in args[0]:
            args[0]["tstart"] = 0

        if "tend" not in args[0]:
            raise ValueError(
                "you must define cfg.tend for the maximum time-of-flight of a photon in seconds"
            )

        if "tstep" not in args[0]:
            args[0]["tsetp"] = args[0]["tend"]

        if "srcpos" not in args[0]:
            raise ValueError(
                "you must define cfg.srcpos to define the x/y/z position of the source in voxel unit"
            )

        if "detphotons" in args[0] and isinstance(args[0]["detphotons"], dict):
            if "data" in args[0]["detphotons"]:
                args[0]["detphotons"] = args[0]["dephotons"]["data"]
            else:
                fulldetdata = ["detid", "nscat", "ppath", "mom", "p", "v", "w0"]
                detfields = [x in args[0]["detphotons"] for x in fulldetdata]
                detdata = []
                for j in range(len(detfields)):
                    if detfields[j]:
                        val = np.array(
                            args[0]["detphotons"][fulldetdata[j]], dtype=np.float32
                        )
                        detdata.append(
                            np.reshape(
                                val, np.shape(args[0]["detphotons"][fulldetdata[j]])
                            )
                        )
                args[0]["detphotons"] = np.vstack(detdata).T
                args[0]["savedetflag"] = "dspmxvw"
                args[0]["savedetflag"][detfields == 0] = []

    varargout = run(args[0])

    if len(args) == 0:
        return

    cfg = args[0]

    if not isinstance(cfg, str):
        if "srcnum" in cfg and cfg["srcnum"] > 1:
            dim = varargout["flux"].shape
            varargout["flux"] = varargout["flux"].reshape(
                [cfg["srcnum"], int(dim[0] / cfg["srcnum"]), *dim[1:]]
            )
            varargout["flux"] = np.transpose(
                varargout["flux"], [i for i in range(1, len(dim) + 1)] + [0]
            )
            if "dref" in varargout and varargout["dref"] is not None:
                varargout["dref"] = varargout["dref"].reshape(
                    [cfg["srcnum"], int(dim[0] / cfg["srcnum"]), *dim[1:]]
                )
                varargout["dref"] = np.transpose(
                    varargout["dref"], [i for i in range(1, len(dim) + 1)] + [0]
                )

    if "detp" in varargout:
        #         for i in range(len(varargout.keys())): #for i in range(varargout[1]):
        if ("savedetflag" not in cfg) or (
            ("savedetflag" in cfg) and (not cfg["savedetflag"])
        ):
            cfg["savedetflag"] = "DP"

        if "issaveexit" in cfg and cfg["issaveexit"]:
            cfg["savedetflag"] += "XV"

        if "ismomentum" in cfg and cfg["ismomentum"]:
            cfg["savedetflag"] += "M"

        if "polprop" in cfg and cfg["polprop"]:
            cfg["savedetflag"] += "PVWI"
        else:
            cfg["savedetflag"] = (
                cfg["savedetflag"].replace("I", "").replace("i", "")
            )  # cfg(i).savedetflag(regexp(cfg(i).savedetflag,'[Ii]'))=[];

        if np.ndim(cfg["vol"]) == 4 and np.shape(cfg["vol"])[0] != 8:
            cfg["savedetflag"] = ""
            if (
                isinstance(cfg["vol"], float) or isinstance(cfg["vol"], np.float64)
            ) and "unitinmm" in cfg:  # if((isa(cfg(i).vol,'single') || isa(cfg(i).vol,'double')) && isfield(cfg(i),'unitinmm'))
                cfg["vol"] = cfg["vol"] * cfg["unitinmm"]

        if "issaveexit" not in cfg or cfg["issaveexit"] != 2:
            medianum = np.shape(cfg["prop"])[0] - 1
            detp = varargout["detp"]  # detp = varargout[1]['data']
            #             if not detp:
            #                 continue

            if "polprop" in cfg and cfg["polprop"]:
                medianum = np.shape(cfg["polprop"])[0]

            flags = [cfg["savedetflag"]]

            if "issaveref" in cfg:
                flags.append(cfg["issaveref"])

            if "srcnum" in cfg:
                flags.append(cfg["srcnum"])

            newdetp = detphoton(
                detp, medianum, *flags
            )  # newdetp=mcxdetphoton(detp,medianum,flags{:});
            newdetp["prop"] = cfg["prop"]

            if "polprop" in cfg and cfg["polprop"] and "prop" in varargout:
                newdetp["prop"][1:] = np.transpose(varargout["prop"][:, 1:])

            if "unitinmm" in cfg:
                newdetp["unitinmm"] = cfg["unitinmm"]

            newdetp["data"] = detp
            newdetpstruct = newdetp
        else:
            newdetpstruct = varargout["detp"]

        if "newdetpstruct" in locals():
            varargout["detp"] = newdetpstruct  # varargout = newdetpstruct

    if "traj" in varargout:
        data = varargout["traj"]
        if len(data):
            traj = {}
            traj["pos"] = np.transpose(data[1:4, :])
            traj["id"] = np.frombuffer(data[0, :].tobytes(), dtype=np.uint32)
            traj["id"], idx = np.sort(traj["id"]), np.argsort(traj["id"])
            traj["pos"] = traj["pos"][idx, :]
            traj["data"] = np.vstack([np.single(traj["id"]), data[1:, idx]])
            varargout["traj"] = traj

    return varargout


def mcxcreate(benchname=None, **kwargs):
    """
    Format:
        list = mcxcreate()
        cfg = mcxcreate(benchname)
        cfg = mcxcreate(benchname, param1=value1, param2=value2, ...)

    Create MCX simulation from built-in benchmarks (similar to "mcx --bench")

    Author: Qianqian Fang <q.fang at neu.edu>
    Converted to Python

    Parameters:
    -----------
    benchname : str, optional
        A string to specify the name of the benchmark
    **kwargs : dict
        Parameter-value pairs to override default settings

    Returns:
    --------
    cfg : dict or list
        A dict defining the parameters associated with a simulation. If
        no input, this function returns a list of supported benchmarks.

    Examples:
    ---------
    list = mcxcreate()
    cfg = mcxcreate('cube60b')
    cfg = mcxcreate('cube60', srctype='isotropic', srcpos=[30, 30, 30])

    This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space

    License: GNU General Public License version 3, please read LICENSE.txt for details
    """

    # Initialize benchmark dictionary
    mcxbench = {}

    mcxbench["cube60"] = {
        "nphoton": int(1e6),
        "vol": np.ones((60, 60, 60), dtype=np.uint8),
        "srctype": "pencil",
        "srcpos": [29, 29, 0],
        "srcdir": [0, 0, 1],
        "prop": [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]],
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "isreflect": 0,
        "seed": 1648335518,
        "issrcfrom0": 1,
        "detpos": [[29, 19, 0, 1], [29, 39, 0, 1], [19, 29, 0, 1], [39, 29, 0, 1]],
    }

    mcxbench["cube60b"] = mcxbench["cube60"].copy()
    mcxbench["cube60b"]["isreflect"] = 1

    mcxbench["cube60planar"] = mcxbench["cube60b"].copy()
    mcxbench["cube60planar"]["srctype"] = "planar"
    mcxbench["cube60planar"]["srcpos"] = [10.0, 10.0, -10.0]
    mcxbench["cube60planar"]["srcparam1"] = [40.0, 0.0, 0.0, 0.0]
    mcxbench["cube60planar"]["srcparam2"] = [0.0, 40.0, 0.0, 0.0]

    mcxbench["skinvessel"] = {
        "nphoton": int(1e6),
        "vol": np.ones((200, 200, 200), dtype=np.uint8),
        "srctype": "disk",
        "srcpos": [100, 100, 20],
        "srcdir": [0, 0, 1],
        "srcparam1": [60, 0, 0, 0],
        "unitinmm": 0.005,
        "prop": [
            [0, 0, 1, 1],
            [3.564e-05, 1, 1, 1.37],
            [23.05426549, 9.398496241, 0.9, 1.37],
            [0.04584957865, 35.65405549, 0.9, 1.37],
            [1.657237447, 37.59398496, 0.9, 1.37],
        ],
        "tstart": 0,
        "tend": 5e-8,
        "tstep": 5e-8,
        "isreflect": 0,
        "seed": 1648335518,
        "issrcfrom0": 1,
    }
    mcxbench["skinvessel"]["shapes"] = (
        '{"Shapes":[{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]},'
        '{"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}]}'
    )

    mcxbench["sphshell"] = {
        "nphoton": int(1e6),
        "vol": np.ones((60, 60, 60), dtype=np.uint8),
        "srctype": "pencil",
        "srcpos": [30, 30.1, 0],
        "srcdir": [0, 0, 1],
        "prop": [
            [0, 0, 1, 1],
            [0.02, 7, 0.89, 1.37],
            [0.004, 0.009, 0.89, 1.37],
            [0.02, 9.0, 0.89, 1.37],
            [0.05, 0.0, 1.00, 1.37],
        ],
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "isreflect": 1,
        "seed": 1648335518,
        "issrcfrom0": 1,
    }
    mcxbench["sphshell"]["shapes"] = (
        '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},'
        '{"Sphere":{"Tag":2,"O":[30,30,30],"R":25}},'
        '{"Sphere":{"Tag":3,"O":[30,30,30],"R":23}},'
        '{"Sphere":{"Tag":4,"O":[30,30,30],"R":10}}]}'
    )
    mcxbench["sphshell"]["detpos"] = mcxbench["cube60"]["detpos"]

    mcxbench["spherebox"] = {
        "nphoton": int(1e6),
        "vol": np.ones((60, 60, 60), dtype=np.uint8),
        "srcpos": [29.5, 29.5, 0],
        "srcdir": [0, 0, 1],
        "prop": [[0, 0, 1, 1], [0.002, 1, 0.01, 1.37], [0.05, 5, 0.9, 1.37]],
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 1e-10,
        "isreflect": 0,
        "seed": 1648335518,
        "issrcfrom0": 1,
    }
    mcxbench["spherebox"]["shapes"] = (
        '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},'
        '{"Sphere":{"Tag":2,"O":[30,30,30],"R":10}}]}'
    )

    if benchname is None:
        cfg = list(mcxbench.keys())
        return cfg

    if benchname in mcxbench:
        cfg = mcxbench[benchname].copy()
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(key, str):
                    cfg[key.lower()] = value
                else:
                    raise ValueError("input must be in the form of keyword arguments")
    else:
        raise ValueError("benchmark name is not supported")

    return cfg


def dcsg1(
    detps,
    tau=None,
    disp_model="brownian",
    DV=1e-7,
    lambda_=785,
    format="float",
    **varargin,
):
    """
    tau, g1 = mcxdcsg1(detps, tau, disp_model, DV, lambda_, format)

    Compute simulated electric-field auto-correlation function using
    simulated photon pathlengths and scattering momentum transfer

    Author: Stefan Carp (carp <at> nmr.mgh.harvard.edu)
    Converted to Python

    Parameters:
    -----------
    detps : str or dict
        The file name of the output .mch file or the 2nd output from mcxlab
    tau : array_like, optional
        Correlation times at which to compute g1
        (default: 1e-7 to 1e-1 seconds, log equidistant)
    disp_model : str, optional
        Displacement model ('brownian', 'random_flow', <custom>)
        (default: 'brownian', see further explanation below)
    DV : float, optional
        Value of displacement variable using mm as unit of
        length and s as unit of time
        (default: 1e-7 mm^2/s, see further explanation below)
    lambda_ : float, optional
        Wavelength of light used in nm
        (default: 785)
    format : str, optional
        The format used to save the .mch file
        (default: 'float')
    **varargin : dict
        Additional parameters for custom displacement models

    Returns:
    --------
    tau : ndarray
        Correlation times at which g1 was computed provided for
        convenience (copied from input if set, otherwise
        outputs default)
    g1 : ndarray
        Field auto-correlation curves, one for each detector

    The displacement model indicates the formula used to compute the root
    mean square displacement of scattering particles during a given delay

    brownian:       RMS = 6 * DV * tau;
                    DV(displacement variable) = Db (brownian diffusion coeff)
    random_flow:    RMS = DV^2 * tau^2;
                    DV = V (first moment of velocity distribution)
    <custom>:       Any string other than 'brownian' or 'random_flow' will
                    be evaluated as is using Python eval, make sure it uses
                    'DV' as the flow related independent variable, tau is
                    indexed as tau[J]. Any additional parameters can be
                    sent via "varargin"

    This file is part of Mesh-Based Monte Carlo
    License: GPLv3, see http://mcx.space for details
    """

    # Set default values
    if tau is None:
        tau = np.logspace(-7, -1, 200)

    # Handle input data
    if isinstance(detps, str):
        from jdata import load as jload

        mch_data, mch_header = loadmch(detps, format)
        fpath, fname = os.path.split(detps)
        fname_no_ext = os.path.splitext(fname)[0]

        json_file = os.path.join(fpath, fname_no_ext + ".json")
        cfg = jload(json_file)

        # Extract optical properties
        media = cfg["Domain"]["Media"]
        if isinstance(media, list):
            prop = np.array([[m["mua"], m["mus"], m["g"], m["n"]] for m in media])
        else:
            prop = media

        mua = prop[1:, 0]  # Skip first row (background), take mua column
        n = prop[1:, 3]  # Skip first row (background), take n column
        medianum = len(mua)

        # Create detps structure
        detps = {}
        detps["detid"] = mch_data[:, 0]  # First column is detector ID
        detps["ppath"] = mch_data[:, 2 : 2 + medianum]  # Path lengths for each medium

        if mch_data.shape[1] >= 2 * medianum + 2:
            detps["mom"] = mch_data[
                :, medianum + 2 : 2 * medianum + 2
            ]  # Momentum transfer data
    else:
        # detps is already a dictionary/structure
        mua = detps["prop"][1:, 0]  # Skip first row (background)
        n = detps["prop"][1:, 3]  # Skip first row (background)

    if "mom" not in detps:
        raise ValueError(
            "No momentum transfer data are found, please rerun your simulation and set cfg.ismomentum=1."
        )

    # Set displacement model string
    if disp_model == "brownian":
        disp_str = "rmsdisp = 6 * DV * tau[J]"
    elif disp_model == "random_flow":
        disp_str = "rmsdisp = DV**2 * tau[J]**2"
    else:
        disp_str = f"rmsdisp = {disp_model}"

    # Calculate wave vector
    k0 = 2 * np.pi * n / (lambda_ * 1e-6)

    # Get unique detector list
    detlist = np.sort(np.unique(detps["detid"]))
    g1 = np.zeros((int(np.max(detlist)), len(tau)))

    # Process each detector
    for detid_idx, detid in enumerate(detlist):
        I = int(detid)
        idx = np.where(detps["detid"] == I)[0]
        print(f"Processing detector {I}: {len(idx)} photons")

        # Get data for this detector
        det_ppath = detps["ppath"][idx, :]
        det_mom = detps["mom"][idx, :]

        # Process each correlation time
        for J in range(len(tau)):
            # Evaluate displacement string to get rmsdisp
            # Create local namespace for eval
            local_vars = {
                "DV": DV,
                "tau": tau,
                "J": J,
                "np": np,
                **varargin,  # Include any additional variables
            }

            exec(disp_str, {"__builtins__": {}}, local_vars)
            rmsdisp = local_vars["rmsdisp"]

            # Calculate g1 for this detector and correlation time
            # Sum over all photons for this detector
            phase_factor = np.exp(
                -(k0**2 * rmsdisp / 3) * np.sum(det_mom, axis=1)
                - np.dot(mua, det_ppath.T)
            )
            g1[I - 1, J] = np.sum(phase_factor)

        # Normalize g1
        g1_norm = np.sum(np.exp(-np.dot(mua, det_ppath.T)))
        g1[I - 1, :] = g1[I - 1, :] / g1_norm

    return tau, g1


def cwdiffusion(mua, musp, Reff, srcpos, detpos):
    """
    Phi, r = cwdiffusion(mua, musp, Reff, srcpos, detpos)

    Semi-infinite medium analytical solution to diffusion model

    Author: Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Parameters:
    -----------
    mua : float
        The absorption coefficients in 1/mm
    musp : float
        The reduced scattering coefficients in 1/mm
    Reff : float
        The effective reflection coefficient
    srcpos : array_like
        Array for the source positions (x, y, z)
    detpos : array_like
        Array for the detector positions (x, y, z)

    Returns:
    --------
    Phi : ndarray
        The output fluence for all source/detector pairs
    r : ndarray (optional)
        Source detector separations

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see https://mcx.space for details
    See Boas2002, Haskell1994
    """

    # Convert inputs to numpy arrays
    srcpos = np.array(srcpos)
    detpos = np.array(detpos)

    # Calculate diffusion coefficient
    D = 1 / (3 * (mua + musp))

    # Calculate extrapolation distance
    zb = (1 + Reff) / (1 - Reff) * 2 * D

    # Calculate source depth
    z0 = 1 / (musp + mua)

    # Create modified source positions for real and image sources
    # Real source: add z0 to z coordinate
    src_real = srcpos.copy()
    if src_real.ndim == 1:
        src_real = src_real.reshape(1, -1)
    src_real[:, 2] = src_real[:, 2] + z0

    # Image source: subtract z0 and 2*zb from z coordinate
    src_image = srcpos.copy()
    if src_image.ndim == 1:
        src_image = src_image.reshape(1, -1)
    src_image[:, 2] = src_image[:, 2] - z0 - 2 * zb

    # Calculate distances
    r = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    # Calculate effective attenuation coefficient
    b = np.sqrt(3 * mua * musp)

    # Calculate fluence using diffusion equation solution
    # Unit of phi: 1/(mm^2)
    Phi = 1 / (4 * np.pi * D) * (np.exp(-b * r) / r - np.exp(-b * r2) / r2)

    return Phi, r


def cwfluxdiffusion(mua, musp, Reff, srcpos, detpos):
    """
    flux = cwfluxdiffusion(mua, musp, Reff, srcpos, detpos)

    Compute surface flux for a semi-infinite medium

    Author: Shijie Yan (yan.shiji <at> northeastern.edu)
    Converted to Python

    Parameters:
    -----------
    mua : float
        The absorption coefficients in 1/mm
    musp : float
        The reduced scattering coefficients in 1/mm
    Reff : float
        The effective reflection coefficient
    srcpos : array_like
        Array for the source positions (x, y, z)
    detpos : array_like
        Array for the detector positions (x, y, z)

    Returns:
    --------
    flux : ndarray
        The diffuse reflectance for all source/detector pairs

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see https://mcx.space for details
    See Kienle1997
    """

    # Convert inputs to numpy arrays
    srcpos = np.array(srcpos)
    detpos = np.array(detpos)

    # Ensure arrays are 2D
    if srcpos.ndim == 1:
        srcpos = srcpos.reshape(1, -1)
    if detpos.ndim == 1:
        detpos = detpos.reshape(1, -1)

    # Calculate diffusion coefficient
    D = 1 / (3 * (mua + musp))

    # Calculate source depth
    z0 = 1 / (mua + musp)

    # Calculate extrapolation distance
    zb = (1 + Reff) / (1 - Reff) * 2 * D

    # Calculate effective attenuation coefficient
    mueff = np.sqrt(3 * mua * (mua + musp))

    # Create modified source positions for real and image sources
    # Real source: z-coordinate shifted by +z0
    src_real = srcpos.copy()
    src_real[:, 2] = src_real[:, 2] + z0

    # Image source: z-coordinate shifted by +z0+2*zb
    src_image = srcpos.copy()
    src_image[:, 2] = src_image[:, 2] + z0 + 2 * zb

    # Calculate distances using getdistance function
    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    # Calculate flux using Eq. 6 of Kienle1997
    flux = (
        1
        / (4 * np.pi)
        * (
            z0 * (mueff + 1 / r1) * np.exp(-mueff * r1) / r1**2
            + (z0 + 2 * zb) * (mueff + 1 / r2) * np.exp(-mueff * r2) / r2**2
        )
    )

    return flux


def cwfluencediffusion(*args, **kwargs):
    return cwdiffusion(*args, **kwargs)


def rfreplay(cfg, f_mod, detp, seeds, detnums):
    """
    rfjac_lnA, rfjac_phase = mcxrfreplay(cfg, f_mod, detp, seeds, detnums)

    Compute the frequency domain (FD) log-amplitude and phase shift Jacobians
    with respect to voxel-wise absorption coefficients using the radio
    frequency (RF) replay algorithm.

    Authors: Pauliina Hirvi (pauliina.hirvi <at> aalto.fi)
             Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Ref.: Hirvi et al. (2023). Effects of atlas-based anatomy on modelled
    light transport in the neonatal head. Phys. Med. Biol.
    https://doi.org/10.1088/1361-6560/acd48c

    Parameters:
    -----------
    cfg : dict
        Dict used in the main forward simulation
    f_mod : float
        RF modulation frequency in Hz
    detp : dict
        The 2nd output from mcxlab, must be a dict
    seeds : dict
        The 4th output from mcxlab
    detnums : array_like
        Array with the indices of the detectors to replay and obtain Jacobians

    Returns:
    --------
    rfjac_lnA : ndarray
        A 4D array with dimensions specified by [size(vol) num-of-detectors];
        each 3D array contains the Jacobians for log-amplitude measurements
    rfjac_phase : ndarray
        A 4D array with dimensions specified by [size(vol) num-of-detectors];
        each 3D array contains the Jacobians for phase shift measurements

    License: GPLv3, see http://mcx.space/ for details
    """

    if cfg is None or f_mod is None or detp is None or seeds is None or detnums is None:
        raise ValueError("you must provide all 5 required input parameters")

    if "unitinmm" not in cfg:
        cfg["unitinmm"] = 1

    # Convert detnums to numpy array for consistent indexing
    detnums = np.array(detnums)

    # Initialize the 4D arrays for collecting the Jacobians. The 4th dimension
    # corresponds to the detector index.
    vol_shape = np.array(cfg["vol"]).shape
    rfjac_lnA = np.zeros(list(vol_shape) + [len(detnums)])
    rfjac_phase = np.zeros(list(vol_shape) + [len(detnums)])

    # Collect Jacobians one detector index at a time.
    for idx, d in enumerate(detnums):
        # MCXLAB REPLAY SETTINGS
        cfg_jac = copy.deepcopy(cfg)
        cfg_jac["seed"] = seeds["data"]
        cfg_jac["detphotons"] = detp["data"]
        cfg_jac["replaydet"] = d
        cfg_jac["outputtype"] = "rf"
        cfg_jac["omega"] = 2 * np.pi * f_mod  # RF modulation frequency
        cfg_jac["isnormalized"] = 0  # Important!
        cfg_jac["issave2pt"] = 1

        # REPLAY SIMULATION
        rfjac_d, detp_d, vol_d, seeds_d = mcxlab(cfg_jac)

        # Array with detected photon weights
        detw = detweight(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"])

        # Array with photon time-of-flights
        dett = dettime(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"])

        # FD MEASUREMENT ESTIMATES
        X = np.dot(detw, np.cos((2 * f_mod) * dett * np.pi))
        Y = np.dot(detw, np.sin((2 * f_mod) * dett * np.pi))
        A = np.sqrt(X**2 + Y**2)  # amplitude [a.u.]

        # Phase shift in [0, 2*pi] [rad]
        phase = np.arctan2(Y, X) + (np.arctan2(Y, X) < 0).astype(float) * 2 * np.pi

        if A == 0:
            print(f"MCX WARNING: No detected photons for detector {d}.")
            continue

        # FD JACOBIANS
        # Compute the Jacobians with the rf replay feature.
        rfjac_d_data = rfjac_d["data"]
        rfjac_d_sum = np.sum(rfjac_d_data, axis=3)  # sum over time instances (axis 3)

        if cfg_jac["isnormalized"] == 0:
            rfjac_d_sum = cfg_jac["unitinmm"] * rfjac_d_sum  # correct units to [mm]

        # Jacobians for X and Y wrt mua:
        rfjac_X = rfjac_d_sum[:, :, :, 0]  # First component
        rfjac_Y = rfjac_d_sum[:, :, :, 1]  # Second component

        # Jacobians for log-amplitude and phase shift wrt mua:
        rfjac_lnA[:, :, :, idx] = (1 / (A**2)) * (X * rfjac_X + Y * rfjac_Y)
        rfjac_phase[:, :, :, idx] = (1 / (A**2)) * (X * rfjac_Y - Y * rfjac_X)

    return rfjac_lnA, rfjac_phase


def rfmusreplay(cfg, f_mod, detp, seeds, detnums):
    """
    rfmusjac_lnA, rfmusjac_phase = mcxrfmusreplay(cfg, f_mod, detp, seeds, detnums)

    Compute the frequency domain (FD) log-amplitude and phase shift Jacobians
    with respect to voxel-wise scattering coefficients using the radio
    frequency mode (RF) replay algorithm.

    Authors: Pauliina Hirvi (pauliina.hirvi <at> aalto.fi)
             Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Ref.: Hirvi et al. (2025): https://www.overleaf.com/read/qgtqcdyvqfrw#485e8c
    Hirvi et al. (2023): https://doi.org/10.1088/1361-6560/acd48c

    Parameters:
    -----------
    cfg : dict
        Dict used in the main forward simulation
    f_mod : float
        RF modulation frequency in Hz
    detp : dict
        The 2nd output from mcxlab, must be a dict
    seeds : dict
        The 4th output from mcxlab
    detnums : array_like
        Array with the indices of the detectors to replay and obtain Jacobians

    Returns:
    --------
    rfmusjac_lnA : ndarray
        A 4D array with dimensions specified by [size(vol) num-of-detectors];
        each 3D array contains the Jacobians for log-amplitude measurements
    rfmusjac_phase : ndarray
        A 4D array with dimensions specified by [size(vol) num-of-detectors];
        each 3D array contains the Jacobians for phase shift measurements

    License: GPLv3, see http://mcx.space/ for details
    """

    # Control.
    if cfg is None or f_mod is None or detp is None or seeds is None or detnums is None:
        raise ValueError("you must provide all 5 required input parameters")

    if "unitinmm" not in cfg:
        cfg["unitinmm"] = 1

    # Convert detnums to numpy array for consistent indexing
    detnums = np.array(detnums)

    # Initialize the 4D arrays for collecting the Jacobians. The 4th dimension
    # corresponds to the detector index.
    vol_shape = np.array(cfg["vol"]).shape
    rfmusjac_lnA = np.zeros(list(vol_shape) + [len(detnums)])
    rfmusjac_phase = np.zeros(list(vol_shape) + [len(detnums)])

    # Return if no photons detected.
    if not detp or not seeds or len(detp) == 0 or len(seeds) == 0:
        print("MCX WARNING: No detected photons for replay.")
        return rfmusjac_lnA, rfmusjac_phase

    # Form matrix with mus for each nonzero voxel, and 1 in 0 type or mus=0.
    vol_array = np.array(cfg["vol"])
    nonzero_ind = np.where(vol_array.flatten() > 0)[0]
    nonzero_prop_row = (
        vol_array.flatten()[nonzero_ind].astype(int) + 1
    )  # +1 for 0-based to 1-based indexing

    mus_matrix = np.ones(vol_shape)
    prop_array = np.array(cfg["prop"])

    # Extract mus values (column 2, which is index 1 in Python)
    for i, idx in enumerate(nonzero_ind):
        flat_idx = np.unravel_index(idx, vol_shape)
        mus_matrix[flat_idx] = prop_array[nonzero_prop_row[i], 1]  # Column 2 is index 1

    # Avoid division by zero if mus=0
    mus_matrix = mus_matrix + (mus_matrix == 0).astype(float)

    # General replay settings.
    cfg_jac = copy.deepcopy(cfg)
    cfg_jac["seed"] = seeds["data"]
    cfg_jac["detphotons"] = detp["data"]
    cfg_jac["omega"] = 2 * np.pi * f_mod  # RF modulation angular frequency
    cfg_jac["isnormalized"] = 0  # Important!
    cfg_jac["issave2pt"] = 1

    # Collect Jacobians one detector index at a time.
    for idx, d in enumerate(detnums):
        # Check if detector has detected photons
        if "detid" in detp and d not in detp["detid"]:
            print(f"MCX WARNING: No detected photons for detector {d}.")
            continue

        # REPLAY SIMULATION 1
        cfg_jac["replaydet"] = d
        cfg_jac["outputtype"] = "rf"  # FD absorption Jacobians
        rfjac_d = mcxlab(cfg_jac)

        rfjac_d_data = rfjac_d["data"]
        rfjac_d_sum = np.sum(rfjac_d_data, axis=3)  # sum over time instances (axis 3)

        if cfg_jac["isnormalized"] == 0:
            rfjac_d_sum = cfg_jac["unitinmm"] * rfjac_d_sum  # correct units to [mm]

        # Jacobians for X and Y wrt mua:
        rfjac_X = rfjac_d_sum[:, :, :, 0]  # (-1*)cos-weighted paths
        rfjac_Y = rfjac_d_sum[:, :, :, 1]  # (-1*)sine-weighted paths
        del rfjac_d, rfjac_d_data, rfjac_d_sum  # Clear memory

        # REPLAY SIMULATION 2
        cfg_jac["outputtype"] = "rfmus"  # FD scattering Jacobians
        rfmusjac_d, detp_d, vol_d, seeds_d = mcxlab(cfg_jac)

        rfmusjac_d_data = rfmusjac_d["data"]
        rfmusjac_d_sum = np.sum(rfmusjac_d_data, axis=3)  # sum over time instances

        # Jacobians for X and Y wrt mus:
        rfmusjac_X = rfmusjac_d_sum[:, :, :, 0]  # cos-weighted nscatt
        rfmusjac_X = rfmusjac_X / mus_matrix + rfjac_X
        del rfjac_X  # Clear memory

        rfmusjac_Y = rfmusjac_d_sum[:, :, :, 1]  # sine-weighted nscatt
        rfmusjac_Y = rfmusjac_Y / mus_matrix + rfjac_Y
        del rfjac_Y, rfmusjac_d, rfmusjac_d_data, rfmusjac_d_sum  # Clear memory

        # FD MEASUREMENT ESTIMATES
        # Array with detected photon weights
        detw = detweight(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"])

        # Array with photon time-of-flights
        dett = dettime(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"])

        X = np.dot(detw, np.cos(cfg_jac["omega"] * dett))
        Y = np.dot(detw, np.sin(cfg_jac["omega"] * dett))
        A = np.sqrt(X**2 + Y**2)  # amplitude [a.u.]

        # Phase shift in [0, 2*pi] [rad]
        phase = np.arctan2(Y, X) + (np.arctan2(Y, X) < 0).astype(float) * 2 * np.pi

        # FINAL SCATTERING JACOBIANS
        if A != 0:
            rfmusjac_lnA[:, :, :, idx] = (1 / (A**2)) * (
                X * rfmusjac_X + Y * rfmusjac_Y
            )
            rfmusjac_phase[:, :, :, idx] = (1 / (A**2)) * (
                X * rfmusjac_Y - Y * rfmusjac_X
            )
        else:
            print(f"MCX WARNING: Zero amplitude for detector {d}.")
            rfmusjac_lnA[:, :, :, idx] = 0
            rfmusjac_phase[:, :, :, idx] = 0

    return rfmusjac_lnA, rfmusjac_phase
