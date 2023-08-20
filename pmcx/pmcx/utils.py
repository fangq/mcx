# Copyright (c) 2023 Kuznetsov Ilya
# Copyright (c) 2023 Qianqian Fang (q.fang <at> neu.edu)
# Copyright (c) 2023 Fan-Yu (Ivy) Yen (yen.f at northeastern.edu)
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
import re
import sys

# sys.path.insert(0, '../../src/build/')

import pmcx


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
    # select the photon data of the specified detector
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
    """
    newdetp = {}
    c0 = 0
    length = 0

    if re.search("[dD]", savedetflag):
        if issaveref is not None and issaveref > 1:
            newdetp["w0"] = detp[0, :].transpose()
        else:
            newdetp["detid"] = detp[0, :].astype(int).transpose()
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
       res=mcxlab(cfg);
          or
       res=mcxlab(cfg, option);

    Input:
       cfg: a struct, or struct array. Each element of cfg defines
            the parameters associated with a simulation.
            if cfg='gpuinfo': return the supported GPUs and their parameters,
            if cfg='version': return the version of MCXLAB as a string,
            see sample script at the bottom
       option: (optional), options is a string, specifying additional options
            option='opencl':  force using mcxcl.mex* instead of mcx.mex* on NVIDIA/AMD/Intel hardware
            option='cuda':    force using mcx.mex* instead of mcxcl.mex* on NVIDIA GPUs

       if one defines USE_MCXCL=1 in as Python global variable, all following
       mcxlab and mcxlabcl calls will use mcxcl.mex; by setting option='cuda', one can
       force both mcxlab and mcxlabcl to use mcx (cuda version). Similarly, if
       USE_MCXCL=0, all mcxlabcl and mcxlab call will use mcx.mex by default, unless
       one set option='opencl'.

    Output:
         fluence: a struct array, with a length equals to that of cfg.
               For each element of fluence,
               fluence(i).data is a 4D array with
                    dimensions specified by [size(vol) total-time-gates].
                    The content of the array is the normalized fluence at
                    each voxel of each time-gate.

                    when cfg.debuglevel contains 'T', fluence(i).data stores trajectory
                    output, see below
               fluence(i).dref is a 4D array with the same dimension as fluence(i).data
                    if cfg.issaveref is set to 1, containing only non-zero values in the
                    layer of voxels immediately next to the non-zero voxels in cfg.vol,
                    storing the normalized total diffuse reflectance (summation of the weights
                    of all escaped photon to the background regardless of their direction);
                    it is an empty array [] when if cfg.issaveref is 0.
               fluence(i).stat is a structure storing additional information, including
                    runtime: total simulation run-time in millisecond
                    nphoton: total simulated photon number
                    energytot: total initial weight/energy of all launched photons
                    energyabs: total absorbed weight/energy of all photons
                    normalizer: normalization factor
                    unitinmm: same as cfg.unitinmm, voxel edge-length in mm

         detphoton: (optional) a struct array, with a length equals to that of cfg.
               Starting from v2018, the detphoton contains the below subfields:
                 detphoton.detid: the ID(>0) of the detector that captures the photon
                 detphoton.nscat: cummulative scattering event counts in each medium
                 detphoton.ppath: cummulative path lengths in each medium (partial pathlength)
                      one need to multiply cfg.unitinmm with ppath to convert it to mm.
                 detphoton.mom: cummulative cos_theta for momentum transfer in each medium
                 detphoton.p or .v: exit position and direction, when cfg.issaveexit=1
                 detphoton.w0: photon initial weight at launch time
                 detphoton.s: exit Stokes parameters for polarized photon
                 detphoton.prop: optical properties, a copy of cfg.prop
                 detphoton.data: a concatenated and transposed array in the order of
                       [detid nscat ppath mom p v w0]'
                 "data" is the is the only subfield in all MCXLAB before 2018
         vol: (optional) a struct array, each element is a preprocessed volume
               corresponding to each instance of cfg. Each volume is a 3D int32 array.
         seeds: (optional), if give, mcxlab returns the seeds, in the form of
               a byte array (uint8) for each detected photon. The column number
               of seed equals that of detphoton.
         trajectory: (optional), if given, mcxlab returns the trajectory data for
               each simulated photon. The output has 6 rows, the meanings are
                  id:  1:    index of the photon packet
                  pos: 2-4:  x/y/z/ of each trajectory position
                       5:    current photon packet weight
                       6:    reserved
               By default, mcxlab only records the first 1e7 positions along all
               simulated photons; change cfg.maxjumpdebug to define a different limit.
    """
    try:
        defaultocl = eval("USE_MCXCL", globals())
    except:
        defaultocl = 0

    useopencl = defaultocl

    if len(args) == 2 and isinstance(args[1], str):
        if args[1] == "preview":
            varargout = mcxpreview(args[0])
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
                args[0]["detphotons"] = detdata.transpose()
                args[0]["savedetflag"] = "dspmxvw"
                args[0]["savedetflag"][detfields == 0] = []

    if useopencl == 0:
        varargout = pmcx.run(args[0])
    else:
        varargout = pmcxcl.run(args[0])

    print(varargout.keys())

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

    if cfg["issavedet"]:
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
                isinstance(cfg["vol"], float) or isinstance(cfg["vol"], double)
            ) and "unitinmm" in cfg:  # if((isa(cfg(i).vol,'single') || isa(cfg(i).vol,'double')) && isfield(cfg(i),'unitinmm'))
                cfg["vol"] = cfg["vol"] * cfg["unitinmm"]

        if "issaveexit" not in cfg or cfg.issaveexit != 2:
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
            data = varargout["traj"]["data"]
            if data:
                traj = {}
                traj["pos"] = np.transpose(data[1:4, :])
                traj["id"] = np.uint32(data[0, :])
                traj["id"], idx = np.sort(traj["id"]), np.argsort(traj["id"])
                traj["pos"] = traj["pos"][idx, :]
                traj["data"] = np.vstack([np.single(traj["id"]), data[1:, idx]])
                varargout["traj"]["data"] = traj

    return varargout
