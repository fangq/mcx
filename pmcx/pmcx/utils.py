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


"""Post-processing, analytical models and the mcxlab wrapper of PMCX.

This module provides the Python versions of the MATLAB helper functions of MCXLAB:

* detected photon processing: :func:`detphoton`, :func:`detweight`, :func:`dettime`,
  :func:`dettpsf`, :func:`meanpath`, :func:`meanscat`, :func:`cwdref`
* analytical diffusion solutions of a semi-infinite medium: :func:`cwdiffusion`,
  :func:`cwfluxdiffusion`, :func:`tddiffusion`, and :func:`getdistance`
* Jacobians from photon replay: :func:`rfreplay`, :func:`rfmusreplay`
* diffuse correlation spectroscopy: :func:`dcsg1`
* inputs: :func:`mcxcreate` (the built-in benchmarks), :func:`svmc` (split-voxel volumes)
* :func:`mcxlab`, which wraps ``pmcx.run`` with the pre- and post-processing of mcxlab.m

All lengths are in mm unless stated otherwise; the optical property table ``prop`` has one
row ``[mua, mus, g, n]`` per medium, row 0 being the background (medium 0).
"""

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
    """Compute the continuous-wave diffuse reflectance at each detector.

    The weights of the detected photons (recomputed from their partial path lengths by
    :func:`detweight`) are summed per detector and divided by the detector area and the
    number of launched photons.

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs the
        ``'detid'`` and ``'ppath'`` keys.
    cfg : dict
        The simulation configuration; uses ``cfg['prop']``, ``cfg['detpos']`` (``N x 4``,
        the 4th column is the detector radius), ``cfg['nphoton']`` and, if present,
        ``cfg['unitinmm']``.

    Returns
    -------
    dref : ndarray, shape (N,)
        Diffuse reflectance (1/mm^2) of each of the ``N`` detectors in ``cfg['detpos']``,
        including those that captured no photon (0).

    Notes
    -----
    Ported from mcxcwdref.m by Kuznetsov Ilya. See Yao et al., Biomed. Opt. Express 9(10),
    4588 (2018).
    """

    unitinmm = 1
    if "unitinmm" in cfg:
        unitinmm = cfg["unitinmm"]

    det_weight = detweight(detp, cfg["prop"])
    detpos = np.atleast_2d(np.asarray(cfg["detpos"], dtype=float))
    detnum = detpos.shape[0]  # all detectors, including those that captured no photon
    detid = np.asarray(detp["detid"]).astype(int).ravel() - 1
    detweightsum = np.bincount(detid, weights=np.ravel(det_weight), minlength=detnum)[
        :detnum
    ]

    area = np.pi * (detpos[:, 3] * unitinmm) ** 2
    dref = detweightsum / area / cfg["nphoton"]

    return dref


# ---------------------------------------------
def meanpath(detp, prop=None):
    """Compute the average path length of the detected photons in each medium.

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs ``'ppath'``,
        and ``'unitinmm'`` if the voxel size is not 1 mm.
    prop : array_like, optional
        Optical property table ``[mua, mus, g, n]`` per medium (``cfg['prop']``); taken
        from ``detp['prop']`` when omitted.

    Returns
    -------
    avgpath : ndarray, shape (Nmedia,)
        Photon-weight-averaged path length (mm) in each medium, medium 1 first.

    Raises
    ------
    ValueError
        If ``prop`` is not given and ``detp`` has no ``'prop'`` key.

    Notes
    -----
    Ported from mcxmeanpath.m by Fan-Yu Yen.
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
    """Recompute the weight of each detected photon from its partial path lengths.

    The weight of a photon is ``w0 * prod_i exp(-mua_i * L_i)``, with ``L_i`` its path length
    in medium ``i``; replacing ``prop`` by perturbed optical properties gives the detector
    reading of the perturbed medium (perturbation Monte Carlo).

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs ``'ppath'``,
        and uses ``'w0'`` (the initial weights) when present.
    prop : array_like, optional
        Optical property table ``[mua, mus, g, n]`` per medium; taken from ``detp['prop']``
        when omitted.
    unitinmm : float, optional
        Voxel edge length in mm used by the simulation; taken from ``detp['unitinmm']``,
        or 1 when neither is given.

    Returns
    -------
    detw : ndarray
        Weight of each detected photon, shape ``(nphoton,)``, or ``(nphoton, nsrc)`` when
        ``detp['w0']`` holds one column per source pattern.

    Raises
    ------
    ValueError
        If ``prop`` is missing or has fewer than 2 rows, or if ``detp`` is not a dict.

    Notes
    -----
    Ported from mcxdetweight.m by Fan-Yu Yen.
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
        ppath = np.asarray(detp["ppath"])
        prop = np.asarray(prop)

        # attenuation of each photon along its partial paths, shape (nphoton,)
        detw = np.ones(ppath.shape[0])
        for i in range(medianum - 1):
            detw = detw * np.exp(-prop[i + 1, 0] * ppath[:, i] * unitinmm)

        if "w0" in detp:
            # pmcx returns w0 as a (nphoton, 1) column, or one column per source
            # for pattern sources; multiplying it by the 1-D attenuation directly
            # broadcast to (nphoton, nphoton), see issue #254
            w0 = np.asarray(detp["w0"]).reshape(ppath.shape[0], -1)
            detw = w0 * detw[:, np.newaxis]
            if detw.shape[1] == 1:
                detw = detw[:, 0]
    else:
        raise ValueError('The first input must be a dict with a key named "ppath"')

    return detw


def meanscat(detp, prop):
    """Compute the photon-weight-averaged number of scattering events.

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs ``'nscat'``
        and ``'ppath'``.
    prop : array_like
        Optical property table ``[mua, mus, g, n]`` per medium.

    Returns
    -------
    avgnscat : float
        The scattering event counts of all media, summed and averaged over the detected
        photons with their weights (from :func:`detweight`) as the weighting.

    Notes
    -----
    Ported from mcxmeanscat.m by Fan-Yu Yen; unlike the MATLAB function, which returns one
    average per medium, the counts of all media are summed.
    """
    detw = detweight(detp, prop)
    avgnscat = np.sum(
        np.array(detp["nscat"], dtype=float) * detw[:, np.newaxis]
    ) / np.sum(detw)

    return avgnscat


def dettpsf(detp, detnum, prop, time):
    """Compute the temporal point spread function (TPSF) of one detector.

    The detected photons of the detector are binned by their time of flight (from
    :func:`dettime`), each contributing its weight (from :func:`detweight`).

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs ``'detid'`` and
        ``'ppath'``, and uses ``'w0'`` when present.
    detnum : int
        The detector ID (1-based).
    prop : array_like
        Optical property table ``[mua, mus, g, n]`` per medium.
    time : array_like
        The time bins ``[tstart, tend, tstep]`` in seconds; they can differ from those of
        the simulation.

    Returns
    -------
    tpsf : ndarray, shape (ntime, 1)
        The summed photon weight in each time bin; photons outside ``[tstart, tend]`` are
        added to the first or the last bin.

    Notes
    -----
    Ported from mcxdettpsf.m by Fan-Yu Yen.
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
    """Recompute the time of flight of each detected photon from its partial path lengths.

    The time of flight is ``sum_i n_i * L_i / c0``, with ``n_i`` the refractive index and
    ``L_i`` the path length in medium ``i``.

    Parameters
    ----------
    detp : dict
        Detected photon data, the ``'detp'`` output of :func:`mcxlab`; needs ``'ppath'``.
    prop : ndarray, optional
        Optical property table ``[mua, mus, g, n]`` per medium; taken from ``detp['prop']``
        when omitted.
    unitinmm : float, optional
        Voxel edge length in mm used by the simulation; taken from ``detp['unitinmm']``,
        or 1 when neither is given.

    Returns
    -------
    dett : ndarray, shape (1, nphoton)
        Time of flight of each detected photon, in seconds.

    Raises
    ------
    ValueError
        If ``prop`` is missing or has fewer than 2 rows, or if ``detp`` is not a dict.

    Notes
    -----
    Ported from mcxdettime.m by Fan-Yu Yen.
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
    """Time-resolved diffusion solution of a semi-infinite medium.

    Parameters
    ----------
    mua : float
        Absorption coefficient, in 1/mm.
    musp : float
        Reduced scattering coefficient, in 1/mm.
    v : float
        Speed of light in the medium, in mm/s.
    Reff : float
        Effective reflection coefficient of the boundary.
    srcpos : array_like
        Source position(s) ``[x, y, z]`` in mm, the medium occupying ``z > 0``.
    detpos : array_like
        Detector position(s) ``[x, y, z]`` in mm.
    t : array_like
        Times in seconds at which the solution is evaluated.

    Returns
    -------
    Phi : ndarray
        Fluence rate in 1/(mm^2 s) at the detectors and times, by the extrapolated-boundary
        image source method.

    Notes
    -----
    Ported from tddiffusion.m by Fan-Yu Yen.
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
    """Compute the distances between all sources and detectors.

    Parameters
    ----------
    srcpos : array_like
        Source positions, one ``[x, y, z]`` row per source.
    detpos : array_like
        Detector positions, one ``[x, y, z]`` row per detector.

    Returns
    -------
    separation : ndarray, shape (ndet, nsrc)
        ``separation[r, s]`` is the distance between detector ``r`` and source ``s``.

    Notes
    -----
    Ported from getdistance.m by Fan-Yu Yen.
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
    """Split the raw detected photon array into named fields.

    Parameters
    ----------
    detp : ndarray, shape (ncol, nphoton)
        The combined detected photon data (``detp['data']`` of :func:`mcxlab`, or the
        ``'detp'`` output of ``pmcx.run``), one column per photon.
    medianum : int
        The number of media, excluding the background (``len(cfg['prop']) - 1``).
    savedetflag : str
        The ``cfg['savedetflag']`` letters, in the order ``d s p m x v w i``, giving the
        fields present in ``detp``.
    issaveref : int, optional
        ``cfg['issaveref']``; when greater than 1 the first row holds the initial weight
        ``'w0'`` instead of the detector ID.
    srcnum : int, optional
        ``cfg['srcnum']``; with more than one source pattern ``'w0'`` holds the pattern
        index as uint32.

    Returns
    -------
    newdetp : dict
        The fields present, each an array with one row per photon:

        ``'detid'``
            ID (1-based) of the detector that captured the photon.
        ``'srcid'``
            ID of the source, when several sources are defined.
        ``'nscat'``
            Scattering event counts in each medium (uint32).
        ``'ppath'``
            Partial path length in each medium, in voxel units (multiply by
            ``cfg['unitinmm']`` for mm).
        ``'mom'``
            Momentum transfer (summed ``1 - cos(theta)``) in each medium.
        ``'p'``, ``'v'``
            Exit position and direction.
        ``'w0'``
            Initial weight of the photon.
        ``'s'``
            Exit Stokes vector ``[I, Q, U, V]`` of polarized photons.

    Notes
    -----
    Ported from mcxdetphoton.m by Fan-Yu Yen.
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
        nscat = np.ascontiguousarray(detp[c0 : c0 + length, :])
        # the kernel stores nscat as uint32 bits inside the float32 detp buffer; other
        # dtypes (e.g. user-assembled float64 arrays) hold the counts as values
        nscat = (
            nscat.view(np.uint32)
            if nscat.dtype == np.float32
            else nscat.astype(np.uint32)
        )
        newdetp["nscat"] = nscat.transpose()
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
    """Run an MCX simulation with the pre- and post-processing of MATLAB's mcxlab.

    :func:`mcxlab` checks and converts the inputs, calls ``pmcx.run`` (CUDA) or
    ``pmcxcl.run`` (OpenCL), and reorganizes the outputs as mcxlab.m does: the detected
    photon data are split into named fields, the trajectories are sorted by photon, and the
    outputs of several source patterns are moved to the last dimension. ``help(pmcx.run)``
    lists all the ``cfg`` keys.

    Parameters
    ----------
    cfg : dict or str
        The simulation configuration (see ``pmcx.run``), or ``'gpuinfo'`` to list the
        GPUs, or ``'version'`` to return the version string.
    option : str, optional
        ``'opencl'`` to run with pmcxcl (OpenCL) instead of pmcx (CUDA), or ``'preview'`` to
        plot the domain, sources and detectors without simulating. Without it, the backend is
        set by a ``USE_MCXCL`` variable in the user's session (1: OpenCL, default 0); pmcxcl
        is also used when pmcx has no CUDA binary on this platform.

    Returns
    -------
    res : dict
        The outputs, as ``pmcx.run`` returns them, with these reorganized:

        ``'flux'``
            Fluence (rate) of shape ``[Nx, Ny, Nz, Ntime]``, with one more last dimension
            for several source patterns; complex for RF outputs.
        ``'dref'``
            Diffuse reflectance, when ``cfg['issaveref']`` is set.
        ``'detp'``
            Detected photon data as a dict of named fields (see :func:`detphoton`), with
            ``'prop'``, ``'unitinmm'`` and the raw array ``'data'``.
        ``'traj'``
            Photon trajectories (``cfg['debuglevel']`` containing ``'M'``): ``'id'``
            (photon index, sorted), ``'pos'`` (positions) and ``'data'``.
        ``'seeds'``, ``'vol'``, ``'prop'``, ``'stat'``
            As returned by ``pmcx.run``.

    Raises
    ------
    ValueError
        If ``cfg`` lacks ``'tend'`` or ``'srcpos'``.
    ImportError
        If neither pmcx nor pmcxcl provides a simulator binary.

    Examples
    --------
    >>> import numpy as np, pmcx
    >>> cfg = {'nphoton': 1000000, 'vol': np.ones([60, 60, 60], dtype='uint8'),
    ...        'tstart': 0, 'tend': 5e-9, 'tstep': 5e-9, 'srcpos': [30, 30, 0],
    ...        'srcdir': [0, 0, 1], 'prop': [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]}
    >>> res = pmcx.mcxlab(cfg)
    >>> res['flux'].shape
    (60, 60, 60, 1)

    Notes
    -----
    Adapted from mcxlab.m by Fan-Yu Yen.
    """
    try:
        defaultocl = eval("USE_MCXCL", globals())
    except:
        # USE_MCXCL set in the user's session lives in __main__, not in this module
        import __main__

        defaultocl = getattr(__main__, "USE_MCXCL", 0)

    useopencl = defaultocl

    # the backend must be chosen before importing run(), so read option='opencl' first
    if len(args) == 2 and isinstance(args[1], str) and args[1] == "opencl":
        useopencl = 1

    if useopencl == 0:
        try:
            from _pmcx import gpuinfo, run, version
        except ImportError:
            # no CUDA build of pmcx (e.g. the pure-python wheel on macOS): use pmcxcl if present
            try:
                from _pmcxcl import gpuinfo, run, version
            except ImportError:
                raise ImportError(
                    "neither the pmcx (CUDA) nor the pmcxcl (OpenCL) binary extension is "
                    'available; run "pip install pmcxcl" to simulate on this platform'
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
            args[0]["tstep"] = args[0]["tend"]

        if "srcpos" not in args[0]:
            raise ValueError(
                "you must define cfg.srcpos to define the x/y/z position of the source in voxel unit"
            )

        if "detphotons" in args[0] and isinstance(args[0]["detphotons"], dict):
            if "data" in args[0]["detphotons"]:
                args[0]["detphotons"] = args[0]["detphotons"]["data"]
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
    """Create the configuration of a built-in benchmark, as ``mcx --bench`` does.

    Parameters
    ----------
    benchname : str, optional
        The benchmark: ``'cube60'``, ``'cube60b'``, ``'cube60planar'``, ``'skinvessel'``,
        ``'sphshell'`` or ``'spherebox'``.
    **kwargs
        ``cfg`` keys (lower case) to add or override.

    Returns
    -------
    cfg : dict or list of str
        The configuration, ready for :func:`mcxlab` or ``pmcx.run``; without
        ``benchname``, the list of the benchmark names.

    Raises
    ------
    ValueError
        If the benchmark name is not known.

    Examples
    --------
    >>> import pmcx
    >>> pmcx.mcxcreate()
    ['cube60', 'cube60b', 'cube60planar', 'skinvessel', 'sphshell', 'spherebox']
    >>> cfg = pmcx.mcxcreate('cube60', srctype='isotropic', srcpos=[30, 30, 30])
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
    """Compute the electric-field autocorrelation g1 of diffuse correlation spectroscopy.

    g1 of each detector is computed from the path lengths and momentum transfers of its
    detected photons, for a model of the root-mean-square displacement of the scatterers.

    Parameters
    ----------
    detps : str or dict
        The detected photon data, the ``'detp'`` output of :func:`mcxlab` with
        ``'ppath'``, ``'mom'`` and ``'prop'`` (run with ``cfg['ismomentum'] = 1``), or the
        name of an ``.mch`` file, whose optical properties are read from the ``.json`` file
        of the same name.
    tau : array_like, optional
        Correlation times in seconds; 200 log-spaced values from 1e-7 to 1e-1 by default.
    disp_model : str, optional
        The displacement model: ``'brownian'`` (default, ``<dr^2> = 6 DV tau``),
        ``'random_flow'`` (``<dr^2> = DV^2 tau^2``), or a Python expression of ``DV`` and
        ``tau[J]`` evaluated as is.
    DV : float, optional
        The displacement variable, in mm and s: the Brownian diffusion coefficient
        (default 1e-7 mm^2/s) or the flow velocity.
    lambda_ : float, optional
        Wavelength in nm (default 785).
    format : str, optional
        Data type of the ``.mch`` file (default ``'float'``).
    **varargin
        Extra variables available to a custom ``disp_model`` expression.

    Returns
    -------
    tau : ndarray
        The correlation times.
    g1 : ndarray, shape (ndet, ntau)
        g1 of each detector, row ``i`` for detector ``i + 1``.

    Raises
    ------
    ValueError
        If the data have no momentum transfer.

    Notes
    -----
    Ported from mcxdcsg1.m (Stefan Carp, MGH).
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
    """Continuous-wave diffusion solution of a semi-infinite medium.

    Parameters
    ----------
    mua : float
        Absorption coefficient, in 1/mm.
    musp : float
        Reduced scattering coefficient, in 1/mm.
    Reff : float
        Effective reflection coefficient of the boundary.
    srcpos : array_like
        Source position(s) ``[x, y, z]`` in mm, the medium occupying ``z > 0``.
    detpos : array_like
        Detector position(s) ``[x, y, z]`` in mm.

    Returns
    -------
    Phi : ndarray, shape (ndet, nsrc)
        Fluence in 1/mm^2 for each detector/source pair, by the extrapolated-boundary image
        source method.
    r : ndarray, shape (ndet, nsrc)
        Distances from the (depth-shifted) real sources to the detectors.

    Notes
    -----
    See Boas et al. (2002) and Haskell et al. (1994).
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
    """Continuous-wave diffuse reflectance (surface flux) of a semi-infinite medium.

    Parameters
    ----------
    mua : float
        Absorption coefficient, in 1/mm.
    musp : float
        Reduced scattering coefficient, in 1/mm.
    Reff : float
        Effective reflection coefficient of the boundary.
    srcpos : array_like
        Source position(s) ``[x, y, z]`` in mm.
    detpos : array_like
        Detector position(s) ``[x, y, z]`` in mm, on the surface.

    Returns
    -------
    flux : ndarray, shape (ndet, nsrc)
        Diffuse reflectance in 1/mm^2 for each detector/source pair (Eq. 6 of Kienle and
        Patterson, 1997).
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
    """Continuous-wave fluence of a semi-infinite medium; the same as :func:`cwdiffusion`.

    Parameters
    ----------
    *args, **kwargs
        The arguments of :func:`cwdiffusion`.

    Returns
    -------
    Phi, r : ndarray
        As :func:`cwdiffusion` returns them.
    """
    return cwdiffusion(*args, **kwargs)


def _replaydata(x):
    """Return the raw array of a detected photon or seed output, or of a MATLAB-style dict holding it in ``'data'``."""
    return x["data"] if isinstance(x, dict) and "data" in x else x


def _rfparts(flux, vol_shape, unitinmm=None):
    """Sum an RF replay output over its time gates and split it into its cosine (X) and sine (Y) weighted parts."""
    jac = np.sum(
        np.asarray(flux), axis=3
    )  # [Nx, Ny, Nz, 1, 2] after summing the time gates
    if unitinmm is not None:
        jac = unitinmm * jac  # correct units to [mm] when the output is not normalized
    jac = np.reshape(jac, list(vol_shape) + [-1])
    return jac[..., 0], jac[..., -1]


def rfreplay(cfg, f_mod, detp, seeds, detnums):
    """Compute frequency-domain Jacobians of log-amplitude and phase with respect to mua by RF replay.

    The detected photons of a baseline simulation, run with ``cfg['issaveseed'] = 1``, are
    replayed for each detector with ``cfg['outputtype'] = 'rf'``. For a forward
    frequency-domain solution without a baseline, run :func:`mcxlab` with
    ``cfg['outputtype'] = 'rf'`` and ``cfg['omega']`` instead.

    Parameters
    ----------
    cfg : dict
        The configuration of the baseline simulation.
    f_mod : float
        Source modulation frequency, in Hz.
    detp : dict or ndarray
        The detected photons of the baseline run (the ``'detp'`` output of
        :func:`mcxlab`).
    seeds : dict or ndarray
        Their RNG seeds (the ``'seeds'`` output).
    detnums : array_like
        The IDs (1-based) of the detectors to replay.

    Returns
    -------
    rfjac_lnA : ndarray, shape (Nx, Ny, Nz, ndet)
        Jacobian of the log-amplitude with respect to the mua of each voxel, per detector.
    rfjac_phase : ndarray, shape (Nx, Ny, Nz, ndet)
        Jacobian of the phase shift (rad), per detector.

    Raises
    ------
    ValueError
        If an input is missing.

    Notes
    -----
    By Pauliina Hirvi and Qianqian Fang. See Hirvi et al., "Effects of atlas-based anatomy
    on modelled light transport in the neonatal head," Phys. Med. Biol. 68, 135019 (2023).
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
        cfg_jac["seed"] = _replaydata(seeds)
        cfg_jac["detphotons"] = _replaydata(detp)
        cfg_jac["replaydet"] = d
        cfg_jac["outputtype"] = "rf"
        cfg_jac["omega"] = 2 * np.pi * f_mod  # RF modulation frequency
        cfg_jac["isnormalized"] = 0  # Important!
        cfg_jac["issave2pt"] = 1

        # REPLAY SIMULATION
        res = mcxlab(cfg_jac)
        rfjac_d, detp_d = res["flux"], res["detp"]

        # Array with detected photon weights
        detw = np.ravel(detweight(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"]))

        # Array with photon time-of-flights
        dett = np.ravel(dettime(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"]))

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
        # Jacobians for X and Y wrt mua, summed over time instances
        rfjac_X, rfjac_Y = _rfparts(
            rfjac_d,
            vol_shape,
            cfg_jac["unitinmm"] if cfg_jac["isnormalized"] == 0 else None,
        )

        # Jacobians for log-amplitude and phase shift wrt mua:
        rfjac_lnA[:, :, :, idx] = (1 / (A**2)) * (X * rfjac_X + Y * rfjac_Y)
        rfjac_phase[:, :, :, idx] = (1 / (A**2)) * (X * rfjac_Y - Y * rfjac_X)

    return rfjac_lnA, rfjac_phase


def rfmusreplay(cfg, f_mod, detp, seeds, detnums):
    """Compute frequency-domain Jacobians of log-amplitude and phase with respect to mus by RF replay.

    Each detector is replayed twice, with ``cfg['outputtype'] = 'rf'`` (the absorption part)
    and ``'rfmus'`` (the scattering part), from a baseline simulation run with
    ``cfg['issaveseed'] = 1``.

    Parameters
    ----------
    cfg : dict
        The configuration of the baseline simulation.
    f_mod : float
        Source modulation frequency, in Hz.
    detp : dict or ndarray
        The detected photons of the baseline run (the ``'detp'`` output of
        :func:`mcxlab`).
    seeds : dict or ndarray
        Their RNG seeds (the ``'seeds'`` output).
    detnums : array_like
        The IDs (1-based) of the detectors to replay.

    Returns
    -------
    rfmusjac_lnA : ndarray, shape (Nx, Ny, Nz, ndet)
        Jacobian of the log-amplitude with respect to the mus of each voxel, per detector.
    rfmusjac_phase : ndarray, shape (Nx, Ny, Nz, ndet)
        Jacobian of the phase shift (rad), per detector.

    Raises
    ------
    ValueError
        If an input is missing.

    Notes
    -----
    By Pauliina Hirvi and Qianqian Fang. See Hirvi et al., "Computation of frequency- and
    time-domain Jacobians in optical tomography with Monte Carlo simulations,"
    arXiv:2604.27558 (2026), and Phys. Med. Biol. 68, 135019 (2023).
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
    if (
        detp is None
        or seeds is None
        or np.size(_replaydata(detp)) == 0
        or np.size(_replaydata(seeds)) == 0
    ):
        print("MCX WARNING: No detected photons for replay.")
        return rfmusjac_lnA, rfmusjac_phase

    # Form matrix with mus for each nonzero voxel, and 1 in 0 type or mus=0.
    vol_array = np.asarray(cfg["vol"]).astype(int)
    prop_array = np.asarray(cfg["prop"], dtype=float)
    mus_matrix = np.where(
        vol_array > 0, prop_array[vol_array, 1], 1.0
    )  # labels index prop rows directly

    # Avoid division by zero if mus=0
    mus_matrix = mus_matrix + (mus_matrix == 0).astype(float)

    # General replay settings.
    cfg_jac = copy.deepcopy(cfg)
    cfg_jac["seed"] = _replaydata(seeds)
    cfg_jac["detphotons"] = _replaydata(detp)
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
        cfg_rf = dict(
            cfg_jac, issavedet=0
        )  # only the Jacobian is needed, as in mcxrfmusreplay.m
        rfjac_d = mcxlab(cfg_rf)["flux"]

        # Jacobians for X and Y wrt mua: (-1*)cos/sine-weighted paths
        rfjac_X, rfjac_Y = _rfparts(
            rfjac_d,
            vol_shape,
            cfg_jac["unitinmm"] if cfg_jac["isnormalized"] == 0 else None,
        )
        del rfjac_d  # Clear memory

        # REPLAY SIMULATION 2
        cfg_jac["outputtype"] = "rfmus"  # FD scattering Jacobians
        res = mcxlab(cfg_jac)
        detp_d = res["detp"]

        # Jacobians for X and Y wrt mus: cos/sine-weighted nscatt, summed over time instances
        rfmusjac_X, rfmusjac_Y = _rfparts(res["flux"], vol_shape)
        rfmusjac_X = rfmusjac_X / mus_matrix + rfjac_X
        rfmusjac_Y = rfmusjac_Y / mus_matrix + rfjac_Y
        del rfjac_X, rfjac_Y, res  # Clear memory

        # FD MEASUREMENT ESTIMATES
        # Array with detected photon weights
        detw = np.ravel(detweight(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"]))

        # Array with photon time-of-flights
        dett = np.ravel(dettime(detp_d, cfg_jac["prop"], cfg_jac["unitinmm"]))

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


def svmc(vol, smoothing=1, kernelsize=3, kernelstd=1, threshold=0.5, curveonly=1):
    """Convert a label volume to the split-voxel (SVMC) format.

    The boundary of each label is extracted by marching cubes, after an optional Gaussian
    smoothing; each voxel cut by a boundary is then described by the labels on its two sides
    and a planar interface (a reference point and a normal). Requires scipy and
    scikit-image.

    Parameters
    ----------
    vol : array_like
        A 3-D integer label volume, up to 255 labels.
    smoothing : int, optional
        1 (default) to smooth each label mask with a Gaussian kernel first.
    kernelsize : int, optional
        Size of the Gaussian kernel (default 3).
    kernelstd : float, optional
        Standard deviation of the Gaussian kernel (default 1).
    threshold : float, optional
        The isosurface level (default 0.5).
    curveonly : int, optional
        1 (default) to keep only the curved interfaces, leaving the voxel-aligned flat
        boundaries as plain label voxels.

    Returns
    -------
    newvol : ndarray, shape (8, Nx, Ny, Nz), uint8
        The SVMC volume, to be passed as ``cfg['vol']``: ``newvol[0]`` the label (the lower
        label of a split voxel), ``newvol[1]`` the upper label of a split voxel or 0,
        ``newvol[2:5]`` the reference point of the interface in the voxel (times 255), and
        ``newvol[5:8]`` the interface normal (``(n + 1) * 255 / 2``).

    Raises
    ------
    ValueError
        If the volume has more than 255 labels.

    Notes
    -----
    Port of mcxsvmc.m. See Yan and Fang, "Hybrid mesh and voxel based Monte Carlo algorithm
    for accurate and efficient photon transport modeling in complex bio-tissues," Biomed.
    Opt. Express 11(11), 6262 (2020).
    """
    from scipy.ndimage import convolve
    from skimage.measure import marching_cubes

    vol = np.asarray(vol)
    dim = vol.shape
    labels = np.unique(vol)
    if len(labels) > 255:
        raise ValueError("MCX currently supports up to 255 labels for this function")

    if smoothing:
        x = np.arange(kernelsize) - (kernelsize - 1) / 2
        g = np.exp(-(x**2) / (2 * kernelstd**2))
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel /= kernel.sum()

    def voxelid(points):
        # linear (column-major) index of the voxel containing each point; points use 1-based
        # voxel coordinates as MATLAB's isosurface over ndgrid(1:N)
        sub = np.floor(points).astype(int)  # = floor(p - 1) + 1 - 1, 0-based
        return np.ravel_multi_index((sub[:, 0], sub[:, 1], sub[:, 2]), dim, order="F")

    bmask = np.full(int(np.prod(dim)), np.nan)
    bmask2 = np.zeros(int(np.prod(dim)))
    verts, faces = [], []
    nvert = 0
    for lab in labels:
        mask = (vol == lab).astype(np.float64)
        volsmooth = convolve(mask, kernel, mode="nearest") if smoothing else mask
        if volsmooth.min() > threshold or volsmooth.max() < threshold:
            continue
        v, f, _, _ = marching_cubes(volsmooth, threshold)
        if len(v) == 0:
            continue
        v = v + 1  # 1-based grid coordinates
        voxid = voxelid(v[f].mean(axis=1))
        uniqidx = np.unique(voxid)
        bmask2[uniqidx] = lab
        uniqidx = uniqidx[np.isnan(bmask[uniqidx])]
        good = np.isin(voxid, uniqidx)
        faces.append(f[good] + nvert)
        verts.append(v)
        nvert += len(v)
        bmask[uniqidx] = lab

    newvol = np.zeros([8] + list(dim), dtype=np.uint8)
    newvol[0] = vol
    if not verts:
        return newvol

    # remove isolated nodes
    verts, faces = np.vstack(verts), np.vstack(faces)
    used, faces = np.unique(faces, return_inverse=True)
    faces = faces.reshape(-1, 3)
    verts = verts[used]

    tri = verts[faces]
    centroids = tri.mean(axis=1)
    voxid = voxelid(centroids)
    uniqidx = np.unique(voxid)

    labpair = np.column_stack([bmask[uniqidx], bmask2[uniqidx]])
    labpair[labpair[:, 0] == labpair[:, 1], 1] = 0

    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = (
        -cross / np.maximum(2 * areas, 1e-30)[:, None]
    )  # orientation of MATLAB's isosurface normals

    nvox = int(np.prod(dim))
    totalarea = np.bincount(voxid, weights=areas, minlength=nvox)
    nc = np.stack(
        [
            np.bincount(voxid, weights=centroids[:, i] * areas, minlength=nvox)
            for i in range(3)
        ]
    )
    nn = np.stack(
        [
            np.bincount(voxid, weights=normals[:, i] * areas, minlength=nvox)
            for i in range(3)
        ]
    )
    nc = (nc[:, uniqidx] / totalarea[uniqidx]).T
    nn = nn[:, uniqidx]
    nn = (nn / np.sqrt((nn * nn).sum(axis=0))).T

    if curveonly:
        # drop voxel-aligned flat interfaces, and their neighbors
        flat = np.any(nc - np.floor(nc) < 1e-6, axis=1) & np.any(
            np.abs(nn) == 1, axis=1
        )
        boxmask = np.zeros(nvox)
        boxmask[uniqidx[flat]] = 1
        boxmask = (
            convolve(
                boxmask.reshape(dim, order="F"), np.ones([3, 3, 3]) / 27, mode="nearest"
            )
            > 0
        )
        keep = ~boxmask.ravel(order="F")[uniqidx]
        nn, nc, uniqidx, labpair = nn[keep], nc[keep], uniqidx[keep], labpair[keep]

    nc = np.floor((nc - np.floor(nc)) * 255)
    nn = np.minimum(np.floor((nn + 1) * 255 / 2), 254)

    flat = newvol.reshape(8, -1, order="F")
    flat[0:2, uniqidx] = labpair.T
    flat[2:5, uniqidx] = nc.T
    flat[5:8, uniqidx] = nn.T
    return flat.reshape([8] + list(dim), order="F")
