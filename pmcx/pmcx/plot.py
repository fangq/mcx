# Copyright (c) 2023-2025 Qianqian Fang (q.fang <at> neu.edu)
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


"""Plotting MCX/MMC inputs and outputs.

* :func:`preview`: draw the domain, sources and detectors of a ``cfg``
* :func:`plotshapes`: draw the objects of a JSON shape description
* :func:`plotphotons`: draw the photon trajectories of a simulation
* :func:`plotvol`: browse a 3-D or 4-D volume interactively

All of them import ``matplotlib`` and ``iso2mesh`` (``pip install iso2mesh``) only when
called.
"""

__all__ = (
    "preview",
    "plotshapes",
    "plotphotons",
    "plotvol",
)

##====================================================================================
## dependent libraries
##====================================================================================

import warnings
import numpy as np


##====================================================================================
## implementation
##====================================================================================


def plotvol(*args, **kwargs):
    """Browse a 3-D or 4-D volume, such as a fluence output, with interactive slices.

    A thin wrapper of ``iso2mesh.plotvolume``, the Python port of ``mcxplotvol``: drag the
    slices with the left mouse button, rotate with the middle button, change the color range
    with the right button, and step through the 4th dimension (e.g. time gates) with the
    up/down keys.

    Parameters
    ----------
    *args, **kwargs
        The arguments of ``iso2mesh.plotvolume``: the volume (``(nx, ny, nz)`` or
        ``(nx, ny, nz, nt)``), and options such as ``subsample`` and ``cmap``.

    Returns
    -------
    viewer : iso2mesh.plotvolume
        The viewer object; ``viewer.setslice(dim, index)`` moves a slice.

    Examples
    --------
    >>> res = pmcx.mcxlab(cfg)
    >>> pmcx.plotvol(np.log10(res["flux"][:, :, :, 0]))
    """
    from iso2mesh import plotvolume

    return plotvolume(*args, **kwargs)


def preview(cfg, **kwargs):
    """Draw the domain, sources and detectors of a simulation, for mcx and mmc inputs.

    For a voxel domain (``cfg['vol']``), the surface of each label is drawn in a fixed
    color; for a mesh (``cfg['node']`` and ``cfg['elem']``), the mesh is drawn instead.
    ``cfg['shapes']`` is drawn with :func:`plotshapes`. The source is a red star with an
    arrow along ``srcdir``; the aperture of area sources (disk, gaussian, planar, pattern,
    fourier, slit, line ...) is outlined, and each detector is drawn as a sphere. Coordinates
    are scaled by ``cfg['unitinmm']``.

    Parameters
    ----------
    cfg : dict or list of dict
        The simulation configuration(s), with the keys of ``pmcx.run`` or ``pmmc.run``;
        one figure is drawn for each. ``srcpos`` and ``srcdir`` are required, as is one of
        ``vol``, ``node`` or ``shapes``.
    **kwargs
        Passed to the matplotlib/iso2mesh plotting calls.

    Returns
    -------
    hs : list of dict
        For each ``cfg``, the plot handles: ``bbx`` (bounding box), ``seg`` (labels or
        shapes), ``src``, ``srcarrow``, ``srcarea`` and ``det``.

    Raises
    ------
    ImportError
        If ``iso2mesh`` or ``matplotlib`` is not installed.
    ValueError
        If a required key is missing.

    Examples
    --------
    >>> pmcx.preview(pmcx.mcxcreate("spherebox"))
    """

    # Import iso2mesh functions
    try:
        import matplotlib.pyplot as plt
        from iso2mesh import plotmesh, plotsurf, latticegrid, binsurface
        from iso2mesh import meshacylinder
    except ImportError:
        warnings.warn("iso2mesh module not available, some functions may not work")
        raise ImportError("you must install the iso2mesh toolbox first")

    if cfg is None:
        raise ValueError("input field cfg must be defined")

    if not isinstance(cfg, (dict, list)):
        raise ValueError("cfg must be a dict or list of dict")

    # Convert single dict to list for uniform processing
    if isinstance(cfg, dict):
        cfg = [cfg]
    cfg = [
        dict(c) for c in cfg
    ]  # shallow copies, so that the 4-D vol replacement below stays local

    length = len(cfg)
    hs = [None] * length

    # a fixed color for each label, the same as mcxpreview.m: a golden-ratio sequence
    # rather than seeded random numbers, so numpy's random state is left untouched
    surfcolors = np.mod(
        np.arange(1, 1025)[:, None]
        * np.array([0.618033988749895, 0.414213562373095, 0.732050807568877]),
        1,
    )

    # Check if we need to hold plots
    isholdplot = plt.isinteractive()

    for i in range(length):
        if "vol" not in cfg[i] and "node" not in cfg[i] and "shapes" not in cfg[i]:
            raise ValueError("cfg.vol or cfg.node or cfg.shapes is missing")

        if i > 0:
            plt.figure()

        voxelsize = 1
        if "unitinmm" in cfg[i]:
            voxelsize = cfg[i]["unitinmm"]

        offset = 1
        if ("issrcfrom0" in cfg[i] and cfg[i]["issrcfrom0"] == 1) or "node" in cfg[i]:
            offset = 0

        hseg = []
        hbbx = []

        if "vol" in cfg[i] and "node" not in cfg[i]:
            # Render mcxlab voxelated domain
            vol = np.array(cfg[i]["vol"])
            dim = vol.shape

            if vol.ndim == 4:  # For spatially varying medium
                dim = dim[1:]
                cfg[i]["vol"] = np.ones(dim)
                vol = np.ones(dim)

            # Create lattice grid for bounding box
            bbxno, bbxfc, *_ = latticegrid(
                np.arange(0, dim[0] + 1, dim[0]),
                np.arange(0, dim[1] + 1, dim[1]),
                np.arange(0, dim[2] + 1, dim[2]),
            )
            hbbx = plotmesh(
                (bbxno + offset) * voxelsize, bbxfc, facecolor=(1, 1, 1, 0), **kwargs
            )

            # Get unique values (excluding 0)
            val = np.unique(vol.flatten())
            val = val[val != 0]

            # Create padded volume for surface extraction
            padvol = np.zeros(np.array(dim) + 2)
            padvol[1:-1, 1:-1, 1:-1] = vol

            if len(val) > 1:
                hseg = [None] * len(val)

                for id_idx, id_val in enumerate(val):
                    no, fc = binsurface((padvol == id_val).astype(np.int8), 0.5)

                    hseg[id_idx] = plotmesh(
                        (no - 1) * voxelsize,
                        fc,
                        alpha=0.3,
                        linewidth=0,
                        facecolor=surfcolors[int(id_val), :],
                        **kwargs
                    )

        elif "node" in cfg[i] and "elem" in cfg[i]:
            # Render mmclab mesh domain
            elemtype = np.ones(cfg[i]["elem"].shape[0])
            if "elemprop" in cfg[i]:
                elemtype = np.array(cfg[i]["elemprop"])
            else:
                if cfg[i]["elem"].shape[1] > 4:
                    elemtype = cfg[i]["elem"][:, 4]

            etypes = np.unique(elemtype)
            no = np.array(cfg[i]["node"]) * voxelsize
            hseg = [None] * len(etypes)

            for id_idx, id_val in enumerate(etypes):
                elem_mask = elemtype == id_val
                elem_subset = cfg[i]["elem"][elem_mask, :]

                hseg[id_idx] = plotmesh(
                    no,
                    [],
                    elem_subset,
                    alpha=0.3,
                    linewidth=0,
                    facecolor=surfcolors[int(id_val), :],
                    **kwargs
                )

        # Handle shapes if present
        if "shapes" in cfg[i]:
            if "vol" in cfg[i]:
                dim = np.array(cfg[i]["vol"]).shape
            else:
                dim = [60, 60, 60]

            hseg = plotshapes(cfg[i]["shapes"], dim, offset, hseg, voxelsize, **kwargs)

        # Rendering source position and direction
        if "srcpos" not in cfg[i] or "srcdir" not in cfg[i]:
            raise ValueError("cfg.srcpos or cfg.srcdir is missing")

        # one row of srcpos per source; a single srcdir/srcparam1/srcparam2 row is shared
        srcpos = (
            np.atleast_2d(np.array(cfg[i]["srcpos"], dtype=float))[:, :3] * voxelsize
        )
        nsrc = srcpos.shape[0]
        srcdir = np.atleast_2d(np.array(cfg[i]["srcdir"], dtype=float))
        srcdir = srcdir[np.minimum(np.arange(nsrc), srcdir.shape[0] - 1), :3]
        hsrc = plotmesh(srcpos, "r*")  # plotmesh takes one node per row

        srcvec = srcdir * 10 * voxelsize
        headsize = 1e2

        # Use matplotlib's quiver for 3D arrow
        ax = (
            plt.gca()
        )  # the 3-D axes created by plotmesh; gca() takes no arguments in matplotlib>=3.6
        hdir = ax.quiver(
            srcpos[:, 0],
            srcpos[:, 1],
            srcpos[:, 2],
            srcvec[:, 0],
            srcvec[:, 1],
            srcvec[:, 2],
            linewidth=3,
            color="r",
            arrow_length_ratio=headsize / 100,
            **kwargs
        )

        def srcrow(name, isrc):
            row = np.atleast_2d(np.array(cfg[i][name], dtype=float))
            return row[min(isrc, row.shape[0] - 1)]

        # Rendering area-source aperture, one per source
        hsrcarea = []
        srctype = cfg[i].get("srctype", "")
        for isrc in range(nsrc):
            sp = srcpos[isrc]
            sd = srcdir[isrc]
            if srctype in ["disk", "gaussian", "zgaussian", "ring"]:
                if "srcparam1" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 is missing")
                p1 = srcrow("srcparam1", isrc)

                ncyl, fcyl = meshacylinder(sp, sp + sd * 1e-5, p1[0] * voxelsize, 0, 0)
                hsrcarea.append(plotmesh(ncyl, fcyl[-1], facecolor="r", linewidth=0))

                if len(p1) > 1 and p1[1] > 0:
                    ncyl, fcyl = meshacylinder(
                        sp, sp + sd * 1e-5, p1[1] * voxelsize, 0, 0
                    )
                    hsrcarea.append(
                        plotmesh(ncyl, fcyl[-1], facecolor="k", linewidth=0)
                    )

            elif srctype in [
                "planar",
                "pattern",
                "fourier",
                "fourierx",
                "fourierx2d",
                "pencilarray",
            ]:
                if "srcparam1" not in cfg[i] or "srcparam2" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 or cfg.srcparam2 is missing")
                p1 = srcrow("srcparam1", isrc)[:3] * voxelsize

                if srctype in ["fourierx", "fourierx2d"]:
                    vec2 = np.cross(sd, p1)
                else:
                    vec2 = srcrow("srcparam2", isrc)[:3] * voxelsize

                nrec = np.array([[0, 0, 0], p1, p1 + vec2, vec2]) + np.tile(sp, (4, 1))
                hsrcarea.append(plotmesh(nrec, [[0, 1, 2, 3, 0]]))

            elif srctype == "pattern3d":
                dim = srcrow("srcparam1", isrc)[:3]
                bbxno, bbxfc, *_ = latticegrid(
                    np.arange(0, dim[0] + 1, dim[0]),
                    np.arange(0, dim[1] + 1, dim[1]),
                    np.arange(0, dim[2] + 1, dim[2]),
                )
                srcpos_tiled = np.tile(sp / voxelsize, (bbxno.shape[0], 1))
                hbbx = plotmesh(
                    ((bbxno + srcpos_tiled) + offset) * voxelsize,
                    bbxfc,
                    facecolor="y",
                    alpha=0.3,
                )

            elif srctype in ["slit", "line"]:
                if "srcparam1" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 is missing")

                # from the source position along srcparam1, as in mcxpreview.m
                line_points = np.array(
                    [sp, sp + srcrow("srcparam1", isrc)[:3] * voxelsize]
                )
                hsrcarea.append(plotmesh(line_points, [[0, 1]], linewidth=3, color="r"))

        # Rendering detectors
        hdet = []
        hdetdir = None
        if "detpos" in cfg[i]:
            detpos_array = np.array(cfg[i]["detpos"])
            hdet = [None] * detpos_array.shape[0]
            detpos = detpos_array[:, :3] * voxelsize

            if detpos_array.shape[1] == 4:
                radii = detpos_array[:, 3] * voxelsize
            else:
                radii = np.ones(detpos.shape[0])

            for id_det in range(detpos.shape[0]):
                # Create sphere using matplotlib
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = radii[id_det] * np.outer(np.cos(u), np.sin(v)) + detpos[id_det, 0]
                y = radii[id_det] * np.outer(np.sin(u), np.sin(v)) + detpos[id_det, 1]
                z = (
                    radii[id_det] * np.outer(np.ones(np.size(u)), np.cos(v))
                    + detpos[id_det, 2]
                )

                hdet[id_det] = ax.plot_surface(
                    x, y, z, alpha=0.3, color="g", linewidth=0
                )

            # detector directions, which the adjoint modes take from cfg.detdir
            if "detdir" in cfg[i] and np.size(cfg[i]["detdir"]) > 0:
                detvec = (
                    np.atleast_2d(np.array(cfg[i]["detdir"], dtype=float))[:, :3]
                    * 10
                    * voxelsize
                )
                hdetdir = ax.quiver(
                    detpos[:, 0],
                    detpos[:, 1],
                    detpos[:, 2],
                    detvec[:, 0],
                    detvec[:, 1],
                    detvec[:, 2],
                    linewidth=3,
                    color="g",
                    arrow_length_ratio=headsize / 100,
                    **kwargs
                )

        # Combining all handles
        hs[i] = {
            "bbx": hbbx,
            "seg": hseg,
            "src": hsrc,
            "srcarrow": hdir,
            "srcarea": hsrcarea,
            "det": hdet,
            "detarrow": hdetdir,
        }

    return hs


def plotshapes(
    jsonshape, gridsize=None, offset=None, hseg=None, voxelsize=None, **kwargs
):
    """Draw the objects of an MCX JSON shape description.

    Supported objects are ``Grid``, ``Box``, ``Subgrid``, ``Sphere``, ``Cylinder``,
    ``Origin``, ``XSlabs``/``YSlabs``/``ZSlabs`` and ``XLayers``/``YLayers``/``ZLayers``;
    each tag is drawn in a fixed color.

    Parameters
    ----------
    jsonshape : str
        The JSON string (or file name) with a root object ``"Shapes"``, as in
        ``cfg['shapes']``.
    gridsize : list of int, optional
        The domain size, ``cfg['vol'].shape``; a ``Grid`` object overrides it. Default
        ``[60, 60, 60]``.
    offset : int, optional
        The index of the first voxel: ``1 - cfg['issrcfrom0']``, default 1.
    hseg : list, optional
        Existing handles; a ``Grid`` object removes them before drawing.
    voxelsize : float, optional
        The voxel size in mm (``cfg['unitinmm']``), default 1.
    **kwargs
        Passed to the matplotlib/iso2mesh plotting calls.

    Returns
    -------
    hseg : list
        The plot handles of the drawn objects.

    Raises
    ------
    ImportError
        If ``iso2mesh`` or ``matplotlib`` is not installed.

    Examples
    --------
    >>> pmcx.plotshapes('{"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,200]}},'
    ...                 '{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]}]}')
    """

    try:
        import matplotlib.pyplot as plt
        from iso2mesh import plotmesh, latticegrid, loadjson, rotatevec3d
    except ImportError:
        raise ImportError("iso2mesh module is required")

    if hseg is None:
        hseg = []

    # Parse JSON shapes: the text itself, as cfg['shapes'] holds it, or a file name
    if isinstance(jsonshape, str) and jsonshape.lstrip().startswith("{"):
        import json

        shapes = json.loads(jsonshape)
    else:
        shapes = loadjson(jsonshape)

    if offset is None:
        offset = 1
        if gridsize is None:
            gridsize = [60, 60, 60]

    orig = np.array([0, 0, 0]) + offset
    if voxelsize is None:
        voxelsize = 1

    # a fixed color for each tag, as in preview(): numpy's random state is left untouched
    surfcolors = np.mod(
        np.arange(1, 1025)[:, None]
        * np.array([0.618033988749895, 0.414213562373095, 0.732050807568877]),
        1,
    )

    if "Shapes" in shapes:
        shapes_list = shapes["Shapes"]
        if not isinstance(shapes_list, list):
            shapes_list = [shapes_list]

        for j, shp in enumerate(shapes_list):
            sname = list(shp.keys())[0]  # Get first key name
            tag = 1

            if sname in ["Grid", "Box", "Subgrid"]:
                if sname == "Grid" and hseg:
                    # Delete existing handles (matplotlib equivalent)
                    for handle in hseg:
                        if hasattr(handle, "remove"):
                            handle.remove()
                    hseg = []

                obj = shp[sname]
                if "Tag" in obj:
                    tag = obj["Tag"]

                gridsize = obj["Size"]

                if "O" in obj:
                    no, fc, *_ = latticegrid(
                        [0, obj["Size"][0]] + obj["O"][0],
                        [0, obj["Size"][1]] + obj["O"][1],
                        [0, obj["Size"][2]] + obj["O"][2],
                    )
                else:
                    no, fc, *_ = latticegrid(
                        [0, obj["Size"][0]], [0, obj["Size"][1]], [0, obj["Size"][2]]
                    )

                handle = plotmesh(
                    no * voxelsize,
                    fc,
                    alpha=0.3,
                    linestyle="-",
                    facecolor=(1, 1, 1, 0),
                    **kwargs
                )
                hseg.append(handle)

            elif sname == "Sphere":
                obj = shp[sname]
                if "Tag" in obj:
                    tag = obj["Tag"]

                # Create sphere using matplotlib
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                sx = np.outer(np.cos(u), np.sin(v))
                sy = np.outer(np.sin(u), np.sin(v))
                sz = np.outer(np.ones(np.size(u)), np.cos(v))

                ax = (
                    plt.gca()
                )  # the 3-D axes created by plotmesh; gca() takes no arguments in matplotlib>=3.6
                handle = ax.plot_surface(
                    voxelsize * (sx * obj["R"] + (obj["O"][0] + orig[0])),
                    voxelsize * (sy * obj["R"] + (obj["O"][1] + orig[1])),
                    voxelsize * (sz * obj["R"] + (obj["O"][2] + orig[2])),
                    alpha=0.3,
                    color=surfcolors[tag, :],
                    linewidth=0,
                )
                hseg.append(handle)

            elif sname == "Cylinder":
                obj = shp[sname]
                if "Tag" in obj:
                    tag = obj["Tag"]

                c0 = np.array(obj["C0"])
                c1 = np.array(obj["C1"])
                length = np.linalg.norm(c0 - c1)

                # Create cylinder
                theta = np.linspace(0, 2 * np.pi, 50)
                z = np.array([0, length])
                theta, z = np.meshgrid(theta, z)
                sx = obj["R"] * np.cos(theta)
                sy = obj["R"] * np.sin(theta)
                sz = z

                # Flatten for rotation
                points = np.column_stack([sx.flatten(), sy.flatten(), sz.flatten()])

                # Rotate points
                no = rotatevec3d(points, c1 - c0)

                # Reshape back
                sx = no[:, 0].reshape(sx.shape)
                sy = no[:, 1].reshape(sy.shape)
                sz = no[:, 2].reshape(sz.shape)

                ax = (
                    plt.gca()
                )  # the 3-D axes created by plotmesh; gca() takes no arguments in matplotlib>=3.6
                handle = ax.plot_surface(
                    voxelsize * (sx + (c0[0] + orig[0])),
                    voxelsize * (sy + (c0[1] + orig[1])),
                    voxelsize * (sz + (c0[2] + orig[2])),
                    alpha=0.3,
                    color=surfcolors[tag, :],
                    linewidth=0,
                )
                hseg.append(handle)

            elif sname == "Origin":
                orig = voxelsize * np.array(shp[sname])
                ax = (
                    plt.gca()
                )  # the 3-D axes created by plotmesh; gca() takes no arguments in matplotlib>=3.6
                handle = ax.scatter(orig[0], orig[1], orig[2], c="m", marker="*", s=100)
                hseg.append(handle)

            elif sname in ["XSlabs", "YSlabs", "ZSlabs"]:
                obj = shp[sname]
                if "Tag" in obj:
                    tag = obj["Tag"]

                bounds = np.array(obj["Bounds"])
                for k in range(bounds.shape[0]):
                    if sname == "XSlabs":
                        no, fc, *_ = latticegrid(
                            [bounds[k, 0], bounds[k, 1]] + orig[0],
                            [0, gridsize[1]],
                            [0, gridsize[2]],
                        )
                    elif sname == "YSlabs":
                        no, fc, *_ = latticegrid(
                            [0, gridsize[0]],
                            [bounds[k, 0], bounds[k, 1]] + orig[1],
                            [0, gridsize[2]],
                        )
                    elif sname == "ZSlabs":
                        no, fc, *_ = latticegrid(
                            [0, gridsize[0]],
                            [0, gridsize[1]],
                            [bounds[k, 0], bounds[k, 1]] + orig[2],
                        )

                    handle = plotmesh(
                        voxelsize * no,
                        fc,
                        alpha=0.3,
                        linewidth=0,
                        facecolor=surfcolors[tag, :],
                        **kwargs
                    )
                    hseg.append(handle)

            elif sname in ["XLayers", "YLayers", "ZLayers"]:
                obj = shp[sname]

                # Convert to list of lists if needed
                if not isinstance(obj[0], list):
                    obj = [list(row) for row in obj]

                for k, layer in enumerate(obj):
                    tag = 1
                    if len(layer) >= 3:
                        tag = layer[2]

                    if sname == "XLayers":
                        no, fc, *_ = latticegrid(
                            [layer[0] - 1, layer[1]] + orig[0] - 1,
                            [0, gridsize[1]],
                            [0, gridsize[2]],
                        )
                    elif sname == "YLayers":
                        no, fc, *_ = latticegrid(
                            [0, gridsize[0]],
                            [layer[0] - 1, layer[1]] + orig[1] - 1,
                            [0, gridsize[2]],
                        )
                    elif sname == "ZLayers":
                        no, fc, *_ = latticegrid(
                            [0, gridsize[0]],
                            [0, gridsize[1]],
                            [layer[0] - 1, layer[1]] + orig[2] - 1,
                        )

                    handle = plotmesh(
                        voxelsize * no,
                        fc,
                        alpha=0.3,
                        linewidth=0,
                        facecolor=surfcolors[tag, :],
                        **kwargs
                    )
                    hseg.append(handle)

            else:
                raise ValueError("unsupported shape constructs")

    return hseg


def plotphotons(traj, *varargin, **kwargs):
    """Draw the photon trajectories recorded by a simulation.

    Parameters
    ----------
    traj : dict or ndarray
        The trajectories, the ``traj`` output of :func:`pmcx.mcxlab` (``cfg['debuglevel']``
        containing ``'M'``): a dict with ``id`` (photon index), ``pos`` (``(n, 3)``
        positions) and optionally ``weight``, or with only ``data``
        (``[id, x, y, z, weight, reserved]`` in rows); or an ``(n, 6)`` array of such
        records.
    *varargin
        ``'noplot'`` returns the sorted trajectories without drawing them.
    **kwargs
        Passed to ``iso2mesh.plotmesh``; ``color`` defaults to ``'b'``.

    Returns
    -------
    sorted : dict
        The records sorted by photon: ``id``, and ``pos``, the scattering sites in order.
    hg : object or None
        The handle of the drawn lines, or None with ``'noplot'``.

    Raises
    ------
    ImportError
        If ``iso2mesh`` is not installed.
    """

    try:
        from iso2mesh import plotmesh
    except ImportError:
        raise ImportError("iso2mesh module is required")

    # Handle different input formats
    if isinstance(traj, dict) and "id" not in traj and "data" in traj:
        # Convert from data format
        data = np.array(traj["data"])
        traj = {
            "id": np.array(data[0, :], dtype=np.uint32),
            "pos": data[1:4, :].T,  # Transpose to match expected shape
            "weight": data[4, :],
            "data": data,
        }
    elif (
        not isinstance(traj, dict)
        and isinstance(traj, np.ndarray)
        and traj.shape[1] == 6
    ):
        # Convert from array format
        traj = {
            "id": traj[:, 0].astype(np.uint32),
            "pos": traj[:, 1:4],
            "weight": traj[:, 4],
        }

    # Sort trajectories by photon ID
    newid, idx = np.unique(traj["id"], return_inverse=True)
    sorted_idx = np.argsort(traj["id"])
    newid = traj["id"][sorted_idx]

    # Find line end points (where photon ID changes)
    lineend = np.diff(newid) > 0
    newidx = np.cumsum(np.concatenate([[0], lineend]) + 1)

    # Create position array with NaN separators for line breaks
    newpos = np.full((len(sorted_idx) + len(lineend), 4), np.nan)
    newpos[newidx - 1, :3] = traj["pos"][sorted_idx, :]

    # Add weight information if available
    if "data" in traj:
        newpos[newidx - 1, 3] = traj["data"][4, sorted_idx]
    elif "weight" in traj:
        newpos[newidx - 1, 3] = traj["weight"][sorted_idx]
    else:
        newpos[:, 3] = 1.0  # Default weight

    # Check if plotting is disabled
    noplot = (
        len(varargin) == 1 and isinstance(varargin[0], str) and varargin[0] == "noplot"
    )

    if not noplot:
        # Determine edge transparency
        edgealpha = 0.25  # Reduced opacity for better visualization

        # Create line segments for each photon trajectory
        lines = []
        colors = []

        # Parse additional arguments
        color = kwargs.get("color", "b")
        marker = kwargs.get("marker", None)

        # Group positions by photon ID for line drawing
        current_photon = []
        current_weight = []

        for i, pos in enumerate(newpos):
            if not np.isnan(pos[0]):  # Valid position
                current_photon.append(pos[:3])
                current_weight.append(pos[3])
            else:  # NaN separator - end of photon trajectory
                if len(current_photon) > 1:
                    lines.append(np.array(current_photon))
                    # Use weight for coloring if available
                    if len(current_weight) > 0:
                        colors.extend(current_weight)
                current_photon = []
                current_weight = []

        # Add the last trajectory if it exists
        if len(current_photon) > 1:
            lines.append(np.array(current_photon))
            if len(current_weight) > 0:
                colors.extend(current_weight)

        # Plot trajectories using Line3DCollection for efficiency
        if lines:
            # Combine all line segments
            all_points = []
            all_lines = []
            point_idx = 0

            for line in lines:
                line_indices = []
                for point in line:
                    all_points.append(point)
                    line_indices.append(point_idx)
                    point_idx += 1
                if len(line_indices) > 1:
                    # Create line segments
                    for i in range(len(line_indices) - 1):
                        all_lines.append([line_indices[i], line_indices[i + 1]])

            hg = plotmesh(np.array(all_points), all_lines, color=color, **kwargs)

        else:
            hg = None

    else:
        hg = None

    # Create output structure
    output_sorted = {"id": newid, "pos": traj["pos"][sorted_idx, :]}

    return output_sorted, hg
