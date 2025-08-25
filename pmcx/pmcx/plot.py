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


"""Plotting MCX input and output data structures"""

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
    from iso2mesh import plotvolume

    return plotvolume(*args, **kwargs)


def preview(cfg, **kwargs):
    """
    hs = preview(cfg, **kwargs)

    Preview the simulation configuration for both MCXLAB and MMCLAB

    Author: Qianqian Fang <q.fang at neu.edu>
    Converted to Python

    Parameters:
    -----------
    cfg : dict or list of dict
        A dict, or list of dict. Each element of cfg defines
        the parameters associated with a simulation. Please run
        'help mcxlab' or 'help mmclab' to see the details.
        preview supports the cfg input for both mcxlab and mmclab.
    **kwargs : tuple
        Additional arguments passed to plotting functions

    Returns:
    --------
    hs : list
        A list of dict containing handles to the plotted domain elements.

    Dependency:
        This function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)

    This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space

    License: GNU General Public License version 3, please read LICENSE.txt for details
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

    length = len(cfg)
    hs = [None] * length

    # Save random state
    rngstate = np.random.get_state()

    # Set random seed for consistent colors
    randseed = 0x623F9A9E
    np.random.seed(randseed)
    surfcolors = np.random.rand(1024, 3)

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
            bbxno, bbxfc = latticegrid(
                np.arange(0, dim[0] + 1, dim[0]),
                np.arange(0, dim[1] + 1, dim[1]),
                np.arange(0, dim[2] + 1, dim[2]),
            )
            hbbx = plotmesh(
                (bbxno + offset) * voxelsize, bbxfc, facecolor="none", **kwargs
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
                        facealpha=0.3,
                        linestyle="none",
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
                    facealpha=0.3,
                    linestyle="none",
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

        srcpos = np.array(cfg[i]["srcpos"]) * voxelsize
        hsrc = plotmesh(srcpos, "r*")

        srcvec = np.array(cfg[i]["srcdir"]) * 10 * voxelsize
        headsize = 1e2

        # Use matplotlib's quiver for 3D arrow
        ax = plt.gca(projection="3d")
        hdir = ax.quiver(
            srcpos[0],
            srcpos[1],
            srcpos[2],
            srcvec[0],
            srcvec[1],
            srcvec[2],
            linewidth=3,
            color="r",
            arrow_length_ratio=headsize / 100,
            **kwargs
        )

        # Rendering area-source aperture
        hsrcarea = []
        if "srctype" in cfg[i]:
            if cfg[i]["srctype"] in ["disk", "gaussian", "zgaussian", "ring"]:
                if "srcparam1" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 is missing")

                ncyl, fcyl = meshacylinder(
                    srcpos,
                    srcpos + np.array(cfg[i]["srcdir"][:3]) * 1e-5,
                    cfg[i]["srcparam1"][0] * voxelsize,
                    0,
                    0,
                )
                hsrcarea = plotmesh(ncyl, fcyl[-1], facecolor="r", linestyle="none")

                if len(cfg[i]["srcparam1"]) > 1 and cfg[i]["srcparam1"][1] > 0:
                    ncyl, fcyl = meshacylinder(
                        srcpos,
                        srcpos + np.array(cfg[i]["srcdir"][:3]) * 1e-5,
                        cfg[i]["srcparam1"][1] * voxelsize,
                        0,
                        0,
                    )
                    hsrcarea = plotmesh(ncyl, fcyl[-1], facecolor="k", linestyle="none")

            elif cfg[i]["srctype"] in [
                "planar",
                "pattern",
                "fourier",
                "fourierx",
                "fourierx2d",
                "pencilarray",
            ]:
                if "srcparam1" not in cfg[i] or "srcparam2" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 or cfg.srcparam2 is missing")

                if cfg[i]["srctype"] in ["fourierx", "fourierx2d"]:
                    vec2 = np.cross(
                        cfg[i]["srcdir"], np.array(cfg[i]["srcparam1"][:3]) * voxelsize
                    )
                else:
                    vec2 = np.array(cfg[i]["srcparam2"][:3]) * voxelsize

                srcparam1_scaled = np.array(cfg[i]["srcparam1"][:3]) * voxelsize
                nrec = np.array(
                    [[0, 0, 0], srcparam1_scaled, srcparam1_scaled + vec2, vec2]
                )

                # Add source position to all points
                nrec = nrec + np.tile(srcpos, (4, 1))
                hsrcarea = plotmesh(nrec, [[0, 1, 2, 3, 0]])

            elif cfg[i]["srctype"] == "pattern3d":
                dim = cfg[i]["srcparam1"][:3]
                bbxno, bbxfc = latticegrid(
                    np.arange(0, dim[0] + 1, dim[0]),
                    np.arange(0, dim[1] + 1, dim[1]),
                    np.arange(0, dim[2] + 1, dim[2]),
                )
                srcpos_tiled = np.tile(cfg[i]["srcpos"][:3], (bbxno.shape[0], 1))
                hbbx = plotmesh(
                    ((bbxno + srcpos_tiled) + offset) * voxelsize,
                    bbxfc,
                    facecolor="y",
                    facealpha=0.3,
                )

            elif cfg[i]["srctype"] in ["slit", "line"]:
                if "srcparam1" not in cfg[i]:
                    raise ValueError("cfg.srcparam1 is missing")

                line_points = np.array(
                    [srcpos[:3], np.array(cfg[i]["srcparam1"][:3]) * voxelsize]
                )
                hsrcarea = plotmesh(line_points, [[0, 1]], linewidth=3, color="r")

        # Rendering detectors
        hdet = []
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

        # Combining all handles
        hs[i] = {
            "bbx": hbbx,
            "seg": hseg,
            "src": hsrc,
            "srcarrow": hdir,
            "srcarea": hsrcarea,
            "det": hdet,
        }

    # Restore random state
    np.random.set_state(rngstate)

    return hs


def plotshapes(
    jsonshape, gridsize=None, offset=None, hseg=None, voxelsize=None, **kwargs
):
    """
    Format:
        mcxplotshapes(jsonshapestr)
        handles = mcxplotshapes(jsonshapestr)
        handles = mcxplotshapes(jsonshapestr, gridsize, offset, oldhandles, voxelsize, ...)

    Create MCX simulation from built-in benchmarks (similar to "mcx --bench")

    Author: Qianqian Fang <q.fang at neu.edu>
    Converted to Python

    Parameters:
    -----------
    jsonshape : str
        An MCX shape json string with a root object "Shapes"
    gridsize : list, optional
        This should be set to size(cfg.vol), default is [60, 60, 60]
    offset : int, optional
        This should be set to 1-cfg.issrcfrom0, default is 1
    hseg : list, optional
        Existing plot handles
    voxelsize : float, optional
        The voxel size of the grid - usually defined as
        cfg.unitinmm, default is 1
    **kwargs : tuple
        Additional arguments passed to plotting functions

    Returns:
    --------
    hseg : list
        An array of all plot object handles

    Dependency:
        This function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)

    Examples:
    ---------
    mcxplotshapes('{"Shapes":[{"Grid":{"Tag":1,"Size":[60, 60, 200]}},{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]}]}')
    hfig = mcxpreview(mcxcreate('sphshell'))
    hfig2 = mcxpreview(mcxcreate('spherebox'))

    This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space

    License: GNU General Public License version 3, please read LICENSE.txt for details
    """

    try:
        import matplotlib.pyplot as plt
        from iso2mesh import plotmesh, latticegrid, loadjson, rotatevec3d
    except ImportError:
        raise ImportError("iso2mesh module is required")

    if hseg is None:
        hseg = []

    # Parse JSON shapes
    shapes = loadjson(jsonshape)

    if offset is None:
        offset = 1
        if gridsize is None:
            gridsize = [60, 60, 60]

    orig = np.array([0, 0, 0]) + offset
    if voxelsize is None:
        voxelsize = 1

    # Save random state
    rngstate = np.random.get_state()
    randseed = 0x623F9A9E
    np.random.seed(randseed)
    surfcolors = np.random.rand(1024, 3)

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
                    no, fc = latticegrid(
                        [0, obj["Size"][0]] + obj["O"][0],
                        [0, obj["Size"][1]] + obj["O"][1],
                        [0, obj["Size"][2]] + obj["O"][2],
                    )
                else:
                    no, fc = latticegrid(
                        [0, obj["Size"][0]], [0, obj["Size"][1]], [0, obj["Size"][2]]
                    )

                handle = plotmesh(
                    no * voxelsize,
                    fc,
                    facealpha=0.3,
                    linestyle="-",
                    facecolor="none",
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

                ax = plt.gca(projection="3d")
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

                ax = plt.gca(projection="3d")
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
                ax = plt.gca(projection="3d")
                handle = ax.scatter(orig[0], orig[1], orig[2], c="m", marker="*", s=100)
                hseg.append(handle)

            elif sname in ["XSlabs", "YSlabs", "ZSlabs"]:
                obj = shp[sname]
                if "Tag" in obj:
                    tag = obj["Tag"]

                bounds = np.array(obj["Bounds"])
                for k in range(bounds.shape[0]):
                    if sname == "XSlabs":
                        no, fc = latticegrid(
                            [bounds[k, 0], bounds[k, 1]] + orig[0],
                            [0, gridsize[1]],
                            [0, gridsize[2]],
                        )
                    elif sname == "YSlabs":
                        no, fc = latticegrid(
                            [0, gridsize[0]],
                            [bounds[k, 0], bounds[k, 1]] + orig[1],
                            [0, gridsize[2]],
                        )
                    elif sname == "ZSlabs":
                        no, fc = latticegrid(
                            [0, gridsize[0]],
                            [0, gridsize[1]],
                            [bounds[k, 0], bounds[k, 1]] + orig[2],
                        )

                    handle = plotmesh(
                        voxelsize * no,
                        fc,
                        facealpha=0.3,
                        linestyle="none",
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
                        no, fc = latticegrid(
                            [layer[0] - 1, layer[1]] + orig[0] - 1,
                            [0, gridsize[1]],
                            [0, gridsize[2]],
                        )
                    elif sname == "YLayers":
                        no, fc = latticegrid(
                            [0, gridsize[0]],
                            [layer[0] - 1, layer[1]] + orig[1] - 1,
                            [0, gridsize[2]],
                        )
                    elif sname == "ZLayers":
                        no, fc = latticegrid(
                            [0, gridsize[0]],
                            [0, gridsize[1]],
                            [layer[0] - 1, layer[1]] + orig[2] - 1,
                        )

                    handle = plotmesh(
                        voxelsize * no,
                        fc,
                        facealpha=0.3,
                        linestyle="none",
                        facecolor=surfcolors[tag, :],
                        **kwargs
                    )
                    hseg.append(handle)

            else:
                raise ValueError("unsupported shape constructs")

    # Restore random state
    np.random.set_state(rngstate)

    return hseg


def plotphotons(traj, *varargin, **kwargs):
    """
    plotphotons(traj)
        or
    plotphotons(traj, 'color', 'r', 'marker', 'o')
    sorted, linehandle = plotphotons(traj)

    Plot photon trajectories from MCXLAB's output

    Author: Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Parameters:
    -----------
    traj : dict or ndarray
        The 5th output of mcxlab, storing the photon trajectory info
        traj['id']: the photon index being recorded
        traj['pos']: the 3D position of the photon; for each photon, the
                    positions are stored in serial order
        traj['data']: the combined output, in the form of
                     [id, pos, weight, reserved]'
    *varargin : tuple
        Additional arguments for plotting
    **kwargs : dict
        Keyword arguments for plotting

    Returns:
    --------
    sorted : dict
        A dict to store the sorted trajectory info
        sorted['id']: the sorted vector of photon id, starting from 0
        sorted['pos']: the sorted position vector of each photon, only
                      recording the scattering sites.
    linehandle : object
        Handle to the plotted lines (when plotting is enabled)

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see http://mcx.sf.net for details
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
