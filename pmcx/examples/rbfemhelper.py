"""
Redbird (FEM diffusion) helpers shared by the demo_validate_mcx_*.py examples, which
compare MCX against the diffusion solutions of redbirdpy (https://github.com/fangq/redbirdpy)

  slabmesh():   the [0,60]x[0,60]x[0,30] mm tetrahedral slab used by the demos
  jacobians():  nodal mua and diffusion-coefficient Jacobians, as rbjac.m in redbird-m
  cut2grid():   cut a nodal quantity at y=const and interpolate it to the MCX voxel centers
"""

import numpy as np
import iso2mesh as i2m
import redbirdpy as rb
from scipy.interpolate import griddata


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def slabmesh(srcpos, detpos, prop, omega):
    """create and prepare the slab mesh (domain [0,60]x[0,60]x[0,30] mm); returns cfg, sd"""
    node, elem = i2m.meshgrid6(np.arange(0, 61), np.arange(0, 61), np.arange(0, 31))
    elem[:, :4] = _first(i2m.meshreorient(node[:, :3], elem[:, :4]))
    cfg = {"node": node, "elem": elem, "face": _first(i2m.volface(elem))}
    cfg["seg"] = np.ones(elem.shape[0], dtype=int)
    cfg["srcpos"] = np.atleast_2d(srcpos)
    cfg["srcdir"] = np.array([[0, 0, 1]])  # pointing into medium (+z)
    if detpos is not None:
        cfg["detpos"] = np.atleast_2d(detpos)
        cfg["detdir"] = np.array([[0, 0, 1]])  # pointing into medium (+z)
    cfg["prop"] = np.asarray(prop, dtype=float)
    cfg["omega"] = omega  # non-zero omega: frequency-domain
    cfg = _first(rb.meshprep(cfg))
    return cfg, (rb.sdmap(cfg) if detpos is not None else None)


# node pairs (0-based) and deldotdel columns of the 12 off-diagonal terms of a tetrahedron, as rbjac.m
_IDX1 = np.array([1, 1, 1, 2, 2, 3, 2, 3, 4, 3, 4, 4]) - 1
_IDX2 = np.array([2, 3, 4, 3, 4, 4, 1, 1, 1, 2, 2, 3]) - 1
_IDX3 = np.array([2, 3, 4, 6, 7, 9, 2, 3, 4, 6, 7, 9]) - 1
_DIAG = np.array([1, 5, 8, 10]) - 1


def jacobians(sd, phi, cfg):
    """
    Jmua_n, Jd_n = jacobians(sd, phi, cfg)

    nodal mua Jacobian and diffusion-coefficient (D) Jacobian of each active source-detector
    pair, [npair, nnode] - a vectorized port of the Jmua_n and Jd_n outputs of rbjac.m
    """
    elem = np.asarray(cfg["elem"])[:, :4].astype(int) - 1
    ddd = np.asarray(cfg["deldotdel"])
    evol = np.asarray(cfg["evol"])
    nn = phi.shape[0]
    pairs = sd[sd[:, 2] == 1, :2].astype(int) if sd.shape[1] >= 3 else sd[:, :2].astype(int)
    Jmua = np.zeros((len(pairs), nn), dtype=phi.dtype)
    Jd = np.zeros((len(pairs), nn), dtype=phi.dtype)
    for k, (s, d) in enumerate(pairs):
        ps, pd = phi[elem, s], phi[elem, d]  # [nelem, 4]
        diag = ps * pd
        offd = ps[:, _IDX1] * pd[:, _IDX2]
        jmua_e = -(diag.sum(axis=1) + offd.sum(axis=1) * 0.5) * (0.1 * evol)
        jd_e = -((diag * ddd[:, _DIAG]).sum(axis=1) + (offd * ddd[:, _IDX3]).sum(axis=1))
        for j in range(4):
            Jmua[k] += np.bincount(elem[:, j], weights=jmua_e.real, minlength=nn) + 1j * np.bincount(
                elem[:, j], weights=jmua_e.imag, minlength=nn
            ) if np.iscomplexobj(phi) else np.bincount(elem[:, j], weights=jmua_e, minlength=nn)
            Jd[k] += np.bincount(elem[:, j], weights=jd_e.real, minlength=nn) + 1j * np.bincount(
                elem[:, j], weights=jd_e.imag, minlength=nn
            ) if np.iscomplexobj(phi) else np.bincount(elem[:, j], weights=jd_e, minlength=nn)
    return Jmua * 0.25, Jd * 0.25


def cut2grid(cfg, value, y, xgrid, zgrid):
    """cut a nodal (real or complex) quantity at the plane y=const and interpolate it to [Nz, Nx]"""
    xi, zi = np.meshgrid(xgrid, zgrid)

    def onepart(v):
        cutpos, cutval = i2m.qmeshcut(cfg["elem"][:, :4], cfg["node"], v, "y=%g" % y)[:2]
        cutpos = np.asarray(cutpos)
        return griddata((cutpos[:, 0], cutpos[:, 2]), np.ravel(cutval), (xi, zi))

    value = np.asarray(value)
    if np.iscomplexobj(value):
        return onepart(value.real) + 1j * onepart(value.imag)
    return onepart(value)
