"""
Analytical diffusion solution of a spherical inclusion in an infinite, semi-infinite or
slab medium - a Python port of the sphere-diffusion toolbox in mmc/matlab (sphdiffusion*.m),
used by demo_validation_heterogeneous.py.

The solution is a series of spherical Bessel functions and spherical harmonics up to the
order cfg['maxl']; positions are given in a spherical coordinate system centered at the
sphere, with the source at cfg['src'] = [r, theta, phi].

Author: Qianqian Fang <q.fang at neu.edu>
"""

import copy
import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv, factorial

DEFAULTS = dict(v=299792458000, a=10, omua=0.002, omusp=0.990, imua=0.050, imusp=0.500, src=[30, np.pi, 0], maxl=20, omega=0)


def _init(cfg):
    cfg = copy.deepcopy(DEFAULTS if cfg is None else cfg)
    cfg["src"] = list(cfg["src"])
    cfg["Din"] = cfg["v"] / (3 * cfg["imusp"])
    cfg["Dout"] = cfg["v"] / (3 * cfg["omusp"])
    cfg["kin"] = np.sqrt(complex(-cfg["v"] * cfg["imua"], cfg["omega"]) / cfg["Din"])
    cfg["kout"] = np.sqrt(complex(-cfg["v"] * cfg["omua"], cfg["omega"]) / cfg["Dout"])
    return cfg


def _cart2sph(x, y, z):
    """MATLAB cart2sph: azimuth, elevation, radius"""
    return np.arctan2(y, x), np.arctan2(z, np.hypot(x, y)), np.sqrt(x * x + y * y + z * z)


def _sph2cart(az, el, r):
    return r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)


def spharmonic(l, m, theta, phi):
    """spherical harmonic Y_l^m(theta, phi), same normalization as mmc/matlab/spharmonic.m"""
    coeff = 1.0
    oldm = m
    if m < 0:
        coeff = (-1) ** m * factorial(l - m) / factorial(l + m)
        m = -m
    Lmn = lpmv(m, l, np.cos(theta))  # includes the Condon-Shortley phase, as MATLAB legendre()
    m = oldm
    return coeff * np.sqrt((2 * l + 1) * factorial(l - m) / (factorial(l + m) * 4 * np.pi)) * Lmn * np.exp(1j * m * phi)


def spbesselh(n, z):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def spbesselhprime(n, z):
    return spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)


def _coeff_common(l, m, cfg):
    return (
        -1j
        * cfg["v"]
        * cfg["kout"]
        / cfg["Dout"]
        * spbesselh(l, cfg["kout"] * cfg["src"][0])
        * np.conj(spharmonic(l, m, cfg["src"][1], cfg["src"][2]))
    )


def _onaxis(m, cfg):
    """a source on the z-axis only excites the m=0 terms"""
    return (cfg["src"][1] == np.pi or cfg["src"][1] == 0) and m != 0


def sphdiffAcoeff(m, l, cfg):
    if _onaxis(m, cfg):
        return 0
    x, y = cfg["kout"] * cfg["a"], cfg["kin"] * cfg["a"]
    Dout, Din = cfg["Dout"], cfg["Din"]
    jx, jy = spherical_jn(l, x), spherical_jn(l, y)
    jxp, jyp = spherical_jn(l, x, derivative=True), spherical_jn(l, y, derivative=True)
    return (
        _coeff_common(l, m, cfg)
        * (Dout * x * jxp * jy - Din * y * jx * jyp)
        / (Dout * x * spbesselhprime(l, x) * jy - Din * y * spbesselh(l, x) * jyp)
    )


def sphdiffBcoeff(m, l, cfg):
    return 1j * sphdiffAcoeff(m, l, cfg)


def sphdiffCcoeff(m, l, cfg):
    if _onaxis(m, cfg):
        return 0
    x, y = cfg["kout"] * cfg["a"], cfg["kin"] * cfg["a"]
    Dout, Din = cfg["Dout"], cfg["Din"]
    jx, jy = spherical_jn(l, x), spherical_jn(l, y)
    jxp, jyp = spherical_jn(l, x, derivative=True), spherical_jn(l, y, derivative=True)
    return (
        _coeff_common(l, m, cfg)
        * (Dout * x * spbesselh(l, x) * jxp - Dout * x * spbesselhprime(l, x) * jx)
        / (Dout * x * spbesselhprime(l, x) * jy - Din * y * spbesselh(l, x) * jyp)
    )


def sphdiffincident(r, theta, phi, cfg):
    xs, ys, zs = _sph2cart(cfg["src"][2], np.pi / 2 - cfg["src"][1], cfg["src"][0])
    x, y, z = _sph2cart(phi, np.pi / 2 - theta, r)
    dist = np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)
    return cfg["v"] / (4 * np.pi * cfg["Dout"] * dist) * np.exp(1j * cfg["kout"] * dist)


def sphdiffscatter(r, theta, phi, cfg):
    res = np.zeros(np.shape(r), dtype=complex)
    for l in range(cfg["maxl"] + 1):
        jl, yl = spherical_jn(l, cfg["kout"] * r), spherical_yn(l, cfg["kout"] * r)
        for m in range(-l, l + 1):
            res += (sphdiffAcoeff(m, l, cfg) * jl + sphdiffBcoeff(m, l, cfg) * yl) * spharmonic(l, m, theta, phi)
    return res


def sphdiffexterior(r, theta, phi, cfg):
    return sphdiffincident(r, theta, phi, cfg) + sphdiffscatter(r, theta, phi, cfg)


def sphdiffinterior(r, theta, phi, cfg):
    res = np.zeros(np.shape(r), dtype=complex)
    for l in range(cfg["maxl"] + 1):
        jl = spherical_jn(l, cfg["kin"] * r)
        for m in range(-l, l + 1):
            res += sphdiffCcoeff(m, l, cfg) * jl * spharmonic(l, m, theta, phi)
    return res


def _grid(xrange, yrange, zrange):
    xi, yi, zi = np.meshgrid(np.atleast_1d(xrange), np.atleast_1d(yrange), np.atleast_1d(zrange))  # MATLAB meshgrid
    P, T, R = _cart2sph(xi.ravel(), yi.ravel(), zi.ravel())
    return xi, yi, zi, P, np.pi / 2 - T, R  # matlab's theta and phi are defined differently


def sphdiffusioninfinite(xrange, yrange, zrange, cfg=None):
    cfg = _init(cfg)
    xi, yi, zi, P, T, R = _grid(xrange, yrange, zrange)
    res = np.zeros(R.shape, dtype=complex)
    out = R > cfg["a"]
    res[out] = sphdiffexterior(R[out], T[out], P[out], cfg)
    res[~out] = sphdiffinterior(R[~out], T[~out], P[~out], cfg)
    return np.squeeze(res.reshape(xi.shape)), np.squeeze(xi), np.squeeze(yi), np.squeeze(zi)


def sphdiffusionscatteronly(xrange, yrange, zrange, cfg=None):
    cfg = _init(cfg)
    xi, yi, zi, P, T, R = _grid(xrange, yrange, zrange)
    res = np.zeros(R.shape, dtype=complex)
    out = R > cfg["a"]
    res[out] = sphdiffscatter(R[out], T[out], P[out], cfg)
    return np.squeeze(res.reshape(xi.shape)), np.squeeze(xi), np.squeeze(yi), np.squeeze(zi)


def sphdiffusionsemi(Reff, xrange, yrange, zrange, cfg=None):
    cfg = _init(cfg)
    D = 1 / (3 * (cfg["omua"] + cfg["omusp"]))
    zb = (1 + Reff) / (1 - Reff) * 2 * D
    z0 = 1 / (cfg["omusp"] + cfg["omua"])
    src0 = list(cfg["src"])

    cfg["src"] = [src0[0] - z0, src0[1], src0[2]]
    res, xi, yi, zi = sphdiffusioninfinite(xrange, yrange, zrange, cfg)  # S1,O1
    cfg["src"] = [src0[0] + z0 + 2 * zb, np.pi, src0[2]]
    res = res - sphdiffusioninfinite(xrange, yrange, zrange, cfg)[0]  # S2,O1

    out = (np.sqrt(xi**2 + yi**2 + zi**2) > cfg["a"]).ravel()
    res = res.ravel()
    zshift = np.asarray(zrange) + 2 * (src0[0] + zb)
    cfg["src"] = [src0[0] + z0 + 2 * zb, 0, src0[2]]
    res[out] += np.ravel(sphdiffusionscatteronly(xrange, yrange, zshift, cfg)[0])[out]  # S1,O2
    cfg["src"] = [src0[0] - z0, 0, src0[2]]
    res[out] -= np.ravel(sphdiffusionscatteronly(xrange, yrange, zshift, cfg)[0])[out]  # S2,O2
    return res.reshape(xi.shape), xi, yi, zi


def sphdiffusionslab(Reff1, Reff2, h, xrange, yrange, zrange, cfg=None):
    """the sphere in a slab of thickness h, with effective reflection coeff. Reff1/Reff2 at the two faces"""
    cfg = _init(cfg)
    src0 = list(cfg["src"])
    res, xi, yi, zi = sphdiffusionsemi(Reff1, xrange, yrange, zrange, cfg)

    D = 1 / (3 * (cfg["omua"] + cfg["omusp"]))
    zb = (1 + Reff2) / (1 - Reff2) * 2 * D
    z0 = 1 / (cfg["omusp"] + cfg["omua"])

    cfg["src"] = [2 * h - src0[0] + 2 * zb - z0, np.pi - src0[1], src0[2]]
    res = res - sphdiffusioninfinite(xrange, yrange, zrange, cfg)[0]  # S2,O1
    cfg["src"] = [2 * h - src0[0] + 2 * zb - z0 + 2 * (z0 + zb), np.pi - src0[1], src0[2]]
    res = res + sphdiffusioninfinite(xrange, yrange, zrange, cfg)[0]  # S2,O1

    out = (np.sqrt(xi**2 + yi**2 + zi**2) > cfg["a"]).ravel()
    res = res.ravel()
    zshift = np.asarray(zrange) - 2 * (h - src0[0] + zb)
    for src, sign in [
        ([2 * h - src0[0] - z0, src0[1], src0[2]], 1),  # S1,O2
        ([2 * h - src0[0] + 2 * zb + z0, src0[1], src0[2]], -1),  # S2,O2
        ([src0[0] - z0, np.pi - src0[1], src0[2]], -1),  # S1,O2
        ([src0[0] + 2 * zb + z0, np.pi - src0[1], src0[2]], 1),  # S2,O2
    ]:
        cfg["src"] = src
        res[out] += sign * np.ravel(sphdiffusionscatteronly(xrange, yrange, zshift, cfg)[0])[out]
    return res.reshape(xi.shape), xi, yi, zi
