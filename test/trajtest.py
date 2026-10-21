#!/usr/bin/env python3
"""
trajtest.py - verify MCX/MCX-CL/MMC photon trajectories step by step

Usage:
    trajtest.py TRAJ --case CASE.json [--ref GOLDEN.json] [--tol 1e-4]
                [--wtol 1e-4] [--min-match 1.0] [--save GOLDEN.json] [--solver mcx]

TRAJ is either a trajectory file saved by "-D M" (session_traj.jdt, text
JData, any _ArrayZipType_ supported by the python stdlib) or a solver-neutral
golden file (.json) previously written by --save.

A golden file stores one list of [x, y, z, w] steps per photon, sorted by the
photon id, so the trajectories of different solvers (mcx, mcxcl, mmc) can be
compared against each other with --ref.

Invariant checks (always performed):
    - every step lies inside the box domain defined by Domain.Dim of CASE
    - the first step of each photon is the pencil-beam source position
    - photon weight never increases along a path
    - in a homogeneous, index-matched case, the weight drop between two
      steps equals exp(-mua * step length) (Beer-Lambert per segment)
    - each photon ends on the domain boundary, or with (near) zero weight

Exit code is 0 when all checks pass, 1 otherwise. Only numpy is required.
"""

import argparse
import base64
import bz2
import gzip
import json
import lzma
import sys
import zlib

import numpy as np

_DTYPES = {
    "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64,
    "single": np.float32, "float32": np.float32, "double": np.float64, "float64": np.float64,
}

_UNZIP = {
    "zlib": zlib.decompress, "gzip": gzip.decompress,
    "lzma": lzma.decompress, "bz2": bz2.decompress,
}


def decode_array(obj):
    """Decode a JData annotated array object into a numpy array"""
    if not isinstance(obj, dict) or "_ArrayType_" not in obj:
        return np.asarray(obj)

    dtype = _DTYPES[obj["_ArrayType_"]]
    size = obj["_ArraySize_"]
    size = [size] if isinstance(size, int) else list(size)

    if "_ArrayZipData_" in obj:
        ztype = obj.get("_ArrayZipType_", "zlib")
        if ztype not in _UNZIP:
            raise ValueError("unsupported _ArrayZipType_: " + ztype)
        raw = _UNZIP[ztype](base64.b64decode(obj["_ArrayZipData_"]))
        arr = np.frombuffer(raw, dtype=dtype)
    elif isinstance(obj.get("_ArrayData_"), str):
        arr = np.frombuffer(base64.b64decode(obj["_ArrayData_"]), dtype=dtype)
    else:
        arr = np.asarray(obj["_ArrayData_"], dtype=dtype)

    order = "F" if obj.get("_ArrayOrder_", "r") in ("c", "col", "column") else "C"
    return arr.reshape(size, order=order)


def load_traj(fname):
    """Return a list (indexed by photon order) of Nx4 [x,y,z,w] arrays"""
    with open(fname, "r") as fp:
        data = json.load(fp)

    if "photons" in data:  # solver-neutral golden file
        return [np.asarray(ph, dtype=np.float64).reshape(-1, 4) for ph in data["photons"]]

    traj = data["MCXData"]["Trajectory"]
    pid = decode_array(traj["photonid"]).ravel().astype(np.int64)
    pos = decode_array(traj["p"]).reshape(-1, 3).astype(np.float64)
    w = decode_array(traj["w0"]).ravel().astype(np.float64)

    # a stable sort keeps the recording order of the steps within each photon
    order = np.argsort(pid, kind="stable")
    pid, pos, w = pid[order], pos[order], w[order]
    steps = np.hstack([pos, w[:, None]])
    ids, start = np.unique(pid, return_index=True)
    return [steps[s:e] for s, e in zip(start, list(start[1:]) + [len(pid)])]


def save_golden(fname, photons, solver, casename):
    out = {
        "case": casename,
        "solver": solver,
        "format": "per-photon list of [x,y,z,w] steps, sorted by photon id",
        "photons": [[[float("%.8g" % v) for v in step] for step in ph] for ph in photons],
    }
    with open(fname, "w") as fp:
        json.dump(out, fp, separators=(",", ":"))
        fp.write("\n")


def check_invariants(photons, case, tol, wtol):
    """Return a list of human-readable failure messages"""
    errors = []
    dom = case["Domain"]
    unit = dom.get("LengthUnit", 1.0)
    dim = np.asarray(dom["Dim"], dtype=np.float64) * unit
    media = dom["Media"]
    src = case["Optode"]["Source"]
    srcpos = np.asarray(src["Pos"][:3], dtype=np.float64) * unit
    ismatched = not case["Session"].get("DoMismatch", 0)
    mua = media[1]["mua"] if (len(media) == 2 and ismatched) else None

    for i, ph in enumerate(photons):
        p, w = ph[:, :3], ph[:, 3]

        if np.any(p < -tol) or np.any(p > dim + tol):
            errors.append("photon %d: step outside of the domain box" % i)

        if src.get("Type", "pencil") == "pencil" and np.max(np.abs(p[0] - srcpos)) > tol:
            errors.append("photon %d: first step %s is not the source position" % (i, p[0]))

        if np.any(np.diff(w) > tol * max(1.0, w[0])):
            errors.append("photon %d: weight increases along the path" % i)

        if mua is not None and len(ph) > 2:
            seg = np.linalg.norm(np.diff(p[:-1], axis=0), axis=1)
            ratio = w[1:-1] / np.maximum(w[:-2], 1e-30)
            expect = np.exp(-mua * seg)
            bad = np.abs(ratio - expect) > 0.1 * wtol * expect
            if np.any(bad):
                k = int(np.argmax(bad))
                errors.append("photon %d: step %d weight ratio %g != exp(-mua*L) %g" % (i, k + 1, ratio[k], expect[k]))

        onbnd = np.any(np.abs(p[-1]) <= tol * 10) or np.any(np.abs(p[-1] - dim) <= tol * 10)
        if not onbnd and w[-1] > 1e-3:
            errors.append("photon %d: ends inside the domain with weight %g" % (i, w[-1]))

    return errors


def compare(photons, ref, tol, wtol, minmatch=1.0):
    """Compare two trajectory sets; return (errors, number of matching steps)

    a photon diverging from the reference is reported as an error unless the
    fraction of matching steps over all photons is at least minmatch
    """
    errors = []
    diverged = []
    matched = 0

    if len(photons) != len(ref):
        errors.append("photon count differs: %d vs reference %d" % (len(photons), len(ref)))

    for i, (a, b) in enumerate(zip(photons, ref)):
        n = min(len(a), len(b))
        dp = np.max(np.abs(a[:n, :3] - b[:n, :3]), axis=1)
        dw = np.abs(a[:n, 3] - b[:n, 3]) / np.maximum(np.abs(b[:n, 3]), 1e-6)
        bad = np.nonzero((dp > tol) | (dw > wtol))[0]
        matched += int(bad[0]) if len(bad) else n
        if len(bad):
            k = int(bad[0])
            diverged.append("photon %d: diverges at step %d/%d, |dp|=%g, |dw|/w=%g" % (i, k, n, dp[k], dw[k]))
        elif len(a) != len(b):
            diverged.append("photon %d: step count differs: %d vs reference %d" % (i, len(a), len(b)))

    total = sum(len(a) for a in photons)

    if diverged and matched < minmatch * total:
        errors += diverged
    else:
        for d in diverged:
            print("  (tolerated) " + d)

    return errors, matched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", help="session_traj.jdt from -D M, or a golden .json")
    ap.add_argument("--case", required=True, help="the JSON input used to run the simulation")
    ap.add_argument("--ref", help="golden .json (any solver) to compare against")
    ap.add_argument("--tol", type=float, default=1e-4, help="position tolerance (default 1e-4)")
    ap.add_argument("--wtol", type=float, default=1e-4, help="relative weight tolerance (default 1e-4)")
    ap.add_argument("--min-match", type=float, default=1.0, dest="minmatch",
                    help="minimum fraction of steps matching --ref to pass (default 1.0)")
    ap.add_argument("--save", help="write the trajectories to a solver-neutral golden .json")
    ap.add_argument("--solver", default="mcx", help="solver name stored with --save")
    opt = ap.parse_args()

    with open(opt.case, "r") as fp:
        case = json.load(fp)

    photons = load_traj(opt.traj)
    nstep = sum(len(ph) for ph in photons)
    errors = check_invariants(photons, case, opt.tol, opt.wtol)
    print("%s: %d photons, %d steps, invariants %s" % (opt.traj, len(photons), nstep, "FAILED" if errors else "ok"))

    if opt.ref:
        cerr, matched = compare(photons, load_traj(opt.ref), opt.tol, opt.wtol, opt.minmatch)
        print("compared with %s: %d/%d steps match %s" % (opt.ref, matched, nstep, "FAILED" if cerr else "ok"))
        errors += cerr

    for e in errors[:20]:
        print("  " + e)

    if len(errors) > 20:
        print("  ... %d more" % (len(errors) - 20))

    if opt.save:
        save_golden(opt.save, photons, opt.solver, case["Session"].get("ID", ""))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
