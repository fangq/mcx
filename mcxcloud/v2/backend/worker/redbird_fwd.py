#!/usr/bin/env python3
"""MCX Cloud v2 redbird worker: minimal FEM diffusion forward solve.

Reads the SAME mcxcloud JSON input that mcx/mmc consume (input.json in the CWD),
builds a redbirdpy cfg from it, runs a single-wavelength CW/RF forward solve, and
writes the nodal fluence to output.jnii (JNIfTI) + detector readings to
output_detp.jdat.

Deliberately minimal: single wavelength, fixed optical properties, forward only.
Reconstruction, multi-spectral and microwave (Helmholtz) modes are out of scope and
are not reachable from the mcxcloud schema.

Field mapping (mcxcloud/mmc JSON -> redbird cfg):
  Shapes.MeshNode        -> cfg['node']   (Nn x 3)
  Shapes.MeshElem        -> cfg['elem']   passed through with its 5th (label) column;
                            redbirdpy's meshprep() splits that into cfg['seg'] itself,
                            exactly as rbmeshprep.m does. redbirdpy keeps 1-BASED
                            indices, same as the JSON/mmc convention, so no rebasing.
  Domain.Media           -> cfg['prop']   1:1, no conversion: redbird's prop is also
                            [mua, mus, g, n] with RAW mus (reduced internally as
                            musp = mus*(1-g)), and seg indexes it 0-based (prop[seg])
                            exactly like mmc's elemprop.
  Optode.Source.Pos/Dir  -> cfg['srcpos'] / cfg['srcdir']
  Optode.Detector[].Pos  -> cfg['detpos']
  Forward.Omega          -> cfg['omega']  (rad/s, 0 = CW; the backend derives this from
                            the canonical Optode.Source.Frequency in Hz)
"""

import base64
import json
import sys
import time
import zlib

import numpy as np

import redbirdpy as rb
from redbirdpy.utility import getdetdir, meshprep

# JData _ArrayType_ -> numpy dtype (mirrors the frontend's util.js TYPED table)
JD_TYPES = {
    "uint8": np.uint8, "int8": np.int8,
    "uint16": np.uint16, "int16": np.int16,
    "uint32": np.uint32, "int32": np.int32,
    "single": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
}


def jd_decode(node):
    """Decode a JData-annotated array node (or pass a plain list through).

    mcxcloud reassembles large mesh arrays into JData form, which may be
    zlib/gzip+base64 packed (_ArrayZipData_). Both codecs are Python stdlib, so unlike
    the Octave path this needs no zmat/zlib mex.
    """
    if not isinstance(node, dict):
        return np.asarray(node)
    dtype = JD_TYPES.get(node.get("_ArrayType_"), np.float64)
    size = node.get("_ArraySize_")
    if "_ArrayZipData_" in node:
        raw = base64.b64decode(node["_ArrayZipData_"])
        zt = str(node.get("_ArrayZipType_", "zlib")).lower()
        buf = zlib.decompress(raw, 16 + zlib.MAX_WBITS) if zt == "gzip" else zlib.decompress(raw)
        arr = np.frombuffer(buf, dtype=dtype)
    else:
        arr = np.asarray(node.get("_ArrayData_", []), dtype=dtype).ravel()
    # The packed buffer is flattened ROW-MAJOR (verified against a known mesh bbox:
    # C-order recovers the true extents, Fortran order scrambles the columns), so the
    # logical shape is _ArraySize_ read in C order. _ArrayZipSize_ describes only the
    # pre-flatten BUFFER layout -- notably [2, numel] for complex (real row, imag row) --
    # and must NOT be used as the output shape.
    if node.get("_ArrayIsComplex_"):
        half = arr.size // 2
        arr = arr[:half] + 1j * arr[half:]
    if size:
        arr = arr.reshape(tuple(int(x) for x in size))
    return np.array(arr)  # writable copy (frombuffer is read-only)


def rows(v, ncol=3):
    """Force a JSON coordinate list into an (N, ncol) array.

    JSON gives [x,y,z] as a flat 3-list and [[..],[..]] as nested; a single row must
    not be reinterpreted as a column of 3 separate optodes.
    """
    a = np.atleast_2d(np.asarray(jd_decode(v), dtype=float))
    if a.shape[0] == ncol and a.shape[1] == 1:
        a = a.T
    return a[:, :ncol]


def media_to_prop(media):
    """Domain.Media (list of {mua,mus,g,n} objects or [mua,mus,g,n] rows) -> (N,4)."""
    if isinstance(media, dict):
        media = [media]
    out = []
    for m in media:
        if isinstance(m, dict):
            out.append([m["mua"], m["mus"], m["g"], m["n"]])
        else:
            out.append(list(np.asarray(m, dtype=float).ravel()[:4]))
    return np.asarray(out, dtype=float)


def annotate(arr):
    """Wrap a numpy array as an uncompressed JData-annotated node.

    Uncompressed _ArrayData_ on purpose: the frontend decodes that form natively
    (util.js decodeJDataArray) and it keeps the output dependency-free.

    Flattened in FORTRAN order so that each source's field is a CONTIGUOUS block: the
    frontend's mesh renderer slices frames with a plain subarray(f*frameLen, ...) and
    does no strided/row-major handling on the mesh path (preview.js setMeshFrame), so
    per-source blocks must be contiguous. _ArrayOrder_ is deliberately left off — the
    mesh-output path is gated on the order tag NOT being 'c' (preview.js drawPreview).
    """
    flat = np.asarray(arr).ravel(order="F")
    return {
        "_ArrayType_": "single" if flat.dtype == np.float32 else "double",
        "_ArraySize_": list(np.shape(arr)),
        "_ArrayData_": [float(x) for x in flat],
    }


def main():
    with open("input.json") as fp:
        cfgin = json.load(fp)

    mesh = cfgin.get("Shapes") or cfgin.get("Mesh")
    if not isinstance(mesh, dict) or "MeshNode" not in mesh or "MeshElem" not in mesh:
        sys.exit("redbird: input requires Shapes.MeshNode + Shapes.MeshElem (a tet mesh)")

    cfg = {}
    cfg["node"] = jd_decode(mesh["MeshNode"])[:, :3].astype(float)
    # keep the label column: meshprep() splits it into cfg['seg'] the way mmc uses col 5.
    # int, not float: pyiso2mesh indexes node[] with these directly (elemvolume/meshreorient)
    # and rejects a float index array, unlike MATLAB where everything is a double.
    elem = jd_decode(mesh["MeshElem"])
    cfg["elem"] = elem.astype(int)
    if elem.shape[1] <= 4:
        cfg["seg"] = np.ones(elem.shape[0], dtype=int)

    cfg["prop"] = media_to_prop(cfgin["Domain"]["Media"])

    src = cfgin["Optode"]["Source"]
    cfg["srcpos"] = rows(src["Pos"])
    cfg["srcdir"] = rows(src["Dir"])

    dets = cfgin.get("Optode", {}).get("Detector") or []
    if isinstance(dets, dict):
        dets = [dets]
    detpos = [rows(d["Pos"])[0] for d in dets if isinstance(d, dict) and "Pos" in d]
    if detpos:
        cfg["detpos"] = np.asarray(detpos, dtype=float)
    else:
        # femrhs needs at least one detector column; a centroid probe keeps a
        # detector-less input runnable (the nodal fluence field is what matters)
        cfg["detpos"] = cfg["node"].mean(axis=0, keepdims=True)
        print("[redbird] no detectors given; probing the mesh centroid", flush=True)
    # mcxcloud mesh detectors only carry {Pos, R} (no direction), but getoptodes pushes
    # each optode one transport mean free path INWARD along its dir -- synthesize the
    # inward surface normals. Slice to 3 cols (getdetdir returns Nd x 4: normal + focus).
    cfg["detdir"] = np.asarray(getdetdir(cfg))[:, :3]

    cfg["omega"] = float(cfgin.get("Forward", {}).get("Omega", 0) or 0)

    nsrc = cfg["srcpos"].shape[0]
    print(
        "[redbird] %d nodes, %d elems, %d src, %d det, omega=%g rad/s"
        % (cfg["node"].shape[0], cfg["elem"].shape[0], nsrc, cfg["detpos"].shape[0], cfg["omega"]),
        flush=True,
    )

    t0 = time.time()
    cfg, _ = meshprep(cfg)
    print("[redbird] mesh prep ... %.3f s" % (time.time() - t0), flush=True)

    t0 = time.time()
    out = rb.runforward(cfg)
    print("[redbird] forward solve ... %.3f s" % (time.time() - t0), flush=True)
    detphi, phi = out[0], out[1]

    # femrhs builds one RHS column per source AND one per detector (detectors double as
    # adjoint sources), so phi is Nn x (Nsrc+Ndet) -- keep only the forward source columns.
    phi = np.asarray(phi)
    if phi.ndim == 1:
        phi = phi[:, None]
    phi = phi[:, :nsrc]
    if np.iscomplexobj(phi):
        # RF (omega>0) gives a complex field; store the amplitude, which is what the
        # preview renders. The complex detector readings are preserved in detphi below.
        print("[redbird] complex (RF) field: saving amplitude", flush=True)
        phi = np.abs(phi)
    phi = phi.astype(np.float32)

    jnii = {
        "NIFTIHeader": {
            # plain JSON list on purpose: the frontend reads Dim to count frames and
            # requires a real array (preview.js drawmeshOutput)
            "Dim": list(phi.shape),
            "DataType": "single",
            "BitDepth": 32,
            "Name": "redbird nodal fluence",
        },
        "NIFTIData": annotate(phi),
    }
    with open("output.jnii", "w") as fp:
        json.dump(jnii, fp)

    dp = np.asarray(detphi)
    detout = {"DetPhi": annotate(np.abs(dp) if np.iscomplexobj(dp) else dp)}
    if np.iscomplexobj(dp):
        detout["DetPhiReal"] = annotate(np.real(dp))
        detout["DetPhiImag"] = annotate(np.imag(dp))
    with open("output_detp.jdat", "w") as fp:
        json.dump(detout, fp)

    print(
        "[redbird] wrote output.jnii (%d nodes x %d src) + output_detp.jdat"
        % (phi.shape[0], phi.shape[1]),
        flush=True,
    )


if __name__ == "__main__":
    main()
