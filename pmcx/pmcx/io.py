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


"""MCX data file parsing and conversion functions"""

__all__ = (
    "loadmc2",
    "loadmch",
    "loadfile",
    "mcx2json",
    "json2mcx",
    "loadnii",
)

##====================================================================================
## dependent libraries
##====================================================================================

import struct
import os
import re
import warnings
import json
import numpy as np

##====================================================================================
## implementation
##====================================================================================


def loadmc2(fname, dim, dataformat="float", offset=None):
    """
    data = loadmc2(fname, dim, format)
        or
    data, dref = loadmc2(fname, dim, format, offset)

    Author: Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Parameters:
    -----------
    fname : str
        The file name to the output .mc2 file
    dim : list or tuple
        An array to specify the output data dimension
        Normally, dim=[nx, ny, nz, nt]
    format : str, optional
        A string to indicate the format used to save
        the .mc2 file; if omitted, it is set to 'float' (default: 'float')
    offset : int, optional
        Byte offset to start reading from

    Returns:
    --------
    data : ndarray
        The output MCX solution data array, in the
        same dimension specified by dim
    dref : ndarray (when requested)
        Diffuse reflectance at the surface of the domain.
        If this output is not given while diffuse reflectance
        is recorded, dref is shown as the negative values in
        the data output.

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see http://mcx.sf.net for details
    """

    # Map MATLAB format strings to NumPy dtypes
    format_map = {
        "float": np.float32,
        "single": np.float32,
        "double": np.float64,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "uint64": np.uint64,
        "int64": np.int64,
    }

    # Handle MATLAB-style format strings (e.g., 'uchar=>uchar', 'float=>float')
    if "=>" in dataformat:
        dataformat = dataformat.split("=>")[0]  # Use the input format part

    if dataformat not in format_map:
        raise ValueError(f"Unsupported format: {dataformat}")

    dtype = format_map[dataformat]

    # Open file and read data
    try:
        fid = open(fname, "rb")
    except IOError:
        raise IOError("can not open the specified file")

    if offset is not None:
        fid.seek(offset, 0)  # 0 corresponds to 'bof' (beginning of file)

    # Read all data from file
    file_data = fid.read()
    data = np.frombuffer(file_data, dtype=dtype)
    fid.close()

    # Reshape data according to specified dimensions
    data = data.reshape(dim)

    return data


def loadmch(fname, dataformat="float32", endian="little"):
    """
    data, header = loadmch(fname, format, endian)

    Author: Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Parameters:
    -----------
    fname : str
        The file name to the output .mch file
    format : str, optional
        A string to indicate the format used to save
        the .mch file; if omitted, it is set to 'float32' (default: 'float32')
    endian : str, optional
        Specifying the endianness of the binary file
        can be either 'big' or 'little' (default: 'little')

    Returns:
    --------
    data : ndarray
        The output detected photon data array
        data has at least M*2+2 columns (M=header.medium), the first column is the
        ID of the detector; columns 2 to M+1 store the number of
        scattering events for every tissue region; the following M
        columns are the partial path lengths (in mm) for each medium type;
        the last column is the initial weight at launch time of each detected
        photon; when the momentum transfer is recorded, M columns of
        momentum transfer for each medium is inserted after the partial path;
        when the exit photon position/dir are recorded, 6 additional columns
        are inserted before the last column, first 3 columns represent the
        exiting position (x/y/z); the next 3 columns are the dir vector (vx/vy/vz).
        in polarized photon simulation, the last 4 columns are the exit Stokes vector.
        in other words, data is stored in the follow format
            [detid(1) nscat(M) ppath(M) mom(M) p(3) v(3) w0(1) s(4)]
    headerstruct : dict
        File header info, a dict has the following fields
        [version,medianum,detnum,recordnum,totalphoton,detectedphoton,
         savedphoton,lengthunit,seedbyte,normalizer,respin,srcnum,savedetflag,totalsource]
    photonseed : ndarray (optional)
        If the mch file contains a seed section, this
        returns the seed data for each detected photon. Each row of
        photonseed is a byte array, which can be used to initialize a
        seeded simulation. Note that the seed is RNG specific. You must use
        an identical RNG to utilize these seeds for a new simulation.

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see http://mcx.sf.net for details
    """

    # Handle format string (remove '=>' part if present)
    if "=>" in dataformat:
        dataformat = dataformat.split("=>")[0]

    # Map format strings to numpy dtypes
    format_map = {
        "float32": np.float32,
        "float": np.float32,
        "single": np.float32,
        "double": np.float64,
        "uint": np.uint32,
        "int": np.int32,
        "uchar": np.uint8,
        "char": np.int8,
    }

    if dataformat not in format_map:
        raise ValueError(f"Unsupported format: {dataformat}")

    dtype = format_map[dataformat]

    # Set endian character for struct format
    endian_char = "<" if endian == "little" else ">"

    try:
        fid = open(fname, "rb")
    except IOError:
        raise IOError("can not open the specified file")

    data = []
    header = []
    photonseed = []
    seedbyte = 0
    normalizer = 0
    respin = 0
    srcnum = 0
    savedetflag = 0
    totalsource = 0

    # Read file in chunks looking for MCXH magic headers
    while True:
        try:
            # Try to read magic header
            magicheader = fid.read(4)
            if len(magicheader) < 4:
                break

            if magicheader != b"MCXH":
                if len(header) == 0:
                    fid.close()
                    raise RuntimeError("can not find a MCX history data block")
                break

            # Read header data: version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton
            hd_data = fid.read(7 * 4)  # 7 uint32 values
            if len(hd_data) < 28:
                break
            hd = list(struct.unpack(endian_char + "7I", hd_data))

            if hd[0] != 1:
                raise ValueError("version higher than 1 is not supported")

            # Read additional header fields
            unitmm_data = fid.read(4)
            unitmm = struct.unpack(endian_char + "f", unitmm_data)[0]

            seedbyte_data = fid.read(4)
            seedbyte = struct.unpack(endian_char + "I", seedbyte_data)[0]

            normalizer_data = fid.read(4)
            normalizer = struct.unpack(endian_char + "f", normalizer_data)[0]

            respin_data = fid.read(4)
            respin = struct.unpack(endian_char + "i", respin_data)[0]

            srcnum_data = fid.read(4)
            srcnum = struct.unpack(endian_char + "I", srcnum_data)[0]

            savedetflag_data = fid.read(4)
            savedetflag = struct.unpack(endian_char + "I", savedetflag_data)[0]

            totalsource_data = fid.read(4)
            totalsource = struct.unpack(endian_char + "I", totalsource_data)[0]

            # Skip junk field
            junk_data = fid.read(4)

            # Convert savedetflag to binary and create detflag array
            detflag_int = savedetflag & (2**8 - 1)  # bitand operation
            detflag = [
                (detflag_int >> i) & 1 for i in range(8)
            ]  # convert to binary array

            if endian == "little":
                detflag = detflag[::-1]  # flip for little endian (equivalent to fliplr)

            # Define data length array
            datalen = [1, hd[1], hd[1], hd[1], 3, 3, 1, 4]
            datlen = [
                detflag[i] * datalen[i] if i < len(detflag) else 0
                for i in range(len(datalen))
            ]

            # Read photon data
            total_elements = hd[6] * hd[3]  # savedphoton * colcount
            dat_bytes = fid.read(total_elements * np.dtype(dtype).itemsize)

            if len(dat_bytes) < total_elements * np.dtype(dtype).itemsize:
                break

            dat = np.frombuffer(dat_bytes, dtype=dtype).copy()
            dat = dat.reshape((hd[6], hd[3]))  # reshape to [savedphoton, colcount]

            # Apply unit conversion for path lengths
            if savedetflag and len(detflag) > 2 and detflag[2] > 0:
                start_col = sum(datlen[:2])
                end_col = sum(datlen[:3])
                dat[:, start_col:end_col] = dat[:, start_col:end_col] * unitmm
            elif savedetflag == 0:
                start_col = (
                    1 + hd[1]
                )  # 2 + hd[1] - 1 (converting from 1-based to 0-based)
                end_col = (
                    1 + 2 * hd[1]
                )  # 1 + 2*hd[1] - 1 (converting from 1-based to 0-based)
                dat[:, start_col:end_col] = dat[:, start_col:end_col] * unitmm

            # Append data
            if len(data) == 0:
                data = dat
            else:
                data = np.vstack([data, dat])

            # Read photon seeds if present
            if seedbyte > 0:
                try:
                    seed_bytes = fid.read(hd[6] * seedbyte)
                    if len(seed_bytes) == hd[6] * seedbyte:
                        seeds = np.frombuffer(seed_bytes, dtype=np.uint8)
                        seeds = seeds.reshape((hd[6], seedbyte))
                        if len(photonseed) == 0:
                            photonseed = seeds
                        else:
                            photonseed = np.vstack([photonseed, seeds])
                    else:
                        seedbyte = 0
                        warnings.warn("photon seed section is not found")
                except:
                    seedbyte = 0
                    warnings.warn("photon seed section is not found")

            # Adjust total photon count for respin
            if respin > 1:
                hd[4] = hd[4] * respin

            # Handle header accumulation
            if len(header) == 0:
                header = hd + [unitmm]
            else:
                current_header = hd + [unitmm]
                # Check consistency: version, maxmedia, detnum, colcount, unitmm
                if (
                    header[0] != current_header[0]
                    or header[1] != current_header[1]
                    or header[2] != current_header[2]
                    or header[3] != current_header[3]
                    or header[7] != current_header[7]
                ):
                    raise ValueError(
                        "loadmch can only load data generated from a single session"
                    )
                else:
                    # Accumulate totalphoton, detectedphoton, savedphoton
                    header[4] += current_header[4]  # totalphoton
                    header[5] += current_header[5]  # detectedphoton
                    header[6] += current_header[6]  # savedphoton

        except struct.error:
            break
        except Exception as e:
            if len(header) == 0:
                fid.close()
                raise e
            break

    fid.close()

    # Convert data to numpy array if it's a list
    if isinstance(data, list) and len(data) == 0:
        data = np.array([])

    # Create header struct
    headerstruct = {
        "version": header[0],
        "medianum": header[1],
        "detnum": header[2],
        "recordnum": header[3],
        "totalphoton": header[4],
        "detectedphoton": header[5],
        "savedphoton": header[6],
        "lengthunit": header[7],
        "seedbyte": seedbyte,
        "normalizer": normalizer,
        "respin": respin,
        "srcnum": srcnum,
        "savedetflag": savedetflag,
        "totalsource": totalsource,
    }

    # Return based on what's available
    if len(photonseed) > 0:
        return data, headerstruct, photonseed
    else:
        return data, headerstruct


def loadfile(fname, *varargin):
    """
    data, header = mcxloadfile(fname)
        or
    data, header = mcxloadfile(fname, dim, format)

    Author: Qianqian Fang (q.fang <at> neu.edu)
    Converted to Python

    Parameters:
    -----------
    fname : str
        The file name to the output .mc2/.nii/binary volume file
    *varargin : tuple
        Additional arguments:
        dim : array_like
            An array to specify the output data dimension
            normally, dim=[nx, ny, nz, nt]
        format : str
            A string to indicate the format used to save
            the .mc2 file; if omitted, it is set to 'float'

    Returns:
    --------
    data : ndarray
        The 3-D or 4-D data being loaded
    header : dict
        A dict recording the metadata of the file

    This file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see http://mcx.sf.net for details
    """

    # Parse file path and extension
    pathstr, name_with_ext = os.path.split(fname)
    name, ext = os.path.splitext(name_with_ext)

    # Convert extension to lowercase for case-insensitive comparison
    ext_lower = ext.lower()

    if ext_lower == ".nii":
        nii = loadnii(fname, *varargin)[0]
        data = nii["NIFTIData"]
        header = nii["NIFTIHeader"]

    elif ext_lower == ".mc2":
        data = loadmc2(fname, *varargin)
        data = np.log10(data)

        # Create header structure
        header = {}
        if len(varargin) >= 1:
            header["dim"] = varargin[0]
        header["format"] = data.dtype.name
        header["scale"] = "log10"

    elif ext_lower == ".mch":
        data, header = loadmch(fname)

    else:
        data = loadmc2(fname, *varargin)
        header = {}

    return data, header


def mcx2json(cfg, filestub):
    """
    Format:
        mcx2json(cfg, filestub)

    Save MCXLAB simulation configuration to a JSON file for MCX binary

    Author: Qianqian Fang <q.fang at neu.edu>
    Converted to Python

    Parameters:
    -----------
    cfg : dict
        A dict defining the parameters associated with a simulation.
        Please run 'help mcxlab' or 'help mmclab' to see the details.
        mcxpreview supports the cfg input for both mcxlab and mmclab.
    filestub : str
        The filestub is the name stub for all output files, including:
        filestub.json: the JSON input file
        filestub_vol.bin: the volume file if cfg.vol is defined
        filestub_shapes.json: the domain shape file if cfg.shapes is defined
        filestub_pattern.bin: the domain shape file if cfg.pattern is defined

    Dependency:
        This function depends on the savejson/saveubjson functions from the
        Iso2Mesh toolbox (http://iso2mesh.sf.net) or JSONlab toolbox
        (http://iso2mesh.sf.net/jsonlab)
    """

    # Define the optodes: sources and detectors
    Optode = {}
    Optode["Source"] = {}
    Optode["Source"] = copycfg(cfg, "srcpos", Optode["Source"], "Pos")
    Optode["Source"] = copycfg(cfg, "srcdir", Optode["Source"], "Dir")
    Optode["Source"] = copycfg(cfg, "srciquv", Optode["Source"], "IQUV")
    Optode["Source"] = copycfg(cfg, "srcparam1", Optode["Source"], "Param1")
    Optode["Source"] = copycfg(cfg, "srcparam2", Optode["Source"], "Param2")
    Optode["Source"] = copycfg(cfg, "srctype", Optode["Source"], "Type")
    Optode["Source"] = copycfg(cfg, "srcnum", Optode["Source"], "SrcNum")
    Optode["Source"] = copycfg(cfg, "lambda", Optode["Source"], "WaveLength")

    if "detpos" in cfg and cfg["detpos"] is not None and len(cfg["detpos"]) > 0:
        Optode["Detector"] = []
        detpos = np.array(cfg["detpos"])
        for i in range(detpos.shape[0]):
            detector = {
                "Pos": detpos[i, :3].tolist(),
                "R": detpos[i, 3] if detpos.shape[1] > 3 else 1,
            }
            Optode["Detector"].append(detector)

        if len(Optode["Detector"]) == 1:
            Optode["Detector"] = [Optode["Detector"][0]]

    if "srcpattern" in cfg and cfg["srcpattern"] is not None:
        Optode["Source"]["Pattern"] = np.array(
            cfg["srcpattern"], dtype=np.float32
        ).tolist()

    # Define the domain and optical properties
    Domain = {}
    Domain = copycfg(cfg, "issrcfrom0", Domain, "OriginType", 0)
    Domain = copycfg(cfg, "unitinmm", Domain, "LengthUnit")
    Domain = copycfg(cfg, "invcdf", Domain, "InverseCDF")
    Domain = copycfg(cfg, "angleinvcdf", Domain, "AngleInverseCDF")

    # Convert prop matrix to Media list
    prop = np.array(cfg["prop"])
    Domain["Media"] = []
    for i in range(prop.shape[0]):
        media = {
            "mua": float(prop[i, 0]),
            "mus": float(prop[i, 1]),
            "g": float(prop[i, 2]),
            "n": float(prop[i, 3]),
        }
        Domain["Media"].append(media)

    if "polprop" in cfg and cfg["polprop"] is not None:
        polprop = np.array(cfg["polprop"])
        Domain["MieScatter"] = []
        for i in range(polprop.shape[0]):
            miescatter = {
                "mua": float(polprop[i, 0]),
                "radius": float(polprop[i, 1]),
                "rho": float(polprop[i, 2]),
                "nsph": float(polprop[i, 3]),
                "nmed": float(polprop[i, 4]),
            }
            Domain["MieScatter"].append(miescatter)

    Shapes = None
    if "shapes" in cfg and isinstance(cfg["shapes"], str):
        Shapes = json.loads(cfg["shapes"])
        Shapes = Shapes["Shapes"]

    if "vol" in cfg and cfg["vol"] is not None and "VolumeFile" not in Domain:
        vol = np.array(cfg["vol"])
        vol_dtype = vol.dtype

        # Determine MediaFormat based on volume data type and dimensions
        if vol_dtype in [np.uint8, np.int8]:
            Domain["MediaFormat"] = "byte"
            if vol.ndim == 4 and vol.shape[0] == 4:
                Domain["MediaFormat"] = "asgn_byte"
            elif vol.ndim == 4 and vol.shape[0] == 8:
                # Reshape and convert to uint64 equivalent
                vol = vol.reshape(-1).view(np.uint64).reshape(vol.shape[1:])
                cfg["vol"] = vol
                Domain["MediaFormat"] = "svmc"
        elif vol_dtype in [np.uint16, np.int16]:
            Domain["MediaFormat"] = "short"
            if vol.ndim == 4 and vol.shape[0] == 2:
                Domain["MediaFormat"] = "muamus_short"
        elif vol_dtype in [np.uint32, np.int32]:
            Domain["MediaFormat"] = "integer"
        elif vol_dtype in [np.float32, np.float64]:
            if vol_dtype == np.float64:
                vol = vol.astype(np.float32)
                cfg["vol"] = vol

            if np.all(np.mod(vol.flatten(), 1) == 0):
                if np.max(vol) < 256:
                    Domain["MediaFormat"] = "byte"
                    cfg["vol"] = vol.astype(np.uint8)
                else:
                    Domain["MediaFormat"] = "integer"
                    cfg["vol"] = vol.astype(np.uint32)
            elif vol.ndim == 4:
                if vol.shape[0] == 1:
                    Domain["MediaFormat"] = "mua_float"
                elif vol.shape[0] == 2:
                    Domain["MediaFormat"] = "muamus_float"
                elif vol.shape[0] == 4:
                    Domain["MediaFormat"] = "asgn_float"
        else:
            raise ValueError("cfg.vol has format that is not supported")

        Domain["Dim"] = list(vol.shape)
        if len(Domain["Dim"]) == 4:
            Domain["Dim"] = Domain["Dim"][1:]

        if Shapes is not None:
            # Check if Shapes contains "Grid"
            shapes_json = json.dumps(Shapes, separators=(",", ":"))
            if '"Grid"' not in shapes_json:
                Domain["VolumeFile"] = filestub + "_vol.bin"
                with open(Domain["VolumeFile"], "wb") as fid:
                    cfg["vol"].tobytes()
                    fid.write(cfg["vol"].tobytes())
        else:
            Domain["VolumeFile"] = ""
            Shapes = cfg["vol"]
            if Shapes.ndim == 4 and Shapes.shape[0] > 1:
                Shapes = np.transpose(Shapes, (1, 2, 3, 0))

    # Define the simulation session flags
    Session = {}
    Session["ID"] = filestub
    Session = copycfg(cfg, "isreflect", Session, "DoMismatch")
    Session = copycfg(cfg, "issave2pt", Session, "DoSaveVolume")
    Session = copycfg(cfg, "issavedet", Session, "DoPartialPath")
    Session = copycfg(cfg, "issaveexit", Session, "DoSaveExit")
    Session = copycfg(cfg, "issaveseed", Session, "DoSaveSeed")
    Session = copycfg(cfg, "isnormalize", Session, "DoNormalize")
    Session = copycfg(cfg, "outputformat", Session, "OutputFormat")
    Session = copycfg(cfg, "outputtype", Session, "OutputType")
    Session = copycfg(cfg, "debuglevel", Session, "Debug")
    Session = copycfg(cfg, "autopilot", Session, "DoAutoThread")
    Session = copycfg(cfg, "maxdetphoton", Session, "MaxDetPhoton")
    Session = copycfg(cfg, "bc", Session, "BCFlags")

    if (
        "savedetflag" in cfg
        and cfg["savedetflag"] is not None
        and isinstance(cfg["savedetflag"], str)
    ):
        cfg["savedetflag"] = cfg["savedetflag"].upper()
    Session = copycfg(cfg, "savedetflag", Session, "SaveDataMask")

    if "seed" in cfg and np.isscalar(cfg["seed"]):
        Session["RNGSeed"] = cfg["seed"]
    Session = copycfg(cfg, "nphoton", Session, "Photons")
    Session = copycfg(cfg, "minenergy", Session, "MinEnergy")
    Session = copycfg(cfg, "rootpath", Session, "RootPath")

    # Define the forward simulation settings
    Forward = {}
    Forward["T0"] = cfg["tstart"]
    Forward["T1"] = cfg["tend"]
    Forward["Dt"] = cfg["tstep"]

    # Assemble the complete input, save to a JSON or UBJSON input file
    mcxsession = {
        "Session": Session,
        "Forward": Forward,
        "Optode": Optode,
        "Domain": Domain,
    }

    if Shapes is not None:
        if isinstance(Shapes, np.ndarray):
            mcxsession["Shapes"] = Shapes.tolist()
        else:
            mcxsession["Shapes"] = Shapes

    from jdata import savejd

    savejd(mcxsession, filestub + ".json")


def copycfg(cfg, name, outroot, outfield, defaultval=None):
    """
    Copy configuration field from cfg to outroot with field name mapping

    Parameters:
    -----------
    cfg : dict
        Source configuration dictionary
    name : str
        Field name in source dictionary
    outroot : dict
        Target dictionary
    outfield : str
        Field name in target dictionary
    defaultval : any, optional
        Default value if field doesn't exist in cfg

    Returns:
    --------
    outroot : dict
        Updated target dictionary
    """
    if defaultval is not None:
        outroot[outfield] = defaultval
    if name in cfg:
        outroot[outfield] = cfg[name]
    return outroot


def json2mcx(filename):
    """
    Format:
        cfg = json2mcx(filename)

    Convert a JSON file for MCX binary to an MCXLAB configuration structure

    Author: Qianqian Fang <q.fang at neu.edu>
    Converted to Python

    Parameters:
    -----------
    filename : str or dict
        The JSON input file path or a dict containing the JSON data

    Returns:
    --------
    cfg : dict
        A dict defining the parameters associated with a simulation.
        Please run 'help mcxlab' or 'help mmclab' to see details.

    Dependency:
        This function depends on the jdata module for loading JSON files
        (https://pypi.org/project/jdata/)

    """
    from jdata import loadjd, load as jload

    if isinstance(filename, str):
        json_data = jload(filename)
    elif isinstance(filename, dict):
        json_data = filename
    else:
        raise ValueError("first input is not supported")

    # Define the optodes: sources and detectors
    cfg = {}

    if "Optode" in json_data:
        if "Source" in json_data["Optode"]:
            cfg = icopycfg(cfg, "srcpos", json_data["Optode"]["Source"], "Pos")
            cfg = icopycfg(cfg, "srcdir", json_data["Optode"]["Source"], "Dir")
            if "srcdir" in cfg:
                srcdir = np.array(cfg["srcdir"][:3])
                cfg["srcdir"][:3] = (srcdir / np.linalg.norm(srcdir)).tolist()
            cfg = icopycfg(cfg, "srcparam1", json_data["Optode"]["Source"], "Param1")
            cfg = icopycfg(cfg, "srcparam2", json_data["Optode"]["Source"], "Param2")
            cfg = icopycfg(cfg, "srctype", json_data["Optode"]["Source"], "Type")
            cfg = icopycfg(cfg, "srcnum", json_data["Optode"]["Source"], "SrcNum")

            if "Pattern" in json_data["Optode"]["Source"] and isinstance(
                json_data["Optode"]["Source"]["Pattern"], dict
            ):
                pattern = json_data["Optode"]["Source"]["Pattern"]
                nz = pattern.get("Nz", 1)

                if isinstance(pattern["Data"], str):
                    with open(pattern["Data"], "rb") as fid:
                        pattern_data = np.frombuffer(fid.read(), dtype=np.float32)
                else:
                    pattern_data = np.array(pattern["Data"], dtype=np.float32)

                cfg["srcpattern"] = pattern_data.reshape(
                    (pattern["Nx"], pattern["Ny"], nz)
                )

        if (
            "Detector" in json_data["Optode"]
            and json_data["Optode"]["Detector"] is not None
            and len(json_data["Optode"]["Detector"]) > 0
        ):
            detectors = json_data["Optode"]["Detector"]
            if isinstance(detectors, list):
                # Convert list of detector dicts to matrix format
                detpos = []
                for det in detectors:
                    if isinstance(det, dict):
                        pos = det.get("Pos", [0, 0, 0])
                        r = det.get("R", 1)
                        detpos.append(pos + [r])
                    else:
                        detpos.append(det)
                cfg["detpos"] = detpos
            else:
                # Handle single detector case
                if isinstance(detectors, dict):
                    pos = detectors.get("Pos", [0, 0, 0])
                    r = detectors.get("R", 1)
                    cfg["detpos"] = [pos + [r]]

    # Define the domain and optical properties
    if "Domain" in json_data:
        cfg = icopycfg(cfg, "issrcfrom0", json_data["Domain"], "OriginType")
        cfg = icopycfg(cfg, "unitinmm", json_data["Domain"], "LengthUnit")

        if "Media" in json_data["Domain"]:
            media = json_data["Domain"]["Media"]
            if isinstance(media, list):
                # Convert list of media dicts to matrix format
                prop = []
                for m in media:
                    if isinstance(m, dict):
                        prop.append(
                            [
                                m.get("mua", 0),
                                m.get("mus", 0),
                                m.get("g", 0),
                                m.get("n", 1),
                            ]
                        )
                    else:
                        prop.append(m)
                cfg["prop"] = prop
            else:
                cfg["prop"] = media

    if "Shapes" in json_data:
        cfg["shapes"] = json.dumps({"Shapes": json_data["Shapes"]})

    format_key = None
    if "Domain" in json_data and "VolumeFile" in json_data["Domain"]:
        volume_file = json_data["Domain"]["VolumeFile"]
        fpath, fname = os.path.split(volume_file)
        fname, fext = os.path.splitext(fname)

        if fext == ".json":
            if "Dim" in json_data["Domain"]:
                cfg["vol"] = np.zeros(json_data["Domain"]["Dim"], dtype=np.uint8)
            volume_json = jload(volume_file)
            cfg["shapes"] = json.dumps(volume_json)

        elif fext == ".bin":
            bytelen = 1
            mediaclass = np.uint8

            if "MediaFormat" in json_data["Domain"]:
                format_map = {
                    "byte": (1, np.uint8),
                    "short": (2, np.uint16),
                    "integer": (4, np.uint32),
                    "muamus_float": (8, np.float32),
                    "mua_float": (4, np.float32),
                    "muamus_half": (4, np.uint16),
                    "asgn_byte": (4, np.uint8),
                    "muamus_short": (4, np.uint16),
                    "svmc": (8, np.uint8),
                    "asgn_float": (16, np.float32),
                }

                format_key = json_data["Domain"]["MediaFormat"].lower()
                if format_key in format_map:
                    bytelen, mediaclass = format_map[format_key]
                else:
                    raise ValueError("incorrect Domain.MediaFormat setting")

            # Load binary volume file
            with open(volume_file, "rb") as fid:
                vol_data = np.frombuffer(fid.read(), dtype=np.uint8)

            # Convert to appropriate data type
            vol_data = vol_data.view(mediaclass)

            # Reshape volume
            dim = json_data["Domain"]["Dim"]
            total_elements = len(vol_data)
            vol_shape = [total_elements // np.prod(dim)] + dim
            cfg["vol"] = vol_data.reshape(vol_shape)

            if cfg["vol"].shape[0] == 1:
                if format_key != "mua_float":
                    cfg["vol"] = np.squeeze(cfg["vol"], axis=0)

        elif fext in (".nii", ".jnii", ".nii", ".gz", ".img", ".bnii"):
            jnii = loadjd(volume_file)
            cfg["vol"] = jnii["NIFTIData"]

    elif "shapes" not in cfg:
        cfg["vol"] = np.zeros((60, 60, 60), dtype=np.uint8)

    # Define the simulation session flags
    if "Session" in json_data:
        cfg = icopycfg(cfg, "session", json_data["Session"], "ID")
        cfg = icopycfg(cfg, "isreflect", json_data["Session"], "DoMismatch")
        cfg = icopycfg(cfg, "issave2pt", json_data["Session"], "DoSaveVolume")
        cfg = icopycfg(cfg, "issavedet", json_data["Session"], "DoPartialPath")
        cfg = icopycfg(cfg, "issaveexit", json_data["Session"], "DoSaveExit")
        cfg = icopycfg(cfg, "issaveseed", json_data["Session"], "DoSaveSeed")
        cfg = icopycfg(cfg, "isnormalize", json_data["Session"], "DoNormalize")
        cfg = icopycfg(cfg, "outputformat", json_data["Session"], "OutputFormat")
        cfg = icopycfg(cfg, "outputtype", json_data["Session"], "OutputType")

        if "outputtype" in cfg and len(str(cfg["outputtype"])) == 1:
            otypemap = {
                "x": "flux",
                "f": "fluence",
                "e": "energy",
                "j": "jacobian",
                "p": "nscat",
                "m": "wm",
                "r": "rf",
                "l": "length",
                "s": "rfmus",
                "t": "wltof",
                "b": "wptof",
            }

            if cfg["outputtype"] not in otypemap:
                raise ValueError(f"output type {cfg['outputtype']} is not supported")
            cfg["outputtype"] = otypemap[cfg["outputtype"]]

        cfg = icopycfg(cfg, "debuglevel", json_data["Session"], "Debug")
        cfg = icopycfg(cfg, "autopilot", json_data["Session"], "DoAutoThread")
        cfg = icopycfg(cfg, "seed", json_data["Session"], "RNGSeed")

        if (
            "seed" in cfg
            and isinstance(cfg["seed"], str)
            and re.search(r"\.mch$", cfg["seed"])
        ):
            # Would need to implement loadmch function for MCH file support
            raise NotImplementedError(
                "MCH file format not implemented - requires loadmch function"
            )

        cfg = icopycfg(cfg, "nphoton", json_data["Session"], "Photons")
        cfg = icopycfg(cfg, "rootpath", json_data["Session"], "RootPath")

    # Define the forward simulation settings
    if "Forward" in json_data:
        forward = json_data["Forward"]
        if "T0" in forward:
            cfg["tstart"] = forward["T0"]
        cfg = icopycfg(cfg, "tstart", forward, "T0")
        cfg = icopycfg(cfg, "tend", forward, "T1")
        cfg = icopycfg(cfg, "tstep", forward, "Dt")

    return cfg


def icopycfg(cfg, name, outroot, outfield, defaultval=None):
    """
    Copy configuration field from outroot to cfg with field name mapping

    Parameters:
    -----------
    cfg : dict
        Target configuration dictionary
    name : str
        Field name in target dictionary
    outroot : dict
        Source dictionary
    outfield : str
        Field name in source dictionary
    defaultval : any, optional
        Default value if field doesn't exist in outroot

    Returns:
    --------
    cfg : dict
        Updated target configuration dictionary
    """
    if defaultval is not None and outfield not in outroot:
        outroot[outfield] = defaultval
    if outfield in outroot:
        cfg[name] = outroot[outfield]
    return cfg


def loadnii(*args, **kwargs):
    from jdata import loadjd

    return loadjd(*args, **kwargs)
