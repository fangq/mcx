![](https://mcx.space/img/mcx_wiki_banner.png)

Monte Carlo eXtreme (MCX)
=========================

-   Author: Qianqian Fang (q.fang at neu.edu)
-   License: GNU General Public License version 3 (GPLv3)
-   Version: 3.0 (v2026, Lucky Lux)
-   Website: <https://mcx.space>
-   Download: <https://mcx.space/wiki/?Get>
-   Discussion: <https://groups.google.com/g/mcx-users>

![Mex and Binaries](https://github.com/fangq/mcx/actions/workflows/build_all.yml/badge.svg)\
![Linux Python Module](https://github.com/fangq/mcx/actions/workflows/build_linux_manywheel.yml/badge.svg)\
![MacOS Python Module](https://github.com/fangq/mcx/actions/workflows/build_macos_wheel.yml/badge.svg)\
![Windows Python Module](https://github.com/fangq/mcx/actions/workflows/build_windows_wheel.yml/badge.svg)

MCX is a GPU-accelerated Monte Carlo simulator of light transport in 3D
turbid media, such as biological tissues. You describe a voxelated domain, its
optical properties, light sources and detectors; MCX launches millions to
billions of photon packets and returns the light distribution inside the domain
(fluence, energy deposition, Jacobians, ...) and the photons that reach your
detectors. A simulation that takes hours on a CPU typically takes seconds on a
modern GPU.

MCX can be used in four ways, all running the same simulation engine:

| Mode | What it is | Best for |
|---|---|---|
| **`mcx` command line** | a standalone executable reading a JSON input file | batch jobs, clusters, scripting, reproducible inputs |
| **MCXLAB** (MATLAB/Octave) | `mcxlab(cfg)`, a mex function with in-memory inputs/outputs | interactive analysis, image reconstruction, plotting |
| **PMCX** (Python) | `pmcx.run(cfg)`, a Python module installed with `pip` | Python/NumPy workflows, notebooks, machine learning |
| **MCX Studio** (GUI) | a graphical editor and viewer that runs mcx, mcxcl and mmc | learning MCX, designing and inspecting simulations |

Table of Contents
-----------------

  * [What's New](#whats-new)
  * [Introduction](#introduction)
  * [Installation and configuration](#installation-and-configuration)
  * [Quick start: one simulation, four ways](#quick-start-one-simulation-four-ways)
  * [Key concepts](#key-concepts)
  * [Running the mcx command line](#running-the-mcx-command-line)
  * [JSON input file format](#json-input-file-format)
  * [Shape constructs](#shape-constructs)
  * [Volume and media formats](#volume-and-media-formats)
  * [Sources](#sources)
  * [Detectors and detected photon data](#detectors-and-detected-photon-data)
  * [Output quantities, files and formats](#output-quantities-files-and-formats)
  * [Reading MCX data in MATLAB and Python](#reading-mcx-data-in-matlab-and-python)
  * [Using MCXLAB in MATLAB and Octave](#using-mcxlab-in-matlab-and-octave)
  * [Using PMCX in Python](#using-pmcx-in-python)
  * [Using MCX Studio](#using-mcx-studio)
  * [Advanced features](#advanced-features)
  * [Performance and best practices](#performance-and-best-practices)
  * [Testing](#testing)
  * [Getting help](#getting-help)
  * [Contributing](#contributing)
  * [Publications and how to cite](#publications-and-how-to-cite)
  * [Acknowledgement](#acknowledgement)

What's New
-------------

MCX v2026 (3.0) is a milestone release. It brings MCX to AMD GPUs, adds adjoint and
frequency-domain (RF) Jacobians for image reconstruction, ships a redesigned GUI and
web service, and adds CPU-verifiable, step-by-step ray-tracing tests. It is highly
recommended to upgrade for all users.

Notable new features include:
* directly building adjoint Jacobians via `cfg.outputtype='adjoint'`: the Jacobians are
  computed from forward simulations of the sources and detectors (`-O A/D/U/V/W/Q`, or
  `cfg.outputtype='adjoint'`, `'adjoint_dcoeff'`, `'adjoint_mus'`, `'adjoint_musp'`,
  `'adjoint_mua_d'`, `'adjoint_mua_musp'`); detectors need a direction
  (`Optode.Detector.Dir` or `cfg.detdir`); both the forward fluence and the Jacobians
  are returned
* frequency-domain (FD/RF) forward simulations and adjoint-based FD-Jacobians
  (complex-valued): with `cfg.omega` set, output type `rf` returns the complex fluence
  using complex photon weights, without replay, and the adjoint output types return
  complex Jacobians
* AMD GPU support via ROCm/HIP; build with `cmake -DUSE_HIP=ON` (default arch `gfx90a`),
  contributed by Jeff Daily (AMD) (#264)
* pattern sources are supported in `wp`/`dcs`/`wptof`/`rfmus` replay; rfmus replay
  uses a double-precision buffer to reduce round-off errors
* `-F bnii` saves binary JData (BJData) volumes, and detected photon/trajectory files
  are now saved as `.jdt` (text JSON) or `.jdb` (binary); `.jdat` files can still be
  loaded for replay
* atomic accumulation is always used; `--atomic 0` is deprecated as it lost energy on
  modern GPUs at no speed benefit
* exact output sums: `make fixed` (`-DUSE_FIXEDPOINT`) accumulates the output as 64-bit
  fixed-point integers instead of a float with a second float buffer; the sums are exact
  and independent of the order of the additions, so a run with `-A 0` is bit-reproducible,
  and the output matches a double-precision build (the float buffers differ from it by up
  to 7e-4 near the source); 1.13x (RTX 4090) and 1.26x (RTX 5090) faster on average over
  13 built-in examples, and as fast on an RTX 2080 SUPER; it runs on every GPU MCX supports
* extended line sources for wedge/fan beams; detected photon scattering counts are
  now saved as unsigned integers to remove the 16777216 cap
* about 10% speed gain in split-voxel MC (SVMC) by reducing register use
* in autopilot mode (`-A 1`, the default), photons are dispatched to the GPU threads
  dynamically and label volumes with up to 127 tissue types are stored in 1 byte per
  voxel; dynamic dispatch alone speeds up simulations by 1.12x (TITAN V) to 1.19x (A100)
  on average, and up to 1.3x for heterogeneous domains such as colin27
* disk sources in 2D domains and Lambertian launch directions (a `-inf` focal length in
  `srcdir`), contributed by @jolkkonen (#274)
* MCXStudio2: a new cross-platform GUI with live JSON preview, shape editor,
  and high-DPI support, to replace MCXStudio
* MCXCloud v2: a redesigned web simulation service supporting both mcx and mmc
* Redbird (MATLAB) and Redbirdpy (Python) image reconstruction toolboxes, which can use
  mcx/mmc as the forward solver, are now bundled with the release packages
* new deterministic tests: `make unittest` checks the ray-tracing helpers on the CPU
  without a GPU, and `test/testtraj.sh` compares deterministic photon trajectories
  step by step against solver-neutral golden data shared with mcxcl and mmc
* the released Linux binaries and mex files are built in a manylinux2014 container and
  run on glibc 2.17 or newer; Octave mex files are now also released for Windows and
  macOS, linked against `liboctmex` and loadable in Octave 10 and newer
* unit tests and coverage: host-side unit tests of `mcx_utils` (JSON and .inp parsing,
  JData decoding, shapes, Mie, output writers) and of the ray-tracing helpers (`make
  unittest`, 1782 checks, no GPU needed) raise the line coverage of the host code to
  about 95%; 141 built-in example tests (`test/testbuiltin.sh`), exported from all
  examples and mcxlab demos, check every run against golden values; the bugs these tests
  found were fixed
* fuzzing: four rounds of JSON-aware mutation fuzzing of the input files and of the
  command-line options, run on AddressSanitizer builds of mcx, mcxcl and mmc, plus a
  fuzzer for the MCX Studio GUI; every crash, hang and memory error they found was fixed
* hardening: JSON arrays are length-checked before use; JData arrays must carry valid
  annotations and decode to their declared size; optical properties, time gates, SrcNum,
  pattern sizes, the voxel size (`-u`), the roulette threshold (`-e`), device workloads
  (`-W`), media formats (`-K`) and compression methods (`-Z`) are validated; string
  options are bounds-checked; invalid input now stops with an error message instead of
  crashing or hanging
* fixes found by fuzzing: a photon whose effective scattering coefficient becomes 0 (g=1
  once `--gscatter` is reached) no longer hangs the kernel, `-e` of 1 or more no longer
  relaunches photons forever, and the zlib decoder stops on truncated streams; fuzzing
  mcx-cl found five more bugs in the shared code, fixed in both: a number given for a
  string field crashed the reader, a source pattern smaller than Param1/Param2 was read
  past its end, a pattern with no non-zero pixel in its sampled area hung the kernel,
  svmc labels beyond the media read past the property table, and a run that detected no
  photon freed the detected-photon buffer
* `Session.OutputType` in JSON inputs accepts the names mcxlab and pmcx use ("flux",
  "adjoint_mus", ...); only the first letter used to be read, so "flux" gave fluence and
  "dcs" an adjoint Jacobian
* `--dumpjson`, `mcx2json` and `json2mcx` now round-trip all settings, including Mie,
  phase function, boundary, RF and polarization settings
* pmcx: all mcxlab example scripts are ported to Python under `pmcx/examples`, and
  `pmcx.svmc` builds split-voxel (SVMC) volumes as `mcxsvmc.m` does

This release also contains a number of bug fixes, including
* multiple fixes in the replay normalization, including `-U 2` to skip the detected
  weight normalization in replay
* srcid is now separated from detid in `.jdt`/`.jdb` detected photon outputs of
  multi-source simulations
* fixed double pre-processing of the input when using `-j`, and the 2^24 cap in
  `wp`/`dcs`/`wptof` replay outputs (reported by Pauliina Hirvi)
* fixed `make volta`/`make ampere` etc. compiling for sm_30
* the host-generated RNG seeds had ~75% of their bits set on Linux and macOS, which
  biased the first photon of each GPU thread; the absorbed fraction drifted by up to
  0.25% with the thread count. The per-thread seeds are now derived from the seed
  (`-E`) by the xorshift128+ generator of the kernels instead of the C library `rand()`,
  which differs between Linux, macOS and Windows, so a fixed seed simulates the same
  photons on every OS; this changes the random streams for a given seed (replay of saved
  seeds is unaffected)
* Stokes vectors are now written to trajectory files with `--trajstokes 1`
* fixed crashes when saving detected photons from `asgn_float`, `mixlabel` and
  `labelplus` volumes, and when reading a JSON detector with a `Dir` field
* pmcx: `detweight` accepts the `w0` returned by `pmcx.mcxlab` (#254); uint32 and
  float32 label volumes are passed as 32-bit labels; `pmcx.mcxlab` honors the
  `'opencl'` option and falls back to pmcxcl when the CUDA module is absent
  (fangq/mcxcl#60); on macOS, where CUDA is not available, pmcx is installed as a
  pure-python package providing the utility functions

We would like to give our special thanks to the following contributors
* @jeffdaily (AMD) for the ROCm/HIP port (#264)
* @PrasoonMishra for the HIP photon-loop optimization (#267)
* @jolkkonen for the 2D disk/Lambertian source (#274), and @jolkkonen and @Jah-yee
  for pmcx fixes (#269, #270)
* @MABansept, @hanno-gmg and Pauliina Hirvi for reporting bugs

The most important changes since v2025.10 are summarized below; `ChangeLog.txt`
lists all of them, with their commit hashes.

**New features**
* [a7ecad7] AMD GPU support via HIP/ROCm (#264), with native FP atomics and fast math [d38c7aa]
* [b22534c] RF (frequency-domain) forward simulations with complex photon weights
* [8dc6071] adjoint Jacobians from forward simulations; `srcid=-2` runs sources and detectors together
* [f54d448] adjoint diffusion-coefficient, mus and mus' Jacobians; [08b1d45] dual Jacobians
  (mua+D, mua+mus') in one run; [2b7f5d4] the forward fluence and the Jacobians are both returned
* [ba36c98] pattern/pattern3d sources in wp/dcs/wptof/rfmus replay (#259)
* [f82c0dd] double-precision buffer for rfmus replay to reduce round-off errors
* [6d7a81a] `-F bnii` binary JNIfTI output; detected photon/trajectory files renamed `.jdat` -> `.jdt`/`.jdb`
* [cf4bc7b] parallel zlib/gzip compression of JData outputs, with a block index for parallel decoding
* [89627a7] line sources extended to wedge/fan beams (#253)
* [12a96f8] 2D disk sources and Lambertian launch directions (#274)
* [46aa8a7] detected photon scattering counts stored as unsigned integers, removing the 2^24 cap
* [d502ece] dynamic photon dispatch and 1-byte label volumes in autopilot mode: 1.07-1.51x faster
* [857b26d] split-voxel MC (SVMC) uses 2 fewer registers, about 10% faster
* [4e2fd41] atomic accumulation is always used; the non-atomic and pre-Fermi paths are removed
* [c59fd99] `make fixed`: exact, order-independent 64-bit fixed-point output sums, matching a double build
* [cc192d9] `photon`: one launcher for mcx, mcxcl and mmc; [9e7a7cc] one JSON input schema for all three

**MCX Studio 2, MCX Cloud and toolboxes**
* [ad17282] MCX Studio 2, a rewritten cross-platform GUI: JSON-centric editing, 3-D preview of domains,
  meshes, fluence and trajectories, shape/domain/mesh editors, themes and high-DPI support
* [af2cb35] MCX Studio 2 on macOS as a universal (arm64 + x86_64) `MCXStudio.app`
* [277ae10] MCX Cloud v2: a redesigned web service (new backend, scheduler and browser frontend)
  supporting both mcx and mmc, with a Redbird FEM engine
* [8ed7c2e] Redbird and Redbirdpy bundled with the release packages
* [46508f0] all mcxlab examples ported to Python (`pmcx/examples`); [ad7a52b] `pmcx.svmc`

**Testing and robustness**
* [e9736d8] host unit tests of the ray-tracing helpers; [894e51f] of `mcx_utils`; host line coverage
  raised to about 95% [a0bb2c1]
* [d410478] step-by-step photon trajectory tests with golden data shared with mcxcl and mmc
* [9f19965] 141 built-in example tests exported from all examples and mcxlab demos
* [c413609] four rounds of JSON-aware fuzzing of inputs and options, and fuzzing of the GUI: all
  crashes, hangs and memory errors fixed [6af09da, 7d01159, a9474cb]; input values now validated
* [641581a] `test/fuzz/dfuzz.py`: mutation fuzzing of the built-in inputs and options, optionally
  differential against a reference build; [45fafd5] with the built-in examples, the tests reach every output deposit path
* [edbb667] fixes for the crashes, hangs and memory errors that fuzzing mcx-cl found in the shared code
* [f8e3b65, 8f985de] unit tests of the mcxlab and pmcx interfaces: input types, every output and source
  type, and short absorbed-fraction checks against 1e7-photon references

**Bug fixes**
* [a29ed84, bc7a88b] host RNG seeds had ~75% of their bits set, biasing each thread's first photon;
  now derived with xorshift128+, uniform and the same on every OS
* [1be8bad, 1877f40, 585f607, 78618aa] replay normalization fixes; `-U 2` skips only the detected-weight
  normalization (#263)
* [5eea2af] 2^24 cap in wp/dcs/wptof replay outputs; [06212d2] replaydet=-1 in nscat/wp/wptof/rfmus (#255)
* [ad0f881, 8781d36] pattern3d and per-pattern weights in RF replay (#259)
* [71b3d6c] double preprocessing of inputs when using `-j`
* [6444821] srcid separated from detid in `.jdt`/`.jdb` detected photon files
* [0bd0a73] nscat saved correctly in `.jdt`/`.jdb` files
* [2170679] `Session.OutputType` names ("flux", "adjoint_mus", ...) read in full, not just the first letter
* [7031386] crash saving detected photons of asgn_float, mixlabel and labelplus volumes
* [45fc209] crash reading a JSON detector with a `Dir` field
* [163753f] the mcx binary saved the slot fields instead of the adjoint Jacobians
* [9ec176e] `detdir(4)` could not set the detector beam profile
* [06fab97, 04eff9c, e3fe174] `--dumpjson`, mcx2json and json2mcx round-trip all settings
* [211fef6] `cfg.flog` honored wherever it appears in cfg; pmcx without `tstep` gives one time gate
* [bec9b10] `make volta`/`make ampere` etc. compiled for sm_30
* [b9359d5] per-voxel media (mua_float, muamus_float, asgn_float, labelplus) scaled mua/mus by
  `unitinmm` twice when `cfg.unitinmm` was set before `cfg.vol`; labelplus scaled mus/g, not mua/mus
* [367bf27] polarized runs without `srciquv` failed at random (an uninitialized Stokes vector);
  `srcdir(4)` = nan (isotropic) or -inf (Lambertian) now also applies to pencil beams, as in mcx-cl

**MCXLAB and PMCX**
* [4959eeb] the mcxlab help rewritten to document every cfg field
* [6863caa] interleaved complex outputs in MATLAB R2018a+
* [2781607] `pmcx.detweight` accepts the `w0` of `pmcx.mcxlab` (#254)
* [e1875a1] `pmcx.mcxlab` honors `'opencl'` and falls back to pmcxcl (fangq/mcxcl#60)
* [ca9d15b] pmcx on macOS installs as a pure-python package with the utility functions
* [7dc7cd1] uint32/float32 label volumes passed as 32-bit labels; unknown media formats rejected
* [69423cb] an empty field of a cfg struct array keeps its default; it was read as 0 (e.g. `isreflect`)
* [1190acf] any numeric class (`gpuid=int32(1)` failed), logical volumes and column vectors are accepted
* [6b1dbb7] `cfg.flog` also controls the echo of the cfg fields: '/dev/null' or 'NUL' silences everything
* [3d52135, ae2dfcf] pmcx: `loadmc2` reads x fastest, `meanscat` per medium, `mcxlab(cfg,'cuda')`, and
  the outputs of several patterns are split by pattern
* [1a2af0d] pmcx: NumPy-style docstrings; `help(pmcx.run)` lists every cfg key and output

**Build and packaging**
* [f36e4e3] Linux releases built in manylinux2014 (glibc 2.17+); [40c276e] Octave mex files for macOS
  and Windows (Octave 10+, liboctmex)
* [2f9b0ce] bundled zmat updated to 1.2.0 (parallel zlib/gzip/LZMA); [29116d5] cJSON updated to 1.7.19
* [aa3476d] smaller release packages (stripped, UPX-compressed binaries)


Introduction
------------

Monte Carlo eXtreme (MCX) simulates how light propagates in 3D heterogeneous,
scattering media by tracing a large number of photon packets through a voxelated
domain. At each step a packet is attenuated by absorption, randomly redirected by
scattering, and reflected or refracted at interfaces between media of different
refractive indices. Accumulating the packets' weights gives a statistically exact
solution of the radiative transfer equation (RTE), the physical model behind
diffuse optics, fluorescence imaging, photobiomodulation dosimetry and many other
biophotonics applications. MCX runs hundreds to thousands of times faster than a
single-threaded CPU code by tracing photons in parallel on a graphics processing
unit (GPU).

Unlike the ray-tracers used in computer graphics, MCX is a quantitative physics
solver: its results have been validated against analytical solutions and
measurements in many publications, and it is widely used to produce reference
solutions. It also computes volumetric quantities - the fluence, energy deposition,
partial path lengths and Jacobians everywhere in the 3D domain - rather than an
image on a camera plane.

**What MCX can do**

-   arbitrary 3D heterogeneous media given as label volumes, continuously varying
    optical properties, or built from simple shapes; 2D simulations
-   time-resolved (time-domain), continuous-wave (CW) and frequency-domain (RF)
    simulations
-   18 source types, from pencil beams to wide-field patterns, spatial-frequency
    sources and user-defined angular distributions; many sources in one run
-   disk detectors and bounding-box faces as detectors; per-photon partial path
    lengths, scattering counts, momentum transfer, exit positions/directions
-   Fresnel reflection, total absorption, mirror and cyclic boundary conditions
-   Jacobians (sensitivity maps) by photon replay or by the adjoint method, for
    absorption, scattering and diffusion coefficients, in CW, time and frequency
    domain
-   photon sharing (many illumination patterns in one run), polarized light,
    split-voxel MC (SVMC) for curved boundaries, custom phase functions
-   multiple GPUs in one simulation; NVIDIA GPUs via CUDA and AMD GPUs via HIP/ROCm
-   JSON-based inputs and JData/NeuroJSON-based outputs that are easy to share and
    to read in any language

**The MCX family**

| Software | Description |
|---|---|
| [MCX](https://github.com/fangq/mcx) | this package: voxel-based MC in CUDA (NVIDIA) or HIP (AMD); includes MCXLAB, PMCX and MCX Studio |
| [MCX-CL](https://github.com/fangq/mcxcl) | the OpenCL version of MCX: NVIDIA, AMD and Intel GPUs, Apple silicon and CPUs; MCXLAB-CL (`mcxlabcl`) and PMCXCL (`pmcxcl`) |
| [MMC](https://github.com/fangq/mmc) | mesh-based MC on tetrahedral meshes, on CPUs (SSE), OpenCL, CUDA and OptiX; MMCLAB and PMMC |
| [MCX Cloud](https://mcx.space/cloud) | run MCX and MMC in a web browser on cloud GPUs |
| [Redbird](https://github.com/fangq/redbird) / [Redbirdpy](https://github.com/fangq/redbirdpy) | diffuse optical tomography forward/inverse solvers that can use MCX/MMC as the forward model |

MCX and MCX-CL read the same input files, accept the same `cfg` structures in
MATLAB/Python and write the same outputs, so scripts move between them by changing
`mcxlab` to `mcxlabcl`, or `pmcx` to `pmcxcl`. The algorithms are described in
[Fang2009], [Yu2018] and [Yan2020]; see [Publications and how to cite](#publications-and-how-to-cite).


Installation and configuration
------------------------------

### What you need

-   **an NVIDIA GPU and its driver** (any GPU from the last 10 years; the released
    binaries run on Kepler, `sm_35`, and every newer generation), or an AMD GPU
    with ROCm when building MCX with HIP. The CUDA toolkit is *not* needed to run
    the released binaries, mex files or Python wheels: the CUDA runtime is linked
    statically.
-   on a computer without a supported NVIDIA/AMD GPU, including Apple silicon
    Macs, use [MCX-CL](https://mcx.space/wiki/?Learn#mcxcl), which runs on almost
    any GPU or CPU.

Most installation problems come from a missing or outdated GPU driver. Run
`nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) first; if it fails, (re)install the driver.
Step-by-step guides are at <https://mcx.space/#documentation>.

### Getting MCX

| Package | How to get it |
|---|---|
| MCX Studio all-in-one package (mcx, mcxcl, mmc, the GUI and the MATLAB/Octave toolboxes) | <https://mcx.space/wiki/?Get> or <https://github.com/fangq/mcx/releases>; on Windows also the installer or `winget install coti.mcxstudio` |
| MCX command line or MCXLAB alone | the same pages, as `mcx-<os>-<arch>-<version>.zip` and `mcxlab-<os>-<arch>-<version>.zip` |
| PMCX (Python) | `python3 -m pip install pmcx` (and `pmcxcl` for MCX-CL, `pmmc` for MMC) |
| Redbird / Redbirdpy | the separate `redbird` package on the same pages; `pip install redbirdpy` |
| nightly builds | <https://mcx.space/nightly/> |
| Docker | `docker pull fangqq/mcx` (needs the NVIDIA Container Toolkit for GPU access) |
| source code | `git clone --recursive https://github.com/fangq/mcx.git` |

The released Linux binaries and mex files are built on glibc 2.17 (manylinux2014)
and run on Linux distributions released since 2014. The Octave mex files for
Linux work with any recent Octave; those for Windows and macOS are linked against `liboctmex` and need Octave
10 or newer.

### The MCX Studio package

The all-in-one package unzips into one folder:

```
MCXStudio/
├── mcxstudio            the GUI (MCXStudio.app on macOS)
├── mcxsuite_addpath.m   adds all the MATLAB/Octave toolboxes to the path
├── MATLAB/              mcxlab, mcxlabcl, mmclab, iso2mesh and redbird
└── MCXSuite/            mcx, mcxcl and mmc, each with its bin/ folder and helper scripts
```

Add `MCXSuite/mcx/bin` (and `mcxcl/bin`, `mmc/bin`) to your `PATH`. In MATLAB or
Octave, `cd` into the `MCXStudio` folder, run `mcxsuite_addpath`, and `savepath` to keep
the paths for later sessions. On macOS, unzip the package under your home folder.

### Setting up the command line

Unzip the package and add its `bin/` folder to your `PATH`, then check that MCX
sees your GPU and runs a built-in benchmark:

```
mcx -L              # list the GPUs MCX can use
mcx -Q cube60       # run the "cube60" benchmark
```

### Setting up MCXLAB (MATLAB/Octave)

Unzip the MCXLAB package (or use the `mcxlab` folder of the full package) and add
it, together with the helper functions in `utils`, to the MATLAB/Octave path:

```matlab
addpath('/path/to/mcxlab');      % mcx.mex*, mcxlab.m and the helper functions
savepath;                        % optional: remember it for future sessions
info = mcxlab('gpuinfo')         % check that the GPU is found
```

### Setting up PMCX (Python)

```
python3 -m pip install pmcx jdata bjdata matplotlib
python3 -c "import pmcx; print(pmcx.gpuinfo())"
```

Binary wheels with the CUDA simulator are provided for Linux (x86_64) and Windows.
On macOS, `pip install pmcx` installs only the utility functions; install `pmcxcl`
to run simulations there.

### Operating system notes

**Windows**: by default, Windows resets a GPU that is busy for more than a few
seconds, which aborts long simulations with "unspecified launch failure". Open the
`setup/win64` folder of the package, right-click `apply_timeout_registry_fix.bat`,
choose **Run as administrator**, and reboot. See the
[FAQ](https://mcx.space/wiki/index.cgi?Doc/FAQ#I_am_getting_a_kernel_launch_timed_out_error_what_is_that).

**Linux laptops with hybrid graphics**: if the screen freezes when MCX runs, use the
NVIDIA GPU as the primary GPU (`sudo prime-select nvidia`, or "NVIDIA (Performance
Mode)" in NVIDIA X Server Settings), or keep the display on the integrated GPU and
leave the NVIDIA GPU for computing (see
[this NVIDIA note](https://nvidia.custhelp.com/app/answers/detail/a_id/3029/~/using-cuda-and-x)).
If MCX fails with `unknown error` after the laptop wakes up from suspension, close
MATLAB/Python and reload the driver module:

```
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm
```

**macOS**: recent Macs have no NVIDIA/AMD GPU support; use MCX-CL (`mcxcl`,
`mcxlabcl`, `pmcxcl`), which runs on Apple silicon GPUs. MCX Studio is provided as
a universal (arm64 and x86_64) `MCXStudio.app`.

### Building from source

```
git clone --recursive https://github.com/fangq/mcx.git
cd mcx/src
make                 # the mcx binary in ../bin, needs the CUDA toolkit
make mex             # mcxlab: ../mcxlab/mcx.mexa64 (needs MATLAB's mex)
make oct             # mcxlab for Octave: ../mcxlab/mcx.mex (needs mkoctfile)
make unittest        # host unit tests, see Testing
make fixed           # exact 64-bit fixed-point output sums (-DUSE_FIXEDPOINT)
```

Targets such as `make pascal`, `make turing` or `make ampere` (and `voltamex`,
`ampereoct`, ...) build for one GPU generation; see `src/Makefile` for the full list.
CMake is also supported, and is how the AMD build is configured:

```
cmake -S src -B build                      # NVIDIA (CUDA)
cmake -S src -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a   # AMD (HIP/ROCm)
cmake --build build
```

`CMAKE_HIP_ARCHITECTURES` accepts any ROCm target (for example `gfx90a`,
`gfx1100`). To build PMCX from source, run `pip install .` in the `pmcx/` folder; see
`pmcx/README.md`. MCX Studio is built with Lazarus; see `mcxstudio2/README.md`.

### Hardware limits worth knowing

-   **GPU memory**: the domain needs Nx\*Ny\*Nz bytes (1-byte labels) and the
    output needs Nx\*Ny\*Nz\*Ng\*4\*2 bytes, where Ng is the number of time gates
    simulated at once (4 bytes per float, doubled to reduce round-off errors; a
    `make fixed` build keeps one 8-byte integer instead). With the default autopilot,
    MCX splits the time gates into as many runs as needed.
-   **media plus detectors**: optical properties and detector positions are kept in
    the GPU's 64 kB constant memory, so their total count must stay below about 4000.
-   **many media**: partial path lengths are kept in shared memory; with many
    media types, MCX launches smaller thread blocks to fit.

GPU speeds across generations are compared at <https://mcx.space/gpubench/>.


Quick start: one simulation, four ways
--------------------------------------

The same simulation - a pencil beam entering a 60x60x60 mm homogeneous cube with
four detectors - in each of the four modes.

**Command line.** Save this as `qtest.json` (adapted from `example/quicktest`):

```json
{
    "Domain": {
        "Dim": [60, 60, 60],
        "OriginType": 1,
        "LengthUnit": 1,
        "Media": [
            {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
            {"mua": 0.005, "mus": 1.0, "g": 0.01, "n": 1.37}
        ]
    },
    "Shapes": [
        {"Grid": {"Tag": 1, "Size": [60, 60, 60]}}
    ],
    "Session": {
        "Photons": 1000000,
        "RNGSeed": 29012392,
        "ID": "qtest"
    },
    "Forward": {
        "T0": 0.0e+00,
        "T1": 5.0e-09,
        "Dt": 5.0e-10
    },
    "Optode": {
        "Source": {
            "Type": "pencil",
            "Pos": [29.0, 29.0, 0.0],
            "Dir": [0.0, 0.0, 1.0]
        },
        "Detector": [
            {"Pos": [29.0, 19.0, 0.0], "R": 1.0},
            {"Pos": [29.0, 39.0, 0.0], "R": 1.0},
            {"Pos": [19.0, 29.0, 0.0], "R": 1.0},
            {"Pos": [39.0, 29.0, 0.0], "R": 1.0}
        ]
    }
}
```

and run

```
mcx -f qtest.json
```

MCX writes `qtest.jnii` (the fluence rate, 60x60x60 voxels x 10 time gates) and
`qtest_detp.jdt` (the detected photons) in the current folder.

**MATLAB/Octave:**

```matlab
cfg.nphoton = 1e6;
cfg.vol = uint8(ones(60, 60, 60));
cfg.prop = [0 0 1 1; 0.005 1 0.01 1.37];   % [mua(1/mm), mus(1/mm), g, n] per label
cfg.issrcfrom0 = 1;
cfg.srcpos = [29 29 0];
cfg.srcdir = [0 0 1];
cfg.detpos = [29 19 0 1; 29 39 0 1; 19 29 0 1; 39 29 0 1];
cfg.tstart = 0; cfg.tend = 5e-9; cfg.tstep = 5e-10;
[flux, detp] = mcxlab(cfg);
imagesc(log10(squeeze(sum(flux.data(:, 30, :, :), 4))));   % CW fluence rate at y=30
```

**Python:**

```python
import numpy as np, pmcx, matplotlib.pyplot as plt
cfg = {
    'nphoton': 1000000,
    'vol': np.ones([60, 60, 60], dtype='uint8'),
    'prop': [[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]],
    'issrcfrom0': 1,
    'srcpos': [29, 29, 0], 'srcdir': [0, 0, 1],
    'detpos': [[29, 19, 0, 1], [29, 39, 0, 1], [19, 29, 0, 1], [39, 29, 0, 1]],
    'tstart': 0, 'tend': 5e-9, 'tstep': 5e-10,
}
res = pmcx.mcxlab(cfg)
plt.imshow(np.log10(res['flux'][:, 29, :, :].sum(axis=2)))
plt.show()
```

**GUI:** start `mcxstudio`, open `qtest.json` (or pick a built-in benchmark), check
the domain in the 3-D view, and press **Run**; the fluence is shown in the same view
when the run ends.


Key concepts
------------

**Grid and units.** The domain is a grid of Nx\*Ny\*Nz cubic voxels of edge length
`LengthUnit` (JSON), `-u` (command line) or `cfg.unitinmm` (MATLAB/Python), in mm.
Positions of sources and detectors are given in *voxel units*. With
`OriginType: 1` (`-z 1`, `cfg.issrcfrom0=1`), the corner of the first voxel is at
[0,0,0]; otherwise (the default in the command line and MCXLAB) it is at [1,1,1],
matching MATLAB's 1-based indices. Absorption (mua) and scattering (mus)
coefficients are in 1/mm regardless of the voxel size.

**Media.** Each voxel holds a medium label. The optical properties of label i are
the (i+1)-th row of `Media` (`cfg.prop`): `[mua, mus, g, n]`, with mus the scattering
(not reduced scattering) coefficient, mus=mus'/(1-g). Label 0 is the background
(outside of the domain): a photon moving from a non-zero voxel into a 0 voxel
leaves the domain. Instead of labels, a voxel can also store its own optical
properties, see [Volume and media formats](#volume-and-media-formats).

**Time gates.** A simulation covers the time window [T0, T1], divided into gates of
width Dt; the output has one volume per gate. Photons are tracked until T1, so for
a CW solution choose T1 long enough for the light to decay (a few ns in tissue),
and sum the gates.

**Photons and statistics.** Results are averages over the simulated photon
packets; their noise falls as 1/sqrt(number of photons). 10^6-10^8 photons are
typical. Russian roulette (`-e`) terminates packets of low weight without biasing
the result.

**Normalization.** By default the output is normalized to a unit source, so MCX
produces the Green's function of the fluence rate (1/(mm^2 s), or W/mm^2 per watt
of source power), directly comparable with analytical solutions. `-U 0`
(`cfg.isnormalized=0`) returns the raw accumulated weights.

**Boundaries.** At the domain boundary, photons are reflected according to the
Fresnel equations when the refractive indices differ (`-b 1`, the default), or
leave the domain (`-b 0`). `-B`/`cfg.bc` sets each face separately, including
mirror and cyclic boundaries.


Running the mcx command line
----------------------------

A simulation needs a JSON input file (`-f`), or a built-in benchmark (`-Q`).
Command-line options override the corresponding settings in the file:

```
mcx -f input.json [options]
mcx -Q cube60 -n 1e7 -G 1
```

Typing `mcx` (or `mcx -h`) prints all options; the most useful ones are below.
Options taking a value accept it after a space; long options are written as
`--name value`.

**Input and simulation control**

| Option | Meaning |
|---|---|
| `-f file`, `--input` | read a JSON input file; a string starting with `{` is read as inline JSON, `-f -` reads stdin |
| `-Q name`, `--bench` | run a built-in benchmark (`-Q` alone lists them): cube60, cube60b, cube60planar, cubesph60b, onecube, twocube, skinvessel, sphshells, spherebox, colin27 |
| `-N name`, `--net` | download and run a simulation shared on [NeuroJSON.io](https://neurojson.io/db/mcx) (`-N` alone lists them) |
| `-j '{...}'`, `--json` | modify any setting of the input with a JSON fragment, e.g. `-j '{"Optode":{"Source":{"Type":"isotropic"}}}'` |
| `-P '{...}'`, `--shapes` | add shapes to the domain, see [Shape constructs](#shape-constructs) |
| `-n num`, `--photon` | number of photons (1e7 notation accepted) |
| `-r num`, `--repeat` | repeat the simulation num times (total = n\*r); negative: split n into \|num\| runs |
| `-E seed`, `--seed` | RNG seed; a `_detp.jdt`/`.mch` file name replays its detected photons |
| `-u mm`, `--unitinmm` | voxel edge length in mm |
| `-z 0/1`, `--srcfrom0` | 1: the first voxel's corner is [0,0,0]; 0: [1,1,1] |
| `-K fmt`, `--mediabyte` | volume data format, see [Volume and media formats](#volume-and-media-formats) |
| `-a 0/1`, `--array` | volume file in column-major (MATLAB, 0) or row-major (C, 1) order |
| `--gscatter n` | after n scattering events, scatter isotropically (faster) |

**Physics and boundaries**

| Option | Meaning |
|---|---|
| `-b 1/0`, `--reflect` | Fresnel reflection at the domain boundary (1) or none (0) |
| `-B '______'`, `--bc` | per-face boundary conditions for -x,-y,-z,+x,+y,+z: `r` Fresnel, `a` absorbing, `m` mirror, `c` cyclic, `_` use `-b`; 6 more digits mark faces as detectors (`'______001000'` detects photons leaving z=0) |
| `-V 0/1`, `--specular` | include the specular reflection when the source is outside the domain |
| `-k 1/0`, `--voidtime` | for a source outside the domain, start the clock at launch (1) or at entry (0) |
| `-e w`, `--minenergy` | Russian roulette threshold |
| `-U 1/0/2`, `--normalize` | normalize the output (1), raw output (0), or in replay skip the detected-weight normalization (2) |

**Output**

| Option | Meaning |
|---|---|
| `-s name`, `--session` | name of the output files |
| `-O X`, `--outputtype` | the output quantity, see [Output quantities](#output-quantities-files-and-formats) |
| `-F fmt`, `--outputformat` | volume file format: `jnii` (default), `bnii`, `nii`, `hdr`, `mc2`, `tx3` |
| `-Z zlib`, `--zip` | compression for jnii/bnii: `zlib`, `gzip`, `base64`, `lzip`, `lzma`, `lz4`, `lz4hc` |
| `-d 1/0/3`, `--savedet` | save detected photons (1), not (0), or stop when `-H` photons are detected (3) |
| `-w DP`, `--savedetflag` | detected photon data to save, see [Detectors](#detectors-and-detected-photon-data) |
| `-H num`, `--maxdetphoton` | the most detected photons kept |
| `-x 1`, `--saveexit` | save exit positions and directions (same as adding `XV` to `-w`) |
| `-X 1`, `--saveref` | save the diffuse reflectance in the background voxels next to the domain |
| `-m 1`, `--momentum` | save momentum transfer (same as adding `M` to `-w`) |
| `-q 1`, `--saveseed` | save the RNG seeds of detected photons, for replay |
| `-S 1/0`, `--save2pt` | save the volume output (1) or skip it (0) |
| `-M 1`, `--dumpmask` | save the detector mask volume |
| `--dumpjson [file]` | print (or save) the complete input, volume included, as one JSON file |

**GPU**

| Option | Meaning |
|---|---|
| `-L`, `--listgpu` | list the GPUs and exit |
| `-G n` or `-G '1101'`, `--gpu` | use GPU n (from 1), or several GPUs given as a 0/1 mask |
| `-W '50,30,20'`, `--workload` | relative share of photons for each selected GPU |
| `-A 1/0`, `--autopilot` | let MCX choose the thread count and dispatch photons dynamically (1, default), or use `-t`/`-T` (0) |
| `-t num`, `--thread` / `-T num`, `--blocksize` | total threads and threads per block when `-A 0` |
| `-g num`, `--gategroup` | time gates simulated per run when GPU memory is short |
| `-I`, `--printgpu` | print the GPU information, then run |

**Messages and debugging**

| Option | Meaning |
|---|---|
| `-D P`, `--debug` | `P` progress bar, `M` save photon trajectories, `T` save trajectories only, `R` RNG test |
| `--maxjumpdebug n` | the most trajectory points saved |
| `-l [1\|0]`, `--log` | 1 (or `-l` alone): write the messages to `session.log`; 0: print them without the banner |
| `-y zh_CN`, `--lang` | print messages in another language (zh_CN, zh_TW, ja_JP, fr_CA, es_MX, de_DE, ko_KR, hi_IN, pt_BR) |
| `-v`, `--version` | print the version |

**Recipes**

```
mcx -Q cube60b -n 1e8 -G 1101 -W 10,10,10       # three GPUs, equal shares
mcx -f input.json -n 1e7 -s run1 -F bnii -Z lz4  # compressed binary output
mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
mcx -Q cube60planar --dumpjson > cube60planar.json   # a benchmark as an editable file
mcx -Q cube60 --dumpjson | sed -e 's/pencil/cone/g' | mcx -f -
mcx -N aircube60                                 # run a simulation from NeuroJSON.io
mcx -f input.json -E run1_detp.jdt -O J          # replay run1's detected photons for a Jacobian
```

MCX also reads the legacy tMCimg-style `.inp` input files, but they are being phased
out and new features are only available through JSON.


JSON input file format
----------------------

An MCX input file is a JSON object with four required sections, `Domain`,
`Session`, `Forward` and `Optode`, and an optional `Shapes` section. Field names are
case sensitive; the order of fields does not matter, and unknown sections (such as
`Help`) are ignored. A field can also be written in the root object with a dotted
name, e.g. `"Session.Photons": 1e7`. Command-line options take precedence over the
file, and `-j` edits any field without touching the file.

The fields MCX reads:

**`Domain`** - the grid and its media

| Field | Meaning |
|---|---|
| `VolumeFile` | a binary volume file (format set by `MediaFormat` or `-K`), a JSON/JNIfTI (`.json`, `.jnii`, `.bnii`) volume, or a shape file; omit it when `Shapes` builds the volume |
| `Dim` | [Nx, Ny, Nz] |
| `MediaFormat` | voxel format of the volume, `"byte"` by default (see below) |
| `Media` | array of `{"mua":..., "mus":..., "g":..., "n":...}`, the first for label 0 |
| `LengthUnit` | voxel size in mm (`-u`) |
| `OriginType` | 1: first voxel's corner at [0,0,0]; 0: at [1,1,1] (`-z`) |
| `MieScatter` | polarized MC: array of `{"mua", "radius" (micron), "rho" (1/micron^3), "nsph", "nmed"}`, one per label from 1 |
| `InverseCDF` | a user-defined scattering phase function, the inverse CDF of cos(theta) |
| `AngleInverseCDF` | a user-defined launch-angle distribution (also accepted under `Optode.Source`) |

**`Session`** - what to compute and save

| Field | Meaning |
|---|---|
| `ID` | name of the output files (`-s`) |
| `Photons` | number of photons (`-n`) |
| `RNGSeed` | RNG seed (`-E`) |
| `OutputType` | output quantity: `"flux"`, `"fluence"`, `"energy"`, `"jacobian"`, ... or the `-O` letter |
| `OutputFormat` | `"jnii"`, `"bnii"`, `"nii"`, `"hdr"`, `"mc2"`, `"tx3"` (`-F`) |
| `DoMismatch` | Fresnel reflection at the boundary (`-b`) |
| `BCFlags` | per-face boundary conditions (`-B`) |
| `DoNormalize` | normalize the output (`-U`) |
| `DoSaveVolume` | save the volume output (`-S`) |
| `DoPartialPath` | save detected photons (`-d`) |
| `SaveDataMask` | detected photon data to save, e.g. `"DPX"` (`-w`) |
| `DoSaveExit`, `DoSaveRef`, `DoSaveSeed`, `DoDCS` | as `-x`, `-X`, `-q` and `-m` |
| `DoSpecular` | specular reflection at entry (`-V`) |
| `DoAutoThread` | autopilot (`-A`) |
| `MaxDetPhoton` | the most detected photons kept (`-H`) |
| `MinEnergy` | Russian roulette threshold (`-e`) |
| `DebugFlag` | debug flags such as `"P"` (`-D`) |
| `RootPath` | folder of the input files (`--root`) |

**`Forward`** - time window: `T0` (start), `T1` (end) and `Dt` (gate width), in seconds.

**`Optode`** - sources and detectors

| Field | Meaning |
|---|---|
| `Source.Type` | source type, see [Sources](#sources) |
| `Source.Pos` | [x, y, z] in voxel units |
| `Source.Dir` | [vx, vy, vz] unit direction, optionally a 4th value: focal length (`nan` isotropic, `-inf` Lambertian) |
| `Source.Param1`, `Source.Param2` | 4 numbers each, depending on `Type` |
| `Source.Pattern` | the pattern of `pattern`/`pattern3d` sources: `{"Nx", "Ny", "Nz", "Data"}`, or a JData array |
| `Source.SrcNum` | number of patterns simulated together (photon sharing) |
| `Source.Weight` | initial photon weight |
| `Source.Frequency` | modulation frequency (Hz) for frequency-domain (RF) outputs |
| `Source.IQUV`, `Source.WaveLength` | Stokes vector and wavelength (nm) for polarized MC |
| `Detector` | array of `{"Pos": [x,y,z], "R": radius}`, and `"Dir"` for the adjoint outputs |
| `SDMap` | channels of the adjoint Jacobians: an array of `[srcid, detid]` arrays (1-based; extra values ignored); default: all source-detector pairs |

`Optode.Source` can also be an array of sources, each with its own `Type`, `Pos`,
`Dir` and parameters. Several examples are in the `example/` folder, for example
`example/quicktest`, `example/skinvessel` and `example/multisrc`. `mcx --dumpjson`
prints the complete input of any simulation, including the defaults, which is a
good template.

Legacy files: `mcx2json.m` and `json2mcx.m` (MATLAB) or `pmcx.mcx2json` /
`pmcx.json2mcx` convert between an MCXLAB/PMCX `cfg` and a JSON input file, so a
simulation set up in MATLAB or Python can be run with the command line, and vice
versa.


Shape constructs
----------------

Instead of supplying a volume file, the domain can be built from simple shapes. The
`Shapes` array is processed in order, each shape overwriting the voxels it covers,
and can be placed in the input file, in a separate `.json` file named by
`Domain.VolumeFile`, or on the command line with `-P`:

```json
"Shapes": [
    {"Name": "head model"},
    {"Origin": [0, 0, 0]},
    {"Grid": {"Tag": 1, "Size": [60, 60, 60]}},
    {"ZLayers": [[1, 5, 1], [6, 8, 2], [9, 60, 3]]},
    {"Sphere": {"Tag": 4, "O": [30, 30, 30], "R": 10}},
    {"Box": {"Tag": 0, "O": [10, 10, 10], "Size": [10, 10, 10]}},
    {"Cylinder": {"Tag": 2, "C0": [0, 30, 20], "C1": [60, 30, 20], "R": 3}}
]
```

| Shape | Parameters | Result |
|---|---|---|
| `Grid` | `Tag`, `Size: [Nx,Ny,Nz]` | (re)creates the grid, filled with `Tag` |
| `Origin` | `[x,y,z]` | origin of the coordinates of the shapes that follow |
| `Sphere` | `Tag`, `O` (center), `R` | a sphere |
| `Box` | `Tag`, `O` (lower corner), `Size` | a box, floating-point coordinates |
| `Subgrid` | `Tag`, `O`, `Size` | a box of whole voxels, integer indices, both ends included |
| `Cylinder` | `Tag`, `C0`, `C1` (axis ends), `R` | a finite cylinder |
| `XLayers`, `YLayers`, `ZLayers` | `[[start, end, tag], ...]` | layers normal to an axis, 1-based voxel indices, ends included |
| `XSlabs`, `YSlabs`, `ZSlabs` | `Tag`, `Bound: [[start, end], ...]` | slabs normal to an axis, floating-point bounds |
| `UpperSpace` | `Tag`, `Coef: [A,B,C,D]` | the half space A\*x+B\*y+C\*z>D |
| `Name` | a string | a comment |

`Tag` is the medium label, and must be smaller than the number of media. Coordinates
start from the corner of the first voxel, at [0,0,0] or [1,1,1] depending on
`OriginType`, unless an `Origin` is given. In MATLAB/Python, the same JSON string can
be passed as `cfg.shapes`; `mcxplotshapes.m` and `pmcx.plotshapes` draw it.
`example/shapetest` has more examples.


Volume and media formats
------------------------

The domain is usually a label volume, but MCX accepts several voxel formats, set by
`Domain.MediaFormat` in JSON or `-K` on the command line. In MCXLAB and PMCX, the
format follows from the type and shape of `cfg.vol`:

| `-K` | `MediaFormat` | MCXLAB/PMCX `cfg.vol` | Each voxel stores |
|---|---|---|---|
| 1 | `byte` | 3D uint8 | a label, 0-255 |
| 2 | `short` | 3D uint16 | a label, 0-65535 |
| 4 | `integer` | 3D uint32 (or single/double) | a label |
| 101 | `mua_float` | 1xNxxNyxNz single | mua; mus, g, n from label 1 |
| 100 | `muamus_float` | - | mua and mus; g, n from label 1 |
| 102 | `muamus_half` | 2xNxxNyxNz single | mua and mus, stored as half floats |
| 104 | `muamus_short` | 2xNxxNyxNz uint16 | mua and mus as 0-65535 levels between labels 1 and 2 |
| 103 | `asgn_byte` | 4xNxxNyxNz uint8 | mua, mus, g, n as 0-255 levels between labels 1 and 2 |
| 96 | `asgn_float` | 4xNxxNyxNz single | mua, mus, g, n |
| 99 | `labelplus` | 3xNxxNyxNz single | a label, plus one property of it replaced by a value |
| 98 | `mixlabel` | 3xNxxNyxNz uint16 | two labels and the fraction of the first |
| 97 | `svmc` | 8xNxxNyxNz uint8 | split-voxel MC: two labels and the interface plane in the voxel |

Volumes with per-voxel properties (all but the label and `svmc` formats) carry no
per-medium partial path data. Binary volume files are read in column-major (MATLAB)
order unless `-a 1` is given. Examples: `demo_continuous_mua_mus.m`,
`demo_label_continous_hybrid.m` and the `demo_svmc_*.m` scripts.


Sources
-------

The source is set by its type, position (`Pos`/`srcpos`), direction (`Dir`/`srcdir`)
and two 4-element parameter vectors (`Param1`/`srcparam1`, `Param2`/`srcparam2`).
The 4th element of the direction is a focal length for the sources marked [\*]:
0 for a collimated beam, positive to converge to a focus that far along the
direction, negative to diverge from a virtual point behind the source, `nan` for an
isotropic launch and `-inf` for a Lambertian (cosine) launch.

| Type | Description and parameters |
|---|---|
| `pencil` | a pencil beam (default) |
| `isotropic` | an isotropic point source |
| `arcsine` | like isotropic, with a uniformly distributed zenith angle |
| `cone` | a uniform cone; Param1[0]: half angle (rad) |
| `zgaussian` | a Gaussian distribution of the zenith angle; Param1[0]: variance |
| `gaussian` [\*] | a collimated Gaussian beam; Param1[0]: waist radius |
| `hyperboloid` [\*] | a hyperboloid Gaussian beam; Param1: [waist radius, distance to focus, Rayleigh range] |
| `disk` [\*] | a uniform disk normal to Dir; Param1[0]: radius, Param1[1]: inner radius of an annulus; in 2D, a line segment |
| `ring` [\*] | a ring or ring sector; Param1: [outer radius, inner radius, start angle, end angle] |
| `planar` [\*] | a uniform quadrilateral with corners Pos, Pos+Param1[0:3], Pos+Param2[0:3] |
| `pattern` [\*] | the `planar` area with intensities from `Pattern` (Param1[3] x Param2[3] pixels, values 0-1) |
| `pattern3d` [\*] | a 3D intensity volume of size Param1[0:3] starting at Pos |
| `fourier` [\*] | a spatial-frequency (SFDI) pattern 0.5\*[1+M\*cos(2\*pi\*(fx\*x+fy\*y)+phi0)] over the `planar` area; fx, fy and phi0, M from Param1[3], Param2[3] |
| `fourierx` [\*] | a general Fourier source; Param1: [v1, \|v2\|], Param2: [kx, ky, phi0, M] |
| `fourierx2d` [\*] | a general 2D Fourier basis; Param1: [v1, \|v2\|], Param2: [kx, ky, phix, phiy] |
| `pencilarray` | an array of pencil beams over the `planar` area; Param1[3] x Param2[3] beams |
| `line` | a line source from Pos to Pos+Param1[0:3], emitting normal to the line; Param2[0]: half angle of a wedge (fan) beam |
| `slit` [\*] | a collimated slit beam along Dir from the same segment; Param2[0:1]: Gaussian broadening |

**Several sources** are given as an array of `Optode.Source` objects, or as one row
per source in `srcpos`/`srcdir`/`srcparam1`/`srcparam2`. By default they are
simulated together into one output; `--srcid -1` (`cfg.srcid=-1`) gives each source
its own output, and `--srcid N` simulates only the N-th source. Example:
`demo_multisrc.m`.

**Initial weight.** A 4th element of `srcpos` (or `Source.Weight`) sets the initial
photon weight, which may be negative; the output scales linearly with it.

**Photon sharing.** A `pattern` source with `SrcNum`/`cfg.srcnum` > 1 simulates
many illumination patterns in a single run, each photon carrying a weight for
every pattern; the output gains a pattern dimension. Example:
`demo_photon_sharing.m` [Yan2020-OL].

**Custom angular distributions.** `Source.AngleInverseCDF` (`cfg.angleinvcdf`)
gives the launch angle distribution as the inverse CDF of theta/pi.
Example: `demo_mcxlab_launchangle.m`.

`mcxpreview(cfg)` (MATLAB) and `pmcx.preview(cfg)` (Python) draw the domain,
sources and detectors, which is the quickest way to check a source setup;
`demo_mcxlab_srctype.m` shows every source type.


Detectors and detected photon data
----------------------------------

Detectors are disks given by a center and a radius (`Optode.Detector`,
`cfg.detpos = [x y z r]`); a photon leaving the domain within a detector is
recorded. Faces of the bounding box can also act as detectors, through the 7th-12th
characters of `-B`/`cfg.bc` (see `demo_bc_det.m`).

For each detected photon, MCX can save the fields selected by `-w`
(`Session.SaveDataMask`, `cfg.savedetflag`), default `DP`:

| Letter | Field | Columns |
|---|---|---|
| `D` | detector ID | 1 |
| `S` | scattering events in each medium | number of media |
| `P` | partial path length in each medium (voxel units) | number of media |
| `M` | momentum transfer in each medium (for DCS) | number of media |
| `X` | exit position | 3 |
| `V` | exit direction | 3 |
| `W` | initial weight | 1 |
| `I` | Stokes vector (polarized MC) | 4 |

These data make it possible to compute detector readings for *different* optical
properties without rerunning the simulation (perturbation Monte Carlo): the
weight of a detected photon is w0\*exp(-sum(mua_i\*L_i)). The helper functions do
this for you: `mcxdetweight`/`pmcx.detweight` (weights), `mcxmeanpath`/
`pmcx.meanpath` (mean partial path lengths), `mcxdettpsf`/`pmcx.dettpsf` (temporal
point spread function), `mcxcwdref`/`pmcx.cwdref` (CW reflectance) and
`mcxdcsg1`/`pmcx.dcsg1` (DCS autocorrelation g1).

Saving the RNG seed of each detected photon (`-q 1`, or requesting the 4th output in
MCXLAB) enables photon replay, see [Advanced features](#advanced-features).


Output quantities, files and formats
------------------------------------

### Output quantities

`-O` (`Session.OutputType`, `cfg.outputtype`) selects what the volume output holds:

| `-O` | Name | Quantity |
|---|---|---|
| `X` | `flux` | fluence rate, 1/(mm^2 s) per unit source (default) |
| `F` | `fluence` | fluence, 1/mm^2, integrated over each time gate |
| `E` | `energy` | energy deposited per voxel |
| `L` | `length` | total path length per voxel |
| `J` | `jacobian` (`wl`) | replay: mua Jacobian |
| `P` | `nscat` (`wp`) | replay: weighted scattering counts, for the mus Jacobian |
| `M` | `wm` | replay: weighted momentum transfer |
| `R` | `rf` | replay: frequency-domain mua Jacobian; without replay: complex fluence rate |
| `S` | `rfmus` | replay: frequency-domain mus Jacobian |
| `T` | `wltof` | replay: time of flight x path length |
| `B` | `wptof` | replay: time of flight x scattering count |
| `A` | `adjoint` | adjoint mua Jacobian |
| `D` | `adjoint_dcoeff` | adjoint diffusion coefficient Jacobian |
| `U` | `adjoint_mus` | adjoint mus Jacobian |
| `V` | `adjoint_musp` | adjoint mus' Jacobian |
| `W` | `adjoint_mua_d` | adjoint mua and D Jacobians together |
| `Q` | `adjoint_mua_musp` | adjoint mua and mus' Jacobians together |

### Interpreting the fluence

The volume output is an Nx\*Ny\*Nz\*Ng array of the chosen quantity (with an extra
dimension for photon-sharing patterns, separate sources or replayed detectors). The
default fluence rate is the Green's function for a unit-energy impulse at t=0: the
fraction of the source energy (or photons) passing through a unit area per unit
time. Each gate stores the average over its time window, i.e. approximately the
value at the center of the window, t0+(k-1/2)\*Dt. To obtain

-   the fluence of each gate: multiply the fluence rate by Dt (or use `-O F`);
-   the CW fluence: sum the fluence rate over all gates and multiply by Dt, with
    T1 long enough for the light to decay;
-   the energy deposition: use `-O E`, or multiply the fluence by mua
    (`mcxfluence2energy.m`).

The diffuse reflectance can be obtained from the detected photons (the detected
weight divided by the detector area gives the current density), or with `-X 1`
(`cfg.issaveref=1`), which records the escaping weight in the background voxels
next to the domain (stored as negative values in the raw volume; MCXLAB/PMCX
return it separately as `dref`). Pad the volume with a layer of 0 voxels where you
want the reflectance. The volume output keeps the voxel order of the input volume
(column-major, or row-major with `-a 1`).

`example/validation/plotsimudata.m` and `example/sphbox/plotresults.m` compare MCX
with analytical solutions (`cwdiffusion.m`, `tddiffusion.m`, `pmcx.cwdiffusion`, ...)
and are good templates for checking your own setup.

### Units of the output

With the default normalization (`-U 1`), the output is the response to a source that
emits one unit of light (one joule, or one photon) in an impulse at t=0, so each
quantity can be read in energy or in photon units:

| Output | Energy units | Photon units |
|---|---|---|
| fluence rate (`flux`, default) | W/mm^2 per J emitted | 1/(mm^2 s) per photon emitted |
| fluence (`fluence`) | J/mm^2 per J emitted | 1/mm^2 per photon emitted |
| energy (`energy`) | J absorbed in the voxel per J emitted | the fraction of the photons absorbed there |

To obtain absolute values, multiply by the energy (or number of photons) your source
delivers; for a continuous source of power P, the CW fluence rate is P times the
time-integrated (summed over gates and multiplied by Dt) fluence rate. Divide the
`energy` output by the voxel volume (`LengthUnit`^3) for the absorbed energy density;
`mcxfluence2energy.m` converts a fluence output into energy deposition. See [Fang2009] and Lihong Wang's *Biomedical Optics: Principles
and Imaging*, chapter 5, for the definitions.

**Fluence rate and current density.** The fluence rate is a scalar: the light passing
through a small sphere around a point from all directions. The current density J is a
vector: the net light flowing through a surface in a given direction; it has the same
units. What a detector or the diffuse reflectance measures is the current density
flowing out of the domain: the total weight of the detected photons divided by the
detector area gives J at the detector, as a function of time when the photons are
binned by their time of flight.

### Output files of the command line

| File | Content |
|---|---|
| `session.jnii` (`-F jnii`, default) | the volume output, JSON-based JNIfTI with compressed binary data |
| `session.bnii` (`-F bnii`) | the same in binary JSON (BJData), smaller and faster to read |
| `session.nii` / `.hdr`+`.img` (`-F nii`/`hdr`) | NIfTI-1 / Analyze 7.5 |
| `session.mc2` (`-F mc2`) | raw float32 array, column-major, no header (legacy) |
| `session_detp.jdt` / `.jdb` | detected photons (and seeds with `-q 1`), with `jnii`/`bnii` |
| `session.mch` | detected photons in the legacy binary format, with `mc2`/`nii`/`hdr` |
| `session_traj.jdt` / `.jdb` (or `.mct`) | photon trajectories, with `-D M` |
| `session.log` | the messages, with `-l` |

JNIfTI (`.jnii`, `.bnii`) and JData (`.jdt`, `.jdb`) files follow the
[NeuroJSON](https://neurojson.org) specifications: they are plain JSON or binary
JSON, readable in any language, with the array size, type and compression stored
alongside the data. Detected photon files have the structure

```
{"MCXData": {
    "Info":       {"Version", "MediaNum", "DetNum", "Media", ...},
    "PhotonData": {"detid", "srcid", "nscat", "ppath", "mom", "p", "v", "w0"},
    "Seed":       [...]
}}
```

with only the fields selected by `-w` (`srcid` only with several sources, `Seed`
only with `-q 1`). Trajectory files hold a `Trajectory` object with `photonid`,
`p`, `w0`, `srcid` and, with `--trajstokes 1`, `iquv`. Files with the older `.jdat`
suffix are still accepted for replay.

The legacy `.mch`/`.mct` files have a 256-byte header (`MCXHistoryHeader`: magic
`MCXH`, version, number of media, detectors, columns, photons simulated, detected
and saved, voxel size, seed bytes, normalizer, respin, srcnum, savedetflag, total
sources) followed by a float32 array of saved photons x columns, and with `-q 1`
the seeds; `loadmch.m` reads them.

### Messages

MCX prints its settings, the GPU(s) used, the time spent in each phase (ms) and,
at the end, the simulation speed (photons/ms) and the absorbed fraction. `-D P`
(`cfg.debuglevel='P'`) adds a progress bar.


Reading MCX data in MATLAB and Python
-------------------------------------

**MATLAB/Octave** - install [JSONLab](https://neurojson.org/download/jsonlab) (or
`octave-jsonlab` on Debian/Ubuntu/Fedora) for JNIfTI/JData files; the other loaders
are in MCX's `utils` folder:

```matlab
nii  = loadjd('qtest.jnii');              % JSONLab: .jnii, .bnii, .jdt, .jdb, ...
flux = nii.NIFTIData;                     % Nx x Ny x Nz x Ng
detp = loadjd('qtest_detp.jdt');
ppath = detp.MCXData.PhotonData.ppath;    % one row per detected photon
traj = loadjd('qtest_traj.jdt');

flux = loadmc2('qtest.mc2', [60 60 60 10]);   % utils: .mc2 needs the dimensions
[data, header] = loadmch('qtest.mch');    % utils: legacy detected photons
nii  = mcxloadnii('qtest.nii');           % utils (MATLAB), or niftiread
```

**Python** - install `jdata` (`pip install jdata bjdata`, or `python3-jdata` on
Debian/Ubuntu):

```python
import jdata as jd
nii  = jd.load('qtest.jnii')              # also .bnii, .jdt, .jdb (needs bjdata)
flux = nii['NIFTIData']                   # numpy array, Nx x Ny x Nz x Ng x 1
detp = jd.load('qtest_detp.jdt')['MCXData']['PhotonData']   # dict of arrays

import pmcx
flux = pmcx.loadmc2('qtest.mc2', [60, 60, 60, 10])
data, header = pmcx.loadmch('qtest.mch')[:2]   # a 3rd value holds the seeds, if saved
```

`.nii` files can be read with `nibabel`.


Using MCXLAB in MATLAB and Octave
---------------------------------

MCXLAB runs MCX inside MATLAB or GNU Octave. The simulation is described by a
struct, `cfg`, and the results come back as variables, with no files involved:

```matlab
[flux, detp, vol, seeds, traj] = mcxlab(cfg);
```

Request only the outputs you need; each one costs time and memory. `cfg` fields
mirror the JSON input (`cfg.nphoton`, `cfg.vol`, `cfg.prop`, `cfg.srcpos`,
`cfg.srcdir`, `cfg.tstart`/`tend`/`tstep`, `cfg.detpos`, `cfg.srctype`,
`cfg.outputtype`, `cfg.gpuid`, ...); `help mcxlab` documents every field, its
default and the outputs. A struct array runs several simulations in one call.

| Output | Content |
|---|---|
| `flux.data` | the volume output, Nx x Ny x Nz x Ng (plus extra dimensions, see above) |
| `flux.dref` | diffuse reflectance, with `cfg.issaveref=1` |
| `flux.stat` | runtime, photons, launched and absorbed energy, normalizer |
| `flux.jmua`, `.jd`, `.jmus`, `.jmusp` | adjoint Jacobians |
| `detp` | detected photons: `detid`, `ppath`, `nscat`, `mom`, `p`, `v`, `w0`, ... |
| `vol` | the preprocessed volume |
| `seeds` | RNG seeds of the detected photons, for replay |
| `traj` | photon trajectories: `id`, `pos`, `data` |

Other calls: `mcxlab('gpuinfo')` lists the GPUs, `mcxlab('version')` returns the
version, `mcxlab(cfg, 'preview')` draws the domain instead of running, and
`mcxlab(cfg, 'opencl')` runs the simulation with MCX-CL (`mcxcl.mex`).

**Helper functions** (in `utils`):

| Task | Functions |
|---|---|
| preview and plotting | `mcxpreview`, `mcxplotshapes`, `mcxplotvol`, `mcxplotphotons`, `islicer`, `slice3i` |
| detected photons | `mcxdetphoton`, `mcxdetweight`, `mcxmeanpath`, `mcxmeanscat`, `mcxdettpsf`, `mcxdettime`, `mcxcwdref`, `mcxdcsg1` |
| analytical solutions | `cwdiffusion`, `cwfluxdiffusion`, `cwfluencediffusion`, `tddiffusion`, `rbgetreff` |
| replay helpers | `mcxrfreplay`, `mcxrfmusreplay` |
| domain helpers | `mcxsvmc` (SVMC volumes), `mcxcreate` (built-in benchmarks), `getdistance` |
| file conversion and loading | `mcx2json`, `json2mcx`, `loadmc2`, `loadmch`, `mcxloadnii`, `mcxfluence2energy` |

**Examples**: about 50 scripts in `mcxlab/examples`, from `demo_mcxlab_basic.m` to
atlas head models (`demo_colin27_atlas.m`, `demo_fullhead_atlas.m`), SFDI
(`demo_digimouse_sfdi.m`), replay and adjoint Jacobians, polarized light and SVMC.
An interactive
[MCXLAB tutorial notebook](https://colab.research.google.com/github/fangq/mcx/blob/master/mcxlab/tutorials/mcxlab_getting_started.ipynb)
runs in the browser with Octave. `mcxlab/README.txt` has more details.


Using PMCX in Python
--------------------

PMCX brings MCX to Python 3.6+, with NumPy arrays for inputs and outputs:

```python
import pmcx
res = pmcx.run(cfg)                        # cfg is a dict with the MCXLAB field names
res = pmcx.run(nphoton=1e6, vol=vol, ...)  # or keyword arguments
res = pmcx.mcxlab(cfg)                     # run(), then decode detp/traj into dicts
```

`res` is a dict with `flux`, `stat` and, when produced, `detp`, `seeds`, `traj`,
`dref`, `vol` and the adjoint Jacobians. `pmcx.run` returns the detected photons as
one 2D array; `pmcx.mcxlab` splits them into named fields, as MCXLAB does, and
handles the MCXLAB-style options (`pmcx.mcxlab(cfg, 'opencl')` uses `pmcxcl`).

| Task | Functions |
|---|---|
| running | `run`, `mcxlab`, `gpuinfo`, `version`, `bench` (built-in benchmark `cfg`s), `mcxcreate` |
| detected photons | `detphoton`, `detweight`, `meanpath`, `meanscat`, `dettpsf`, `dettime`, `cwdref`, `dcsg1` |
| analytical solutions | `cwdiffusion`, `cwfluxdiffusion`, `cwfluencediffusion`, `tddiffusion` |
| replay and domains | `rfreplay`, `rfmusreplay`, `svmc`, `getdistance` |
| files | `loadmc2`, `loadmch`, `mcx2json`, `json2mcx` |
| plotting | `preview`, `plotshapes`, `plotphotons`, `plotvol` |

Every MCXLAB example has a Python counterpart in `pmcx/examples`, and the
[PMCX tutorial notebook](https://colab.research.google.com/github/fangq/mcx/blob/master/pmcx/tutorials/pmcx_getting_started.ipynb)
runs in the browser. `pmcx/README.md` covers installation and building from source.
Replacing `import pmcx` with `import pmcxcl as pmcx` runs the same scripts with
MCX-CL.


Using MCX Studio
----------------

MCX Studio is a graphical front end for MCX, MCX-CL and MMC, for Windows, Linux and
macOS. Starting with v2026 it is MCX Studio 2, a rewrite of the original GUI
(`mcxstudio2/`).

-   **A simulation is its JSON file.** MCX Studio edits the same `.json` file the
    simulators read; there is no separate project format, and the Command and JSON
    tabs show exactly what will be run.
-   **Settings in the order a simulation is described**, with a simple (wizard) and
    a full (expert) view; F1 over any setting explains it and names the matching
    mcx option.
-   **A 3-D view** of the domain, shapes, MMC meshes, sources and detectors, and of
    the results: fluence (`.jnii`, `.bnii`, `.nii`, `.mc2`) and photon trajectories.
    It uses OpenGL 3.3, with a software renderer as a fallback (e.g. on remote
    displays).
-   **Runs the right simulator** for the selected processor (`nvidia`, `amd`, `intel`
    or `cpu`) and domain (voxels or mesh): mcx, mcx built with HIP, mcxcl or mmc.
    They are found beside MCX Studio (as in the all-in-one package), in `~/MCXStudio`
    or on the `PATH`.

It can also be scripted:

```
mcxstudio benchmark1.json benchmark1.jnii          # open an input and its result
mcxstudio --run --processor intel qtest.json       # run with mcxcl
xvfb-run -a mcxstudio --run --quit --processor cpu onecube.json   # headless, mmc on the CPU
xvfb-run -a mcxstudio --snapshot qtest.png qtest.json qtest.jnii  # render a thumbnail
```

Projects of the original MCX Studio (`.mcxp`) open with `mcxstudio -p project.mcxp`
or convert with `utils/mcxp2json.py`. See `mcxstudio2/README.md` for details and
building instructions. The original MCX Studio, `mcxshow` and `mcxviewer` remain in
`mcxstudio/`.


Advanced features
-----------------

### Photon replay and Jacobians

A Jacobian (sensitivity profile) tells how a detector reading changes when the
optical property of a voxel changes; it is the core of diffuse optical tomography.
MCX computes it by *replay* [Yao2018]: run a baseline simulation saving the seeds
of the detected photons, then rerun only those photons, accumulating their
contributions to every voxel:

```matlab
[flux, detp, vol, seeds] = mcxlab(cfg);     % baseline, with seeds
cfg.seed = seeds.data;                      % replay these photons
cfg.detphotons = detp.data;
cfg.outputtype = 'jacobian';                % mua Jacobian ('nscat'/'wp' for mus, 'rf', 'rfmus', ...)
cfg.replaydet = 1;                          % for detector 1 (0: all summed, -1: each separately)
jac = mcxlab(cfg);
```

On the command line: `mcx -f input.json -q 1 -s base`, then
`mcx -f input.json -E base_detp.jdt -O J -Y 1`. Examples: `demo_mcxlab_replay.m`,
`demo_replay_timedomain.m`, `demo_replay_frequencydomain.m`,
`demo_replay_all_jacobian.m`; `example/rfreplay` for the command line.

### Adjoint Jacobians

The adjoint method computes the Jacobians of *all* source-detector pairs from
forward simulations alone: each detector is also launched as a source (it needs a
direction, `Optode.Detector.Dir`/`cfg.detdir`), and the Jacobian is formed from the
forward and adjoint fluences. MCX provides mua (`adjoint`), diffusion coefficient
(`adjoint_dcoeff`), mus (`adjoint_mus`) and mus' (`adjoint_musp`) Jacobians, and two
of them at once (`adjoint_mua_d`, `adjoint_mua_musp`), in CW and, with a modulation
frequency, in the frequency domain. MCXLAB/PMCX return the forward fluence of every
source and detector in `flux.data` and the Jacobians in `flux.jmua`, `flux.jd`, ...
The Jacobians have one channel per source-detector pair, `[Nx,Ny,Nz,Ns*Nd]` with the
detector index varying the fastest; to compute only some pairs, list them, in the
order wanted, as rows of `[srcid detid]` (1-based) in `Optode.SDMap` or `cfg.sdmap`,
and the Jacobians become `[Nx,Ny,Nz,Nch]`, one channel per row:
```json
"Optode": {"Source": {"Pos": [[29,29,0],[20,29,0]], "Dir": [[0,0,1],[0,0,1]]},
           "Detector": [{"Pos": [30,20,0], "R": 2, "Dir": [0,0,1]},
                        {"Pos": [40,30,0], "R": 2, "Dir": [0,0,1]}],
           "SDMap": [[1,1], [2,2]]}
```
Examples: `demo_mcx_adjoint_jacobian.m` and the `demo_validate_mcx_adjoint_*.py`
scripts. The Redbird toolboxes use these outputs for image reconstruction.

### Frequency-domain (RF) simulations

With a modulation frequency (`Optode.Source.Frequency` in Hz, or `cfg.omega` in
rad/s) and output type `rf`, MCX tracks complex photon weights and returns the
complex fluence rate - amplitude and phase - in a single run. Example:
`demo_mcxlab_rfforward.m`.

### Polarized light

With `Domain.MieScatter` (`cfg.polprop`), MCX simulates polarized light in media of
spherical scatterers, tracking the Stokes vector of each photon [Yan2022-pol]; the
detected photons can save their Stokes vectors (`I` in `-w`), and trajectories can
record them (`--trajstokes 1`). Example: `demo_polarized_photon.m`.

### Split-voxel MC (SVMC) and continuous media

Voxels cut a curved boundary into staircases; SVMC [Yan2020] stores, in each
boundary voxel, the two media and the plane separating them, so refraction and
reflection happen at the true interface. Create SVMC volumes from label volumes
with `mcxsvmc.m`/`pmcx.svmc`. For smoothly varying media, give per-voxel optical
properties instead of labels (see the media formats above).

### Custom phase functions

`Domain.InverseCDF` (`cfg.invcdf`) replaces the Henyey-Greenstein phase function by
any distribution, given as the inverse CDF of cos(theta). Example:
`demo_mcxlab_phasefun.m`.

### Diffuse correlation spectroscopy (DCS)

Saving the momentum transfer of detected photons (`-m 1`, `M` in `-w`) gives what
is needed to compute the DCS field autocorrelation g1 with `mcxdcsg1.m` or
`pmcx.dcsg1`.

### Multiple GPUs

`-G 1101` (`cfg.gpuid='1101'`) uses the 1st, 2nd and 4th GPUs together, and
`-W 30,30,40` (`cfg.workload`) splits the photons among them; speed scales nearly
linearly with the number of GPUs.

### Sharing simulations: NeuroJSON and MCX Cloud

`mcx --dumpjson` packs a whole simulation - settings and volume - into one JSON
file that anyone can run with `mcx -f`. Simulations shared on
[NeuroJSON.io](https://neurojson.io/db/mcx) run directly with `mcx -N name`.
[MCX Cloud](https://mcx.space/cloud) [Fang2022-cloud] runs MCX and MMC simulations
from a web browser on cloud GPUs, with no installation.

### Denoising

The `filter` submodule provides a GPU-accelerated adaptive non-local means filter
[Yuan2018] to denoise MCX outputs, so fewer photons are needed for a given noise
level.


Performance and best practices
------------------------------

-   **Use a capable GPU, or several.** MCX speed scales with the number of GPU
    cores: a high-end desktop GPU can be 10x faster than a laptop GPU of the same
    generation, and several GPUs work together almost linearly (`-G`).
-   **Keep the autopilot on** (`-A 1`, the default). It picks the thread and block
    counts for your GPU and hands out photons dynamically, so threads that finish
    early take more work. Set `-t`/`-T` only when you know why.
-   **Use `-A 0` to reproduce a run exactly.** With dynamic dispatch, two runs with
    the same seed agree statistically but not bit by bit. For identical results,
    use `-A 0` and keep the seed, `-t` and `-T` unchanged. Replay always uses the
    static dispatch.
-   **Use 1-byte labels when you can.** Label volumes with up to 127 media are
    stored in one byte per voxel, the fastest format.
-   **Save only what you need.** Detected photon data, seeds and trajectories cost
    memory and time; `-S 0` (or `cfg.issave2pt=0`) skips the volume output when only
    detected photons are wanted.
-   **Mind the time gates.** Many gates multiply the output memory; with limited
    GPU memory, MCX simulates them in groups (`-g`).
-   **More photons, less noise.** Noise falls as 1/sqrt(photons): 4 times the
    photons halves it. Check convergence by comparing runs with different seeds.


Testing
-------

MCX includes several test suites; all but the fuzzer run in its continuous integration:

| Test | What it checks | How to run |
|---|---|---|
| host unit tests | the ray-tracing helpers, JSON/inp parsing, JData, shapes, Mie and output writers, on the CPU (no GPU needed) | `make unittest` in `src`, then `bin/mcx_unittest` |
| regression tests | the command line options and outputs against expected values | `test/testmcx.sh` |
| built-in examples | 141 cases exported from all examples and MCXLAB demos, against golden values | `test/testbuiltin.sh` |
| trajectory tests | step-by-step photon trajectories against solver-neutral golden data shared with MCX-CL and MMC | `test/testtraj.sh` |
| fuzzing | mutated built-in inputs and options: crashes, hangs and sanitizer errors, optionally against a reference build | `test/fuzz/dfuzz.py` |
| PMCX tests | the Python module | `python3 -m unittest discover -s test` in `pmcx/` |

See `test/README.md` for details.


Getting help
------------

-   documentation, FAQ and tutorials: <https://mcx.space/wiki/>
-   the `help mcxlab` text, and `pmcx/README.md`
-   questions and discussion: the [mcx-users mailing list](https://groups.google.com/g/mcx-users)
-   bug reports and feature requests: <https://github.com/fangq/mcx/issues>
-   release announcements: the [mcx-news mailing list](https://groups.google.com/g/mcx-news)

Contributing
------------

MCX is open-source software under the GPLv3 license: you may use, modify and share
it, as long as derived work is also released under the GPLv3. Bug fixes and new
features are welcome:

-   fork <https://github.com/fangq/mcx>, make your change in your fork, keep it small
    and local, and open a pull request; describe what it fixes and how you tested it;
-   or send the change as a `git diff` to the mailing list or a GitHub issue.

Please format C/C++ code with `make pretty` (astyle) in `src`, MATLAB code with
`mh_style --fix` (miss_hit) and Python code with `black` before submitting, and add a
test when you can. Contributions are acknowledged in the What's New section and the
ChangeLog.



Publications and how to cite
----------------------------

If you use MCX in your research, please cite the core papers [Fang2009], [Yu2018]
and [Yan2020], and the papers of the specific features you use. The complete
list is maintained at <https://mcx.space/#publications>.

**Core papers**

- [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon
  Migration in 3D Turbid Media Accelerated by Graphics Processing Units," Opt.
  Express, 17(22), 20178-20190 (2009).
  <https://opg.optica.org/oe/abstract.cfm?uri=oe-17-22-20178>
- [Yu2018] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, "Scalable
  and massively parallel Monte Carlo photon transport simulations for
  heterogeneous computing platforms," J. Biomed. Opt. 23(1), 010504 (2018).
  <https://doi.org/10.1117/1.JBO.23.1.010504>
- [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte Carlo
  algorithm for accurate and efficient photon transport modeling in complex
  bio-tissues," Biomed. Opt. Express, 11(11), 6262-6270 (2020).
  <https://opg.optica.org/boe/abstract.cfm?uri=boe-11-11-6262>

**Features**

- [Hirvi2026] Pauliina Hirvi, Jaakko Olkkonen, Qianqian Fang, Ilkka Nissilä, "Computation
  of frequency- and time-domain Jacobians in optical tomography with Monte Carlo
  simulations," arXiv:2604.27558 (2026), preprint. <https://arxiv.org/abs/2604.27558>
- [Yao2018] Ruoyang Yao, Xavier Intes, Qianqian Fang, "A direct approach to compute
  Jacobians for diffuse optical tomography using perturbation Monte Carlo-based
  photon 'replay'," Biomed. Opt. Express, 9(10), 4588-4603 (2018).
  <https://opg.optica.org/boe/abstract.cfm?uri=boe-9-10-4588>
- [Yan2020-OL] Shijie Yan, Ruoyang Yao, Xavier Intes, Qianqian Fang, "Accelerating
  Monte Carlo modeling of structured-light-based diffuse optical imaging via
  'photon sharing'," Opt. Lett., 45, 2842-2845 (2020).
- [Yan2022-pol] Shijie Yan, Steven L. Jacques, Jessica C. Ramella-Roman, Qianqian
  Fang, "Graphics processing unit-accelerated Monte Carlo simulation of polarized
  light in complex three-dimensional media," J. Biomed. Opt. 27(8), 083015 (2022).
  <https://doi.org/10.1117/1.JBO.27.8.083015>
- [Fang2022-cloud] Qianqian Fang and Shijie Yan, "MCX Cloud - a modern, scalable,
  high-performance and in-browser Monte Carlo simulation platform with cloud
  computing," J. Biomed. Opt. 27(8), 083008 (2022).
  <https://doi.org/10.1117/1.JBO.27.8.083008>
- [Yuan2018] Yaoshen Yuan, Leiming Yu, Zafer Doğan, Qianqian Fang, "Graphics
  processing units-accelerated adaptive nonlocal means filter for denoising
  three-dimensional Monte Carlo photon transport simulations," J. Biomed. Opt.
  23(12), 121618 (2018).
- Matin Raayai Ardakani, Leiming Yu, David R. Kaeli, Qianqian Fang, "Framework for
  denoising Monte Carlo photon transport simulations using deep learning," J.
  Biomed. Opt. 27(8), 083019 (2022). <https://doi.org/10.1117/1.JBO.27.8.083019>

**Mesh-based Monte Carlo (MMC)**, for the related mesh-based simulator

- Shijie Yan, Douglas Dwyer, David R. Kaeli, Qianqian Fang, "Accelerating mesh-based
  Monte Carlo simulations using contemporary graphics ray-tracing hardware," J.
  Biomed. Opt. 31(3), 035002 (2026), RT-MMC. <https://doi.org/10.1117/1.JBO.31.3.035002>
- Qianqian Fang, "Mesh-based Monte Carlo method using fast ray-tracing in Plücker
  coordinates," Biomed. Opt. Express, 1(1), 165-175 (2010).
  <https://opg.optica.org/boe/abstract.cfm?uri=boe-1-1-165>
- Qianqian Fang and Shijie Yan, "Graphics processing unit-accelerated mesh-based
  Monte Carlo photon transport simulations," J. Biomed. Opt. 24(11), 115002 (2019).
  <https://doi.org/10.1117/1.JBO.24.11.115002>
- Shijie Yan, Anh Phong Tran, Qianqian Fang, "A dual-grid mesh-based Monte Carlo
  algorithm for efficient photon transport simulations in complex 3D media," J.
  Biomed. Opt. 24(2), 020503 (2019). <https://doi.org/10.1117/1.JBO.24.2.020503>
- Yaoshen Yuan, Shijie Yan, Qianqian Fang, "Light transport modeling in highly
  complex tissues using the implicit mesh-based Monte Carlo algorithm," Biomed.
  Opt. Express, 12(1), 147-161 (2021).
  <https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-1-147>
- Ruoyang Yao, Xavier Intes, Qianqian Fang, "Generalized mesh-based Monte Carlo for
  wide-field illumination and detection via mesh retessellation," Biomed. Opt.
  Express, 7(1), 171-184 (2016).
  <https://opg.optica.org/boe/abstract.cfm?uri=boe-7-1-171>
- Yuxuan Zhang and Qianqian Fang, "BlenderPhotonics - an integrated open-source
  software environment for three-dimensional meshing and photon simulations in
  complex tissues," J. Biomed. Opt. 27(8), 083014 (2022).
  <https://doi.org/10.1117/1.JBO.27.8.083014>

**Applications**: MCX has been used to model light in the brain (fNIRS and
photobiomodulation dosimetry), neonatal heads, bone, and many other tissues; see
<https://mcx.space/#publications> for examples, and Google Scholar for the many
works citing MCX.


Acknowledgement
---------------

The MCX project is funded by the US National Institutes of Health (NIH) / National
Institute of General Medical Sciences (NIGMS) under grant R01-GM114365. We thank
the users and contributors who reported bugs, suggested features and contributed
code; they are credited in the What's New section and the ChangeLog.

MCX contains modified versions of the below source codes from other open-source
projects (with a compatible license).

### cJSON library by Dave Gamble

- Files: src/cjson folder
- Copyright (c) 2009 Dave Gamble
- URL: https://github.com/DaveGamble/cJSON
- License: MIT License, https://github.com/DaveGamble/cJSON/blob/master/LICENSE

### ZMat data compression unit

- Files: src/zmat/*
- Copyright: 2019-2026 Qianqian Fang
- URL: https://github.com/fangq/zmat
- License: GPL version 3 or later, https://github.com/fangq/zmat/blob/master/LICENSE.txt

### miniz data compression library

- Files: src/zmat/miniz/*
- Copyright: 2010-2014 Rich Geldreich and Tenacious Software LLC, 2013-2014 RAD Game Tools and Valve Software
- URL: https://github.com/richgel999/miniz
- License: MIT License, src/zmat/miniz/LICENSE

### LZ4 data compression library

- Files: src/zmat/lz4/*
- Copyright: 2011-2020, Yann Collet
- URL: https://github.com/lz4/lz4
- License: BSD-2-clause, https://github.com/lz4/lz4/blob/dev/lib/LICENSE

### LZMA SDK and Easylzma data compression libraries

- Files: src/zmat/easylzma/*
- Copyright: 2009, Lloyd Hilaiel, 2008, Igor Pavlov
- License: public-domain
- Comment:
 All the cruft you find here is public domain.  You don't have to
 credit anyone to use this code, but my personal request is that you mention
 Igor Pavlov for his hard, high quality work.

### UBJ (UBJSON) library by Steven Braeger

- Files: src/ubj/*
- Copyright (c) 2014 Steven Braeger
- URL: https://github.com/Steve132/ubj
- License: MIT License, src/ubj/LICENSE

### Pybind11

- Files: src/pybind11/*
- Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
- URL: https://github.com/pybind/pybind11/
- License: BSD-style license, https://github.com/pybind/pybind11/blob/master/LICENSE

### MCX Filter submodule

- Files: filter/*
- Copyright (c) 2018 Yaoshen Yuan, 2018 Qianqian Fang
- URL: https://github.com/fangq/GPU-ANLM/
- License: MIT License, https://github.com/fangq/GPU-ANLM/blob/master/LICENSE.txt

### myslicer toolbox by Anders Brun

- Files: utils/{islicer.m, slice3i.m, image3i.m}
- Copyright (c) 2009 Anders Brun, anders@cb.uu.se
- URL: https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume
- License: BSD-3-clause License, https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume#license_modal

### GLScene library for Lazarus by GLScene developers

- Files: mcxstudio/glscene/*
- Copyright (c) GLScene developers
- URL: http://glscene.org, https://sourceforge.net/p/glscene/code/HEAD/tree/branches/GLSceneLCL/
- License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
- Comment:
  A subset of the GLSceneLCL branch is included to compile the original MCX
  Studio (`mcxstudio/`) without installing the full package.

### Texture3D sample project by Jürgen Abel

- Files: mcxstudio/mcxview.pas
- Copyright (c) 2003 Jürgen Abel
- License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
- Comment:
  The volume renderer of the original MCX Studio (mcxviewer) was adapted from the
  Texture3D example of the GLScene Project (http://glscene.org) by Jürgen Abel.

### Synapse communication library for Lazarus

- Files: mcxstudio/synapse/*
- Copyright (c) 1999-2017, Lukas Gebauer
- URL: http://www.ararat.cz/synapse/
- License: MIT License or LGPL version 2 or later or GPL version 2 or later
- Comment:
  A subset of the Synapse units is included to compile the original MCX Studio
  (`mcxstudio/`) without installing the full package.

### pymcx Python module

- Copyright (c) 2020 Maxime Baillot <maxime.baillot.1 at ulaval.ca>
- URL: https://github.com/4D42/pymcx
- License: GPL version 3 or later, https://github.com/4D42/pymcx/blob/master/LICENSE.txt
- Comment: an earlier Python interface to MCX.


License
-------

MCX is free software, released under the GNU General Public License version 3 or
later (GPLv3+); see `LICENSE.txt`. MCXLAB and PMCX are distributed under the same
license.
