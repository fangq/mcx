
Monte Carlo eXtreme (MCX) - CUDA Edition
=========================

-   Author: Qianqian Fang (q.fang at neu.edu)
-   License: GNU General Public License version 3 (GPLv3)
-   Version: 3.0 (v2026, Lucky Lux)
-   Website: <https://mcx.space>
-   Download: <https://mcx.space/wiki/?Get>

![Mex and Binaries](https://github.com/fangq/mcx/actions/workflows/build_all.yml/badge.svg)\
![Linux Python Module](https://github.com/fangq/mcx/actions/workflows/build_linux_manywheel.yml/badge.svg)\
![MacOS Python Module](https://github.com/fangq/mcx/actions/workflows/build_macos_wheel.yml/badge.svg)\
![Windows Python Module](https://github.com/fangq/mcx/actions/workflows/build_windows_wheel.yml/badge.svg)

Table of Contents:

  * [What's New](#whats-new)
  * [Introduction](#introduction)
  * [Requirement and Installation](#requirement-and-installation)
  * [Running Simulations](#running-simulations)
  * [Using JSON-formatted input files](#using-json-formatted-input-files)
  * [Using JSON-formatted shape description files](#using-json-formatted-shape-description-files)
  * [Output data formats](#output-data-formats)
    + [Volumetric output](#volumetric-output)
    + [Detected photon data](#detected-photon-data)
    + [Photon trajectory data](#photon-trajectory-data)
  * [Using MCXLAB in MATLAB and Octave](#using-mcxlab-in-matlab-and-octave)
  * [Using PMCX in Python](#using-pmcx-in-python)
  * [Using MCX Studio GUI](#using-mcx-studio-gui)
  * [Interpreting the Output](#interpreting-the-output)
    + [Output files](#output-files)
    + [Console print messages](#console-print-messages)
  * [Best practices guide](#best-practices-guide)
    + [Use a mid-range or high-end GPU, and multiple GPUs if possible](#use-a-mid-range-or-high-end-gpu-and-multiple-gpus-if-possible)
    + [Launch as many threads as possible](#launch-as-many-threads-as-possible)
    + [Use -A 0 to reproduce a simulation exactly](#use--a-0-to-reproduce-a-simulation-exactly)
  * [Acknowledgement](#acknowledgement)
  * [Reference](#reference)

What's New
-------------

MCX v2026 (3.0) is a milestone release. It brings MCX to AMD GPUs, adds adjoint and
frequency-domain (RF) Jacobians for image reconstruction, ships a redesigned GUI and
web service, and adds CPU-verifiable, step-by-step ray-tracing tests. It is highly
recommended to upgrade for all users.

Notable new features include:
* AMD GPU support via ROCm/HIP; build with `cmake -DUSE_HIP=ON` (default arch `gfx90a`),
  contributed by Jeff Daily (AMD) (#264)
* RF (frequency-domain) forward simulation with complex photon weights, set `cfg.omega`
  and use output type `rf` without replay
* adjoint-method Jacobians computed directly from forward simulations of sources and
  detectors (`-O A/D/U/V/W/Q`, or `cfg.outputtype='adjoint'`, `'adjoint_dcoeff'`,
  `'adjoint_mus'`, `'adjoint_musp'`, `'adjoint_mua_d'`, `'adjoint_mua_musp'`); detectors
  need a direction (`Optode.Detector.Dir` or `cfg.detdir`); both the forward fluence and
  the Jacobians are returned, supporting CW and RF
* pattern sources are supported in `wp`/`dcs`/`wptof`/`rfmus` replay; rfmus replay
  uses a double-precision buffer to reduce round-off errors
* `-F bnii` saves binary JData (BJData) volumes, and detected photon/trajectory files
  are now saved as `.jdt` (text JSON) or `.jdb` (binary); `.jdat` files can still be
  loaded for replay
* atomic accumulation is always used; `--atomic 0` is deprecated as it lost energy on
  modern GPUs at no speed benefit
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
* input parsing is hardened by fuzzing the JSON inputs and command line options;
  invalid values (such as `-u`, `-W`, `-K`, `-Z`) now stop with an error message
  instead of crashing or hanging
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
  0.25% with the thread count. Seeds are now uniform, which changes the random
  streams for a given seed (replay of saved seeds is unaffected)
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

The detailed updates can be found in the below change log

* 2026-10-21 [5013494] [bug] shrink the block when the per-photon shared memory exceeds the device limit
* 2026-10-21 [0d17987] [test] run testmcx.sh in a scratch folder instead of overwriting the files tracked in test/
* 2026-10-21 [7031386] [bug] fix crash saving detected photons with asgn_float, mixlabel and labelplus volumes
* 2026-10-21 [45bc438] [doc] refresh the v2026 ChangeLog, NEWS and README change lists
* 2026-10-21 [a26c733] [pmcx] detphoton: read nscat as uint32 bits only from float32 detp buffers
* 2026-10-21 [7dc7cd1] [bug] pass uint32 and float32 label volumes as 32bit labels in pmcx/mcxlab, reject unknown media formats
* 2026-10-21 [8d03dd3] [doc] refresh the v2026 ChangeLog, NEWS and README change lists
* 2026-10-21 [d502ece] [perf] port mcxcl speedopt to CUDA: dynamic photon dispatch, 1-byte label volume; 1.07-1.51x
* 2026-10-21 [16f3df9] [doc] refresh the v2026 ChangeLog, NEWS and README change lists
* 2026-10-21 [c368809] [test] compare trajectories per photon: 5e-2 mm tolerance, tolerate up to 10% forked photons
* 2026-10-21 [4b063f2] [test] build host unit tests in ~2s by not instantiating the kernel templates
* 2026-10-21 [fa736fc] [release] set the v2026 (3.0) codename to Lucky Lux
* 2026-10-21 [04187cf] [doc] refresh the v2026 ChangeLog, NEWS and README change lists
* 2026-10-21 [45fc209] [bug] fix segfault reading a JSON detector object that has a Dir field
* 2026-10-21 [73b2d79] [doc] refresh v2026 change lists
* 2026-10-21 [0bd0a73]*[bug] save nscat in .jdt/.jdb by reinterpreting the uint bits written by the kernel
* 2026-10-21 [4dca9f6] [test] untrack python bytecode cache
* 2026-10-21 [e0c948d] [test] one photon per thread in trajectory tests, tolerate round-off drift across solvers
* 2026-10-21 [99cd680] [doc] v2026 What's New, ChangeLog/NEWS entries, document the test suites
* 2026-10-21 [5df0751] [release] bump version to v2026 (3.0), pmcx to 0.8.0, update copyright years
* 2026-10-21 [8ed7c2e] [ci] run host unit tests, build mcxstudio2, bundle redbird/redbirdpy
* 2026-10-21 [e9736d8]*[test] host unit tests of ray-tracing helpers, mark pure helpers MCX_HD
* 2026-10-21 [d410478]*[test] add step-by-step photon trajectory tests with solver-neutral golden data
* 2026-10-21 [6444821]*[bug] split srcid from detid in .jdt/.jdb detected photon output
* 2026-10-21 [bec9b10] [build] make arch aliases (volta, ampere, ...) depend on fermi directly
* 2026-10-21 [5333d1e] [test] use compute-sanitizer when available, do not hard-code zlib stream header
* 2026-09-13 [83c828f] [mcxstudio2] paste a simulation into the JSON pane
* 2026-09-13 [87d0406] [mcxstudio2] the JSON pane, coloured
* 2026-09-13 [a915dda] [mcxstudio2] left-aligned titles, and a subsection band that is really a pill
* 2026-09-13 [009b021] [mcxstudio2] open where you are, and give the groups room
* 2026-09-13 [c2d0b7f] [mcxstudio2] the section band is a pill
* 2026-09-13 [6087899] [mcxstudio2] find Lazarus wherever it is, whichever version
* 2026-09-13 [745c1c8] [mcxstudio2] flat: the fields, the buttons and the tab strip
* 2026-09-13 [85f3064] [mcxstudio2] rounder section bands, and air between them
* 2026-09-13 [b8b820b] [mcxstudio2] the hover is the band, not the widget set's idea of one
* 2026-09-13 [ef98727] [mcxstudio2] edit a simulation that has more than one source
* 2026-09-13 [1c3be24] [mcxstudio2] a theme that reaches the buttons, and a band that says which level
* 2026-09-13 [5f2f5dc] [mcxstudio2] start a second time
* 2026-09-13 [37d54c1] [mcxstudio2] the section band is rounded after all, and the chevron comes back up
* 2026-09-13 [facff2a] [mcxstudio2] a band for the selected subsection, and a pointer that says click
* 2026-09-13 [3898f9d] [mcxstudio2] the Shapes editor: a list of commands, and the one selected
* 2026-09-13 [8ae02e1] [mcxstudio2] a toolbar you can hit, at any display scale
* 2026-09-13 [ae5f4eb] [mcxstudio2] fit the result, and a ruler that is behind what it measures
* 2026-09-13 [8513b02] [mcxstudio2] open the binary results mcx actually writes
* 2026-09-13 [d6e3476] [mcxstudio2] build on the Lazarus the README says it needs
* 2026-09-13 [05f20d4] [mcxstudio2] pick the problem, not the program
* 2026-09-13 [5bb7939] [mcxstudio2] a mode switch where the settings are
* 2026-09-13 [5d82797] [mcxstudio2] axes that can be read a number off
* 2026-09-13 [2253a8c] [mcxstudio2] the parts of the window the LCL cannot colour
* 2026-09-13 [63fec5b] [mcxstudio2] mmc, end to end: mesh, fluence and paths
* 2026-09-13 [ce0bb0e] [mcxstudio2] paths that finish, weight that decays, axes that are labelled
* 2026-09-13 [19c74e6] [mcxstudio2] pick which photons to draw
* 2026-09-13 [687881d] [mcxstudio2] make the trajectory checkbox actually produce trajectories
* 2026-09-12 [de8249b] [mcxstudio2] the other two spellings, and what each detector caught
* 2026-09-12 [bd0d0ce] [mcxstudio2] fewer samples while the picture is moving
* 2026-09-12 [f0da10e] [mcxstudio2] packaging: a README, an install, a CI job, and a way in from the old projects
* 2026-09-12 [16750df] [mcxstudio2] open an mmc file as an mmc file, and stop rewriting it
* 2026-09-12 [762f2c7] [mcxstudio2] the source parameters, named for the source
* 2026-09-12 [489309e] [mcxstudio2] colour themes
* 2026-09-12 [f652b62] [mcxstudio2] what each setting does: hints, F1 and About
* 2026-09-12 [50e0fd4] [mcxstudio2] find a backend cloned next to this one
* 2026-09-12 [9fd4529] [mcxstudio2] display controls under the picture
* 2026-09-12 [b604521] [mcxstudio2] photon paths, drawn
* 2026-09-12 [1e210f6] [mcxstudio2] what the options mean, not what they are called
* 2026-09-12 [ea45348] [mcxstudio2] the source is an arrow, and the shape it emits from
* 2026-09-12 [423a7e8] [mcxstudio2] mmc meshes, drawn as surfaces
* 2026-09-12 [6c490f4] [mcxstudio2] picking, and a picture at any size
* 2026-09-12 [790ea80] [mcxstudio2] benchmarks to start from, and Back/Next for the wizard
* 2026-09-12 [87106ef] [mcxstudio2] shapes are surfaces, not outlines
* 2026-09-12 [42cd608] [mcxstudio2] the volume view: a result on the card, everything else a uniform
* 2026-09-12 [c281555] [mcxstudio2] mcxjd: read an array out of JData or BJData
* 2026-09-12 [2229551] [mcxstudio2] the two tables: optical properties and detectors
* 2026-09-12 [391f2ec] [mcxstudio2] mcxview: the 3-D preview moves out of the main form
* 2026-09-12 [d450b6a] [mcxstudio2] draw the Shapes list
* 2026-09-12 [92766df] [mcxstudio2] mcxgl: a 3.3 core context, and the domain drawn in it
* 2026-09-12 [b92f107] [mcxstudio2] mcxrun: find a backend, ask it what it has, run it
* 2026-09-12 [105584b] [mcxstudio2] the settings are a page of the notebook, not a pane of their own
* 2026-09-12 [fb5e50e] [mcxstudio2] real docking, the way led does it: AnchorDocking
* 2026-09-12 [e0f32fa] [mcxstudio2] the preview is one dockable object: right, bottom, or its own window
* 2026-09-12 [195cd0c] [mcxstudio2] tear a page off by dragging its tab, or double-clicking it
* 2026-09-12 [0ff4100] [mcxstudio2] a preview page can be torn off into a window of its own
* 2026-09-12 [7dc6043] [mcxstudio2] Advanced was arriving third, and it was its own stale Top
* 2026-09-12 [5adf3a8] [mcxstudio2] drop the slider; print a float as short as it can be
* 2026-09-12 [5996c5b] [mcxstudio2] spin edits for counts, a slider where both ends are known
* 2026-09-12 [734a2ca] [mcxstudio2] a card is a panel, because a group box cannot lose its frame
* 2026-09-12 [698be5e] [mcxstudio2] three colours, and a selection that stays put
* 2026-09-12 [e5a80d8] [mcxstudio2] flat surfaces and a colour per section
* 2026-09-12 [6ee93f9] [mcxstudio2] bigger icons, and twice the type in the navigator
* 2026-09-12 [c2ff716] [mcxstudio2] the icon set goes back to the original artwork
* 2026-09-12 [3143ee3] [mcxstudio2] the accordion navigates, the detail pane holds the settings
* 2026-09-12 [be48a89] [mcxstudio2] do not track build artifacts from the test build
* 2026-09-12 [869f463] [mcxstudio2] the binding table: one row per setting, no serialisers
* 2026-09-12 [d7fecd4] [mcxstudio2] main window: accordion sections and live preview panes
* 2026-09-12 [a14c087] [mcxstudio2] the simulation document and its round-trip tests
* 2026-09-12 [ad17282]*[mcxstudio2] skeleton: project, high-DPI scaling and drawn icons
* 2026-09-01 [8a224ca] [mcxcloud] v2: record the submitting client's IP
* 2026-08-31 [9102cfa] [mcxcloud] v2: raise the preview domain-dimension cap from 300 to 400
* 2026-08-30 [4fd5c3d] [feat] make isrfforward a template constant to recover the lean-kernel register budget
* 2026-08-30 [4e2fd41]*[feat] always use atomic accumulation, drop the non-atomic and pre-Fermi paths
* 2026-08-30 [959109a] [bug] fix correctness bugs and doc drift in the utils MATLAB functions
* 2026-08-30 [585f607] [bug] accumulate wltof/wptof time-of-flight in nanoseconds so the double-buffer engages
* 2026-08-29 [1be8bad] [bug] make -U 2 mean only "skip the detected-weight normalization in replay", #263
* 2026-08-29 [39c026e] [test] add pattern-source photon-sharing replay regression tests
* 2026-08-27 [ba36c98]*[feat] support pattern/pattern3d sources in wp/dcs/wptof/rfmus replay, #259
* 2026-08-27 [8781d36] [bug] apply per-pattern weight to the sine-weighted RF mua Jacobian, #259
* 2026-08-27 [8d571b6] [doc] swap mistakenly reversed descriptions of outputtype t and b
* 2026-08-27 [1877f40] [bug] divide mus in otWPTOF SVMC replay, scale unitinmm in normalization, #263
* 2026-08-23 [3e05491] [mcxcloud] v2 redbird: emit JMesh (.jmsh) + a self-describing measurement file
* 2026-08-23 [651dc45] [mcxcloud] v2: capitalize Redbird (and MCX/MMC) in user-facing text
* 2026-08-23 [de68770] [mcxcloud] v2 redbird: require redbirdpy >= 0.4.2
* 2026-08-23 [4d1a4c7] [mcxcloud] v2 frontend: tint builtin library cards by solver
* 2026-08-23 [bbfe60c] [mcxcloud] v2 redbird: correct inverted element volumes from meshprep
* 2026-08-23 [789cd6b] [mcxcloud] v2 redbird: refuse to publish a non-converged FEM solve
* 2026-08-23 [23f6130] [mcxcloud] v2: purge blanks job content in place instead of deleting rows
* 2026-08-23 [25bc3e4] [mcxcloud] v2: redbird FEM diffusion engine (CPU), disabled by default
* 2026-08-23 [16e1dd6] [mcxcloud] v2 schema: add redbird engine discriminator (rejected for now)
* 2026-08-23 [d53b076] [mcxcloud] v2 schema: unify RF frequency field across mcx/mmc
* 2026-08-22 [e1f6b8f] [mcxcloud] v2 preview: adaptive mesh zoom range + drop misleading mesh bbox
* 2026-08-22 [5521c25] [mcxcloud] v2 backend: GPU dispatch cold-start fix + swarm retry-on-failure
* 2026-08-20 [5576ee0] Revert "fix: correct dephotons to detphotons typo in pmcx/utils.py (closes #268)"
* 2026-08-15 [a375004] fix: correct dephotons to detphotons typo in pmcx/utils.py (closes #268)
* 2026-08-14 [182f233] Update utils.py
* 2026-08-03 [7e53b92] [perf] scope slen to loop body to keep HIP photon loop flat
* 2026-07-22 [145c717] [mcxcloud] v2 preview: mesh edges + interior region-interface surfaces
* 2026-07-20 [1413b51] [mcxcloud] v2: per-node mmc placement + replace mmc demos with a verified slab
* 2026-07-19 [9eb8702] [mcxcloud] v2: fix mmc worker segfault on editor-emitted Detector.Dir
* 2026-07-19 [5a6285c] [mcxcloud] v2 preview: render tet meshes + cross-sections + mmc outputs (Phase C)
* 2026-07-19 [60cf0f1] [mcxcloud] v2 backend: mmc job routing + worker (Phase B)
* 2026-07-19 [aed0069] [mcxcloud] v2 schema: mmc (mesh) support — plan + Phase A validation
* 2026-07-19 [f8daf12] [mcxcloud] v2 library: seed builtin multi-source examples
* 2026-07-19 [21b2cee] [mcxcloud] v2 preview: render 4D+ volumes with a frame selector
* 2026-07-19 [1881132] [mcxcloud] v2 schema: multi-src support + missing mcx json constructs
* 2026-07-18 [1bf46f1] [mcxcloud] v2 frontend: fix shared-link (?data=) load + auto-render
* 2026-07-18 [6ea36b4] [mcxcloud] v2 preview: DVR (MIDA) render mode + contrast control
* 2026-07-18 [00e1f7b] [mcxcloud] v2 Phase 6: library review, purge/rate-limit, preview UX, deploy tooling
* 2026-07-14 [7c09a0e] [mcxcloud] v2 Phase 5: v1 library migration, deploy tooling, private-cloud docs
* 2026-07-14 [5acf578] [mcxcloud] v2 Phase 4: no-build ESM frontend
* 2026-07-14 [1fe3be1] [mcxcloud] v2 Phase 3: event-driven scheduler + push-based worker protocol
* 2026-07-14 [3032cab] [mcxcloud] v2: port backend to plain ESM JS + JSDoc (Node 20+)
* 2026-07-14 [c8879c4] [mcxcloud] v2 Phase 2: Postgres schema + Fastify API (no-build TS)
* 2026-07-14 [277ae10]*[mcxcloud] v2 Phase 1: design doc + data/API contracts
* 2026-07-14 [ebdf88b] [mcxcloud] relocate v1 frontend/backend into v1/ subfolder
* 2026-06-17 [d38c7aa] [ROCm] Enable native FP atomics and fast-math for the HIP build
* 2026-06-16 [a7ecad7]*[ROCm] Add HIP/ROCm support for AMD GPUs
* 2026-05-10 [6d7a81a]*[feat] port -F bnii support from mmc to mcx, rename .jdat to .jdt
* 2026-05-10 [bc5148f] [utils] update mcx2json to match umcxlab
* 2026-05-09 [c24b8bd] [ci] prevent choco checksum error
* 2026-05-09 [d750374] [ci] use cache to optimize ci runtime
* 2026-04-23 [1d44d85] [ci] fix windows binary build
* 2026-04-23 [8c348b0] [ci] fix macos bug
* 2026-04-22 [53dddfc] [ci] fix link error for lzma sdk
* 2026-04-22 [e71c1d6] [ci] fix choco mingw installation error
* 2026-04-22 [ba5491d] [ci] disable multi-threading lzma
* 2026-04-22 [387c7c9] [ci] fix cmake and pmcx build
* 2026-04-22 [da5ad22] [zmat] update zmat to 1.2.preview to use parallel lzma/lzip
* 2026-04-15 [d18407f] [zmat] sync bundled zmat lib to 1.0 release
* 2026-04-06 [f82c0dd] [feat] migrate rfmus replay output to double-buffer to minimize round-off error
* 2026-04-06 [f4836d0] [pmcx] bump version to 0.7.1 to support detdir(4) in adjoint output
* 2026-04-06 [7aaeb06] [demo] mcxlab demo script update for how to set detdir(4)
* 2026-04-06 [9ec176e] [bug] fix unable to use detdir(4) to set detector beam profile
* 2026-04-05 [7483c66] [bug] remove c++11 feature in mcxlab.cpp to allow ci build
* 2026-04-05 [ad0f881] [bug] fix pattern3d in rf-replay, fix #259, ref #255
* 2026-04-05 [06212d2] [bug] fix replaydet=-1 fails to replay in nscat, wp, wptof and rfmus, #255
* 2026-04-05 [74f4790] [bug] divide mus during ray-tracing in otRFmus and otWP in replay
* 2026-04-05 [5eea2af] [bug] fix 2^24 cap in otWP/otDCS/otWPTOF output, reported by Pauliina Hirvi
* 2026-04-05 [2b7f5d4]*[feat] save/export both forward and Jacobians in the adjoint mode
* 2026-04-02 [0426e28] [bug] fix mcxplotshapes bug for spheres
* 2026-04-02 [4b349cf] [bug] fix line/slit source preview
* 2026-04-01 [537ce24] [bug] update pmcx to corectly cast detp.nscat to uint32 to keep precision
* 2026-04-01 [46aa8a7] [feat] make detp.nscat buffer unsigned integer to avoid float 16777216 cap
* 2026-04-01 [dffe4ff] [bug] fix matlab compilation error for MX_HAS_INTERLEAVED_COMPLEX
* 2026-04-01 [6863caa] [mex] support interleaved complex number output in matlab r2018+
* 2026-03-31 [d802d78] [feat] make mcxplotphotons support traj output data
* 2026-03-27 [08b1d45]*[feat] export dual Jacobians for mua and D or musp
* 2026-03-20 [53e3895] [doc] partial update of the changelog
* 2026-03-20 [ff1cc2a] [doc] update help info regarding adjoint normalization
* 2026-03-17 [f54d448]*[feat] add adjoint_dcoeff, adjoint_mus and adjoint_musp Jacobian outputs
* 2026-03-15 [8c248b7] [pmcx] bump pmcx version to 0.7.0 for newly added rf-forward and adjoint output
* 2026-03-15 [fb15b1a] [bug] fix adjoint output normalization, sign, and test script bugs
* 2026-03-15 [8dc6071]*[feat] support adjoint output type, build adjoint jacobian, srcid=-2 solves src+det
* 2026-03-14 [b22534c]*[feat] support RF forward, use Eq 9 in Leino/Pulkkinen/Tarvainen 2019
* 2026-03-05 [22a058d] [ci] fix mcxshow build error
* 2026-03-05 [c911f5d] [ci] fix mcxstudio build error
* 2026-03-05 [e001ef5] [ci] fix lazarus installation on macos
* 2026-03-05 [12cf789] [ci] debug lazarus installation on macos
* 2026-03-05 [8074f5d] [ci] try again to fix macos-15 lazarus installation error
* 2026-03-05 [11ccc9c] [ci] fix macos-15 lazarus installation error
* 2026-03-05 [8a7731f]*[feat] improve shape editor to allow select and scale shapes
* 2026-02-17 [857b26d] [svmc] svmc optimization, reduces 2 regigsters, 10% speed gain
* 2026-02-16 [71b3d6c] [bug] fix double pre-processing when using -j/--json flag
* 2026-01-11 [89627a7] [feat/bug] extends line src for wedge/fan beam, fix #253 for line/slit
* 2025-12-16 [2fdd3d8] [ci] test solution to macos-15
* 2025-12-15 [bff56e7] [preview] render noderoi
* 2025-12-10 [cc756d7] [ci] remove retired macos-13 runner, use macos-15-intel
* 2025-12-10 [b6b8573] [pmcx] remove sparse_numba as a dependency on windows
* 2025-10-21 [fbe3de0] [winget] update installer after fixing mmc bug
* 2025-10-14 [4f953aa] [doc] fix typos
* 2025-10-13 [5715c2b] [winget] fix mismatch of schema version
* 2025-10-13 [6c4f29f] [winget] update installer hash
* 2025-10-13 [14b933f] [doc] update README

Introduction
---------------

Monte Carlo eXtreme (MCX) is a fast physically-accurate photon simulation 
software for 3D heterogeneous complex media. By taking advantage of 
the massively parallel threads and extremely low memory latency in a 
modern graphics processing unit (GPU), this program is able to perform Monte 
Carlo (MC) simulations at a blazing speed, typically hundreds to
a thousand times faster than a single-threaded CPU-based MC implementation.

MCX is written in C and NVIDIA CUDA, and runs on NVIDIA GPUs. The same CUDA
source can also be compiled for AMD GPUs through HIP/ROCm by configuring with
`-DUSE_HIP=ON` (see Requirement and Installation below). If you want to run
hardware-accelerated MCX simulations on Intel GPUs or CPUs, please download
MCX-CL (MCX for OpenCL), which is written in OpenCL. MCX and MCX-CL are highly
compatible.

Due to the nature of the underlying MC algorithms, MCX and MCX-CL are
ray-tracing/ray-casting software under-the-hood. Compared to commonly
seen ray-tracing libraries used in computer graphics or gaming
engines, MCX-CL and MCX have many unique characteristics. The most
important difference is that MCX/MCX-CL are rigorously based on physical
laws. They are numerical solvers to the underlying radiative transfer equation
(RTE) and their solutions have been validated across many publications
using optical instruments and experimental measurements. In comparison,
most graphics-oriented ray-tracers have to make many approximations in
order to achieve fast rendering, and are unable to provide quantitatively accurate
light simulation results. Because of this, MCX/MCX-CL have been extensively
used by biophotonics research communities to obtain reference solutions and
guide the development of novel medical imaging systems or clinical
applications. Additionally, MCX/MCX-CL are volumetric ray-tracers; they
traverse photon-rays throughout complex 3-D domains and compute physically
meaningful quantities such as spatially resolved fluence, flux, diffuse
reflectance/transmittance, energy deposition, partial pathlengths,
among many others. In contrast, most graphics ray-tracing engines
only trace the RGB color of a ray and render it on a flat 2-D screen.
In other words, MCX/MCX-CL gives physically accurate 3-D light distributions
while graphics ray-tracers focus on 2-D rendering of a scene at the camera.
Nonetheless, they share many similarities, such as ray-marching computation,
GPU acceleration, scattering/absorption handling etc.

The algorithm of this software is detailed in the References 
[Fang2009,Yu2018,Yan2020]. A short summary of the main features includes:

-   3D heterogeneous media represented by voxelated array
-   support complex sources including wide-field and pattern illuminations
-   boundary reflection support
-   time-resolved photon transport simulations
-   saving photon partial path lengths and trajectories
-   optimized random number generators
-   build-in flux/fluence normalization to output Green's functions
-   user adjustable voxel resolution
-   improved accuracy with atomic operations
-   cross-platform graphical user interface
-   native Matlab/Octave support for high usability
-   flexible JSON interface for future extensions
-   multi-GPU support
-   advanced features: photon-replay, photon-sharing, and more
-   frequency-domain (RF) simulations and adjoint/replay-based Jacobians for
    image reconstruction
-   NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs, with MATLAB/Octave (MCXLAB) and
    Python (PMCX) interfaces

This software can be used on Windows, Linux and macOS. MCX is written in C/CUDA
and runs on NVIDIA GPUs, or on AMD GPUs when built with HIP/ROCm. A more portable
OpenCL implementation of MCX, MCX-CL, was first released in July 2012 and supports
almost all NVIDIA/AMD/Intel CPU and GPU models, including Apple silicon. If your
hardware supports neither CUDA nor ROCm, please download MCX-CL from the below URL:

https://mcx.space/wiki/index.cgi?Learn#mcxcl


Requirement and Installation
-------------------------------

Please read this section carefully. The majority of failures using MCX were 
found related to incorrect installation of NVIDIA GPU driver.

Please browse <https://mcx.space/#documentation> for step-by-step instructions.

For MCX-CUDA, the requirements for using this software include

-   a CUDA capable NVIDIA graphics card
-   pre-installed NVIDIA graphics driver

MCX can also be built for AMD GPUs using HIP/ROCm. In that case the
requirements are

-   a ROCm capable AMD graphics card
-   a pre-installed AMD ROCm toolkit (including HIP)

To compile the AMD build from source, configure the CMake project with HIP
enabled and select the target GPU architecture, for example

```
cmake -S src -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build
```

`CMAKE_HIP_ARCHITECTURES` accepts any ROCm GPU target (for example `gfx90a`
or `gfx1100`); if omitted it defaults to `gfx90a`. The default NVIDIA build
(`USE_HIP=OFF`) is unchanged.

You must make sure that your NVIDIA graphics driver was installed properly.
A list of CUDA-capable cards can be found at [2]. The oldest 
GPU architecture that MCX source code can be compiled is Fermi (`sm_20`).
Using the latest NVIDIA card is expected to produce the best
speed. The officially released binaries (including mex files and `pmcx` modules)
can run on NVIDIA GPUs as old as Kepler (GTX-730, `sm_35`). All MCX binaries
can run directly on future generations of NVIDIA GPUs without needing to
be recompiled, therefore forward-compatible.

In the below webpage, we summarized the speed differences
between different generations of NVIDIA GPUs

https://mcx.space/gpubench/

For simulations with large volumes, sufficient graphics memory is also required 
to perform the simulation. The minimum amount of graphics memory required for a 
MC simulation is Nx\*Ny\*Nz bytes for the input tissue data plus 
Nx\*Ny\*Nz\*Ng\*4\*2 bytes for the output flux/fluence data - where Nx,Ny,Nz are 
the dimensions of the tissue volume, Ng is the number of concurrent time gates, 
4 is the size of a single-precision floating-point number, 2 is for the extra memory
needed to ensure output accuracy (https://github.com/fangq/mcx/issues/41). MCX does not require 
double-precision support in your hardware.

MCX stores optical properties and detector positions in the constant memory.
NVIDIA GPUs usually provide 64 kB of constant memory. As a result, the total
number of optical properties plus the number of detectors can not exceed 4000
(4000 * 4 * 4 = 64 k).

In addition, MCX stores the partial path data of the photons being tracked in the
shared memory, which ranges between 48 kB and 100 kB per streaming multiprocessor
across GPU generations. If your domain contains many medium types, MCX reduces the
thread block size to fit the limit.

To install MCX, you need to download the binary executable compiled for your 
computer architecture and platform, extract the package and run 
the executable under the `{mcx root}/bin` directory. MCX can be obtained in several ways

- the MCX download page (https://mcx.space/wiki/?Get) and the GitHub release page
  (https://github.com/fangq/mcx/releases) provide the MCX and MCXLAB packages,
  as well as the all-in-one "MCX Studio" packages for Linux, macOS (x86_64 and
  arm64) and Windows, which bundle the simulators (mcx, mcxcl and mmc, as
  supported on each platform), the MCX Studio GUI and the MATLAB/Octave
  toolboxes; the Redbird (MATLAB/Octave) and Redbirdpy (Python) image
  reconstruction toolboxes are provided as a separate package
- the latest automatically built packages can be found at https://mcx.space/nightly/
- on Windows, MCX Studio can be installed with an installer, or by
  `winget install coti.mcxstudio`
- Python users can install `pmcx` (MCX), `pmcxcl` (MCX-CL) and `pmmc` (MMC) via
  `pip`, see the "Using PMCX in Python" section below
- a Docker image is available as `fangqq/mcx` on Docker Hub (requires the
  NVIDIA Container Toolkit to access the GPU)

The released Linux binaries and mex files are built against glibc 2.17
(manylinux2014), and run on most Linux distributions released since 2014. The
released Octave mex files for Windows and macOS are linked against `liboctmex`
and require Octave 10 or newer; on Linux, they work with any recent Octave.

For Windows users, you must make sure you have installed the appropriate NVIDIA 
driver for your GPU. You should also configure your OS to run CUDA simulations. 
This requires you to open the mcx/setup/win64 folder using your file explorer, 
right-click on the `apply_timeout_registry_fix.bat` file and select 
**“Run as administrator”**. After confirmation, you should see a windows 
command window with message

      Patching your registry
      Done
      Press any key to continue ...

You MUST REBOOT your Windows computer to make this setting effective. The above 
patch modifies your driver settings so that you can run MCX simulations for 
longer than a few seconds. Otherwise, when running MCX for over a few seconds, 
you will get a CUDA error: “unspecified error”.

Please see the below link for details

https://mcx.space/wiki/index.cgi?Doc/FAQ#I_am_getting_a_kernel_launch_timed_out_error_what_is_that

If you use Linux, you may enable Intel integrated GPU (iGPU) for display while
leaving your NVIDIA GPU dedicated for computing using `nvidia-prime`, see

https://forums.developer.nvidia.com/t/solved-run-cuda-on-dedicated-nvidia-gpu-while-connecting-monitors-to-intel-hd-graphics-is-this-possible/47690/6

or choose one of the 4 other approaches in this blog post

https://nvidia.custhelp.com/app/answers/detail/a_id/3029/~/using-cuda-and-x

We noticed that running Ubuntu Linux 22.04 with a 6.5 kernel on a laptop with 
a hybrid GPU with an Intel iGPU and an NVIDIA GPU, you must configure the
laptop to use the NVIDIA GPU as the primary GPU by choosing "NVIDIA (Performance Mode)"
in the PRIME Profiles section of **NVIDIA X Server Settings**. You can also run 

```
sudo prime-select nvidia
```

to achieve the same goal. Otherwise, the simulation may hang your system
after running for a few seconds. A hybrid GPU laptop combing an NVIDIA GPU 
with an AMD iGPU does not seem to have this issue if using Linux.

In addition, the NVIDIA driver (520 or newer) has a known glitch running on Linux kernel
6.x (such as those in Ubuntu 22.04). See

https://forums.developer.nvidia.com/t/dev-nvidia-uvm-io-error-on-ubuntu-22-04-520-to-535-driver-versions/262153

When the laptop is in the "performance" mode and wakes up from suspension, MCX or any
CUDA program fails to run with an error


```
MCX ERROR(-999):unknown error in unit mcx_core.cu:2523
```

This is because the kernel module `nvidia-uvm` fails to be reloaded after suspension.
If you had an open MATLAB session, you must close MATLAB first, and
run the below commands (if MATLAB is open, you will see `rmmod: ERROR: Module nvidia_uvm is in use`)

```
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm
```

after the above command, MCX should be able to run again.

New generations of Mac computers no longer support NVIDIA or AMD GPUs. You will
have to use the OpenCL version of MCX, MCX-CL, which runs on Apple silicon GPUs; download it from

https://mcx.space/wiki/?Learn#mcxcl


Running Simulations
----------------------

To run a simulation, the minimum input is a configuration (text) file, and, if
the input file does not contain built-in domain shape descriptions, an external
volume file (a binary file with a specified voxel format via `-K/--mediabyte`). 
Typing `mcx` without any parameters prints the help information and a list of 
supported parameters, as shown below:

```
###############################################################################
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #
#          Copyright (c) 2009-2026 Qianqian Fang <q.fang at neu.edu>          #
#                https://mcx.space/  &  https://neurojson.io                  #
#                                                                             #
# Computational Optics & Translational Imaging (COTI) Lab- https://fanglab.org #
#   Department of Bioengineering, Northeastern University, Boston, MA, USA    #
###############################################################################
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #
###############################################################################
#  Open-source codes and reusable scientific data are essential for research, #
# MCX proudly developed human-readable JSON-based data formats for easy reuse.#
#                                                                             #
#Please visit our free scientific data sharing portal at https://neurojson.io #
# and consider sharing your public datasets in standardized JSON/JData format #
###############################################################################
$Rev::d20953$v2026$Date::2025-10-10 23:08:18 -04$ by $Author::Qianqian Fang$
###############################################################################

usage: mcx <param1> <param2> ...
where possible parameters include (the first value in [*|*] is the default)

== Required option ==
 -f config     (--input)       read an input file in the .json format,if config
                               string starts with '{',it is parsed as an inline
                               JSON input file; if -f is followed by nothing or
                               a single '-', it reads input from stdin via pipe
      or
 -Q/--bench [cube60, skinvessel,...] run a built-in benchmark specified by name
                               run -Q without parameter to get a list
 -N benchmark  (--net)         get benchmark from NeuroJSON.io, -N only to list
                               benchmark can be dataset URL,or dbname/benchname
                               requires 'curl', install from https://curl.se/

== MC options ==
 -n [0|int]    (--photon)      total photon number (exponential form accepted)
                               max accepted value:9.2234e+18 on 64bit systems
 -r [1|+/-int] (--repeat)      if positive, repeat by r times,total= #photon*r
                               if negative, divide #photon into r subsets
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit
 -B '______'   (--bc)          per-face boundary condition (BC), 6 letters for
    /case insensitive/         bounding box faces at -x,-y,-z,+x,+y,+z axes;
                               overwrite -b if given.
                               each letter can be one of the following:
                               '_': undefined, fallback to -b
                               'r': like -b 1, Fresnel reflection BC
                               'a': like -b 0, total absorption BC
                               'm': mirror or total reflection BC
                               'c': cyclic BC, enter from opposite face

                               if input contains additional 6 letters,
                               the 7th-12th letters can be:
                               '0': do not use this face to detect photon, or
                               '1': use this face for photon detection (-d 1)
                               the order of the faces for letters 7-12 is
                               the same as the first 6 letters
                               eg: --bc ______010 saves photons exiting at y=0
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge
 -U [1|0|2]    (--normalize)   1 to normalize flux to unitary; 0 save raw;
                               2 behaves as 1, except in replay mode, where
                               the normalization by the total detected weight
                               is skipped and only the unit conversion applies
 -E [1648335518|int|mch](--seed) set rand-number-generator seed, -1 to generate
                               if an mch file is followed, MCX "replays"
                               the detected photon; the replay mode can be used
                               to calculate the mua/mus Jacobian matrices
 -z [0|1]      (--srcfrom0)    1 volume origin is [0 0 0]; 0: origin at [1 1 1]
 -k [1|0]      (--voidtime)    when src is outside, 1 enables timer inside void
 -Y [0|int]    (--replaydet)   replay only the detected photons from a given
                               detector (det ID starts from 1), used with -E
                               if 0, replay all detectors and sum all Jacobians
                               if -1, replay all detectors and save separately
 -V [0|1]      (--specular)    1 source located in the background,0 inside mesh
 -e [0.|float] (--minenergy)   minimum energy level to trigger Russian roulette
 -g [1|int]    (--gategroup)   number of maximum time gates per run

== GPU options ==
 -L            (--listgpu)     print GPU information only
 -t [16384|int](--thread)      total thread number
 -T [64|int]   (--blocksize)   thread number per block
 -A [1|int]    (--autopilot)   1 let mcx decide thread/block size, 0 use -T/-t
                               autopilot also dispatches photons dynamically,
                               so runs with the same seed agree statistically
                               but not bit-wise; use -A 0 to reproduce a run
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto
      or
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum
 -I            (--printgpu)    print GPU information and run program
 --atomic [1|0]                deprecated and ignored; atomic accumulation is
                               always used, non-atomic mode raced and lost data

== Input options ==
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid.
                               only the root object named 'Shapes' is parsed
                               and added to the existing domain defined via -f
                               or --bench
 -j '{...}'    (--json)        a JSON string for modifying all input settings.
                               this input can be used to modify all existing
                               settings defined by -f or --bench
 -K [1|int|str](--mediabyte)   volume data format, use either a number or a str
       voxel binary data layouts are shown in {...}, where [] for byte,[i:]
       for 4-byte integer, [s:] for 2-byte short, [h:] for 2-byte half float,
       [f:] for 4-byte float; on Little-Endian systems, least-sig. bit on left
                               1 or byte: 0-128 tissue labels
                               2 or short: 0-65535 (max to 4000) tissue labels
                               4 or integer: integer tissue labels
                              96 or asgn_float: mua/mus/g/n 4xfloat format
                                {[f:mua][f:mus][f:g][f:n]}
                              97 or svmc: split-voxel MC 8-byte format
                                {[n.z][n.y][n.x][p.z][p.y][p.x][upper][lower]}
                              98 or mixlabel: label1+label2+label1_percentage
                                {[label1][label2][s:0-32767 label1 percentage]}
                              99 or labelplus: 32bit composite voxel format
                                {[h:mua/mus/g/n][s:(B15-16:0/1/2/3)(label)]}
                             100 or muamus_float: 2x 32bit floats for mua/mus
                                {[f:mua][f:mus]}; g/n from medium type 1
                             101 or mua_float: 1 float per voxel for mua
                                {[f:mua]}; mus/g/n from medium type 1
                             102 or muamus_half: 2x 16bit float for mua/mus
                                {[h:mua][h:mus]}; g/n from medium type 1
                             103 or asgn_byte: 4-byte gray-levels for mua/s/g/n
                                {[mua][mus][g][n]}; 0-255 mixing prop types 1&2
                             104 or muamus_short: 2-short gray-levels for mua/s
                                {[s:mua][s:mus]}; 0-65535 mixing prop types 1&2
       when formats 99 or 102 is used, the mua/mus values in the input volume
       binary data must be pre-scaled by voxel size (unitinmm) if it is not 1.
       pre-scaling is not needed when using these 2 formats in mcxlab/pmcx
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array

== Output options ==
 -s sessionid  (--session)     a string to label all output file names
 -O [X|XFEJPMRLSTBADCUVWQ](--outputtype) X - output flux, F - fluence, E - energy
    /case insensitive/         J - Jacobian (replay mode),   P - scattering,
                               event counts at each voxel (replay mode only)
                               M - momentum transfer; L - total pathlength
                               R - RF/FD mua Jacobian (replay, needs -E)
                                   or RF forward complex fluence (forward,
                                   without -E, needs omega); output is complex
                               S - RF/FD mus Jacobian (replay mode only)
                               T - time-of-flight*path;B - time-of-flight*nscat
                               A - adjoint mua Jacobian (needs detpos+detdir)
                               D - adjoint D-coeff Jacobian (needs detpos+dir)
                               U - adjoint mus Jacobian (grad*grad*3*D^2*(1-g))
                               V - adjoint mus' Jacobian (grad*grad*3*D^2)
                               W - dual adjoint [J_mua,J_D] pair in 1 session
                               Q - dual adjoint [J_mua,J_mus'] pair in 1 session
 -d [1|0-3]    (--savedet)     1 to save photon info at detectors; 0 not save
                               2 reserved, 3 terminate simulation when detected
                               photon buffer is filled
 -w [DP|DSPMXVW](--savedetflag)a string controlling detected photon data fields
    /case insensitive/         1 D  output detector ID (1)
                               2 S  output partial scat. even counts (#media)
                               4 P  output partial path-lengths (#media)
                               8 M  output momentum transfer (#media)
                              16 X  output exit position (3)
                              32 V  output exit direction (3)
                              64 W  output initial weight (1)
      combine multiple items by using a string,or add selected numbers together
      by default, mcx only saves detector ID and partial-path data
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions
                               setting -x to 1 also implies setting '-d' to 1.
                               same as adding 'XV' to -w.
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels
                               right outside of the domain; if non-zero voxels
                               appear at the boundary, pad 0s before using -X
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save
                               same as adding 'M' to the -w flag
 -q [0|1]      (--saveseed)    1 to save photon RNG seed for replay; 0 not save
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save
 -H [1000000] (--maxdetphoton) max number of detected photons
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save
 -F [jnii|...](--outputformat) fluence data output format:
                               mc2 - MCX mc2 format (binary 32bit float)
                               jnii - JNIfTI format (https://neurojson.org)
                               bnii - Binary JNIfTI (https://neurojson.org)
                               nii - NIfTI format
                               hdr - Analyze 7.5 hdr/img format
                               tx3 - GL texture data for rendering (GL_RGBA32F)
    the bnii/jnii formats support compression (-Z) and generate small files
    load jnii (JSON) and bnii (BJData) files using below lightweight libs:
      MATLAB/Octave: JNIfTI toolbox   https://neurojson.org/download/jnifty
      MATLAB/Octave: JSONLab toolbox  https://neurojson.org/download/jsonlab
      Python:        PyJData:         https://neurojson.org/download/pyjdata
      JavaScript:    JSData:          https://neurojson.org/download/jsdata
 -Z [zlib|...] (--zip)         set compression method if -F jnii or --dumpjson
                               is used (when saving data to JSON/JNIfTI format)
                               0 zlib: zip format (moderate compression,fast)
                               1 gzip: gzip format (compatible with *.gz)
                               2 base64: base64 encoding with no compression
                               3 lzip: lzip format (high compression,very slow)
                               4 lzma: lzma format (high compression,very slow)
                               5 lz4: LZ4 format (low compression,extrem. fast)
                               6 lz4hc: LZ4HC format(moderate compression,fast)
 --dumpjson [-,0,1,'file.json'] export all settings,including volume data using
                               JSON/JData (https://neurojson.org) format for
                               easy sharing; can be reused using -f
                               if followed by nothing or '-', mcx will print
                               the JSON to the console; write to a file if file
                               name is specified; by default, prints settings
                               after pre-processing; '--dumpjson 2' prints
                               raw inputs before pre-processing

== User IO options ==
 -h            (--help)        print this message
 -y [zh_CN,..] (--lang)        select language, followed by nothing to print
 -v            (--version)     print MCX revision number
 -l            (--log)         print messages to a log file instead
 -i            (--interactive) interactive mode

== Debug options ==
 -D [0|int]    (--debug)       print debug information (you can use an integer
  or                           or a string by combining the following flags)
 -D [''|RMPT]                  1 R debug RNG
    /case insensitive/         2 M store photon trajectory info
                               4 P print progress bar
                               8 T save trajectory data only, disable flux/detp
      combine multiple items by using a string,or add selected numbers together

== Additional options ==
 --root         [''|string]    full path to the folder storing the input files
 --gscatter     [1e9|int]      after a photon completes the specified number of
                               scattering events, mcx then ignores anisotropy g
                               and only performs isotropic scattering for speed
 --srcid  [0|-1,0,1,2,..]     -1 simulate multi-source separately;0 all sources
                               together;a positive integer runs a single source
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch
 --trajstokes   [0|1]          set to 1 to save Stokes IQUV in trajectory data
 --maxvoidstep  [1000|int]     max distance (in voxel unit) of a photon that
                               can travel before entering the domain, if
                               launched outside (i.e. a widefield source)
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),
                               use this parameter to set the maximum positions
                               stored (default: 1e7)

== Example ==
example: (list built-in benchmarks: -Q/--bench)
       mcx -Q
or (list supported GPUs on the system: -L/--listgpu)
       mcx -L
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)
       mcx -Q cube60b -n 1e7 -G 1101 -W 10,10,10
or (use inline domain definition)
       mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
or (use inline json setting modifier)
       mcx -f input.json -j '{"Optode":{"Source":{"Type":"isotropic"}}}'
or (dump simulation in a single json file)
       mcx -Q cube60planar --dumpjson
or (use -N/--net to browse community-contributed mcx simulations at https://neurojson.io)
       mcx -N
or (run user-shared mcx simulations, see full list at https://neurojson.org/db/mcx)
       mcx -N aircube60
or (print in simplified Chinese using -y/--lang)
       mcx -y zh_CN -Q cube60
or (use -f - to read piped input file modified by shell text processing utilities)
       mcx -Q cube60 --dumpjson | sed -e 's/pencil/cone/g' | mcx -f -
or (download/modify simulations from NeuroJSON.io and run with mcx -f)
       curl -s -X GET https://neurojson.io:7777/mcx/aircube60 | jq '.Forward.Dt = 1e-9' | mcx -f
```

To further illustrate the command line options, below one can find a sample command
```
mcx -A 0 -t 16384 -T 64 -n 1e7 -G 1 -f input.json -r 2 -s test -g 10 -d 1 -w dpx -b 1
```
the command above asks mcx to manually (`-A 0`) set GPU threads, and launch 16384 
GPU threads (`-t`) with every 64 threads a block (`-T`); a total of 1e7 photons (`-n`)
are simulated by the first GPU (`-G 1`) and repeat twice (`-r`) - i.e. total 2e7 photons;
the media/source configuration will be read from a JSON file named `input.json` 
(`-f`) and the output will be labeled with the session id “test” (`-s`); the 
simulation will run 10 concurrent time gates (`-g`) if the GPU memory can not 
simulate all desired time gates at once. Photons passing through the defined 
detector positions are saved for later rescaling (`-d`); refractive index 
mismatch is considered at media boundaries (`-b`).

Historically, MCX supports an extended version of the input file format (.inp)
used by tMCimg. However, we are phasing out the .inp support and strongly 
encourage users to adopt JSON formatted (.json) input files. Many of the 
advanced MCX options are only supported in the JSON input format.

A legacy .inp MCX input file looks like this:

```
1000000              # total photon, use -n to overwrite in the command line
29012392             # RNG seed, negative to generate, use -E to overwrite
30.0 30.0 0.0 1      # source position (in grid unit), the last num (optional) sets --srcfrom0 (-z)
0 0 1 0              # initial directional vector, 4th number is the focal-length, 0 for collimated beam, nan for isotropic, -inf for Lambertian
0.e+00 1.e-09 1.e-10 # time-gates(s): start, end, step
semi60x60x60.bin     # volume ('unsigned char' binary format, or specified by -K/--mediabyte)
1 60 1 60            # x voxel size in mm (isotropic only), dim, start/end indices
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices 
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices
1                    # num of media
1.010101 0.01 0.005 1.37  # scat. mus (1/mm), g, mua (1/mm), n
4       1.0          # detector number and default radius (in grid unit)
30.0  20.0  0.0  2.0 # detector 1 position (real numbers in grid unit) and individual radius (optional)
30.0  40.0  0.0      # ..., if individual radius is ignored, MCX will use the default radius
20.0  30.0  0.0      #
40.0  30.0  0.0      # 
pencil               # source type (optional)
0 0 0 0              # parameters (4 floats) for the selected source
0 0 0 0              # additional source parameters
```

Note that the scattering coefficient mus=musp/(1-g).

The volume file (`semi60x60x60.bin` in the above example), can be read in two 
ways by MCX: row-major[3] or column-major depending on the value of the user 
parameter `-a`. If the volume file was saved using matlab or fortran, the 
byte order is column-major, and you should use `-a 0` or leave it out of 
the command line. If it was saved using the `fwrite()` in C, the order is 
row-major, and you should use `-a 1`.

You may replace the binary volume file by a JSON-formatted shape file. Please 
refer to Section V for details.

The time gate parameter is specified by three numbers: start time, end time and 
time step size (in seconds). In the above example, the configuration specifies 
a total time window of [0 1] ns, with a 0.1 ns resolution. That means the 
total number of time gates is 10.

MCX provides an advanced option, -g, to run simulations when the GPU memory is 
limited. It specifies how many time gates to simulate concurrently. Users may 
want to limit that number to less than the total number specified in the input 
file - and by default it runs one gate at a time in a single simulation. But if 
there's enough memory based on the memory requirement in Section II, you can 
simulate all 10 time gates (from the above example) concurrently by using 
`-g 10` in which case you have to make sure the video card has at least 
60\*60\*60\*10\*5=10MB of free memory. If you do not include the `-g`, MCX will 
assume you want to simulate just 1 time gate at a time. If you specify a  
time-gate number greater than the total number in the input file, (e.g, 
`-g 20`) MCX will stop when the 10 time-gates are completed. If you use the 
autopilot mode (`-A`), then the time-gates are automatically estimated for you.


Using JSON-formatted input files
-----------------------------------

Starting from version 0.7.9, MCX accepts a JSON-formatted input file in 
addition to the conventional tMCimg-like input format. JSON (JavaScript Object 
Notation) is a portable, human-readable and “fat-free” text format to 
represent complex and hierarchical data. Using the JSON format makes an input  
file self-explanatory, extensible and easy-to-interface with other applications 
(like MATLAB).

A sample JSON input file can be found under the example/quicktest folder. The  
same file, `qtest.json`, is also shown below:
```
{
    "Help": {
      "[en]": {
        "Domain::VolumeFile": "file full path to the volume description file, can be a binary or JSON file",
        "Domain::Dim": "dimension of the data array stored in the volume file",
        "Domain::OriginType": "similar to --srcfrom0, 1 if the origin is [0 0 0], 0 if it is [1.0,1.0,1.0]",
	"Domain::LengthUnit": "define the voxel length in mm, similar to --unitinmm",
        "Domain::Media": "the first medium is always assigned to voxels with a value of 0 or outside of
                         the volume, the second row is for medium type 1, and so on. mua and mus must 
                         be in 1/mm unit",
        "Session::Photons": "if -n is not specified in the command line, this defines the total photon number",
        "Session::ID": "if -s is not specified in the command line, this defines the output file name stub",
        "Forward::T0": "the start time of the simulation, in seconds",
        "Forward::T1": "the end time of the simulation, in seconds",
        "Forward::Dt": "the width of each time window, in seconds",
        "Optode::Source::Pos": "the grid position of the source, can be non-integers, in grid unit",
        "Optode::Detector::Pos": "the grid position of a detector, can be non-integers, in grid unit",
        "Optode::Source::Dir": "the unitary directional vector of the photon at launch",
        "Optode::Source::Type": "source types, must be one of the following: 
                   pencil,isotropic,cone,gaussian,planar,pattern,fourier,arcsine,disk,fourierx,fourierx2d,
		   zgaussian,line,slit,pencilarray,pattern3d; in 2D domains (one Dim is 1), disk is a line
		   segment of half-length Param1[0] perpendicular to Dir in the simulation plane",
        "Optode::Source::Param1": "source parameters, 4 floating-point numbers",
        "Optode::Source::Param2": "additional source parameters, 4 floating-point numbers"
      }
    },
    "Domain": {
	"VolumeFile": "semi60x60x60.bin",
        "Dim":    [60,60,60],
        "OriginType": 1,
	"LengthUnit": 1,
        "Media": [
             {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
             {"mua": 0.005,"mus": 1.0, "g": 0.01, "n": 1.0}
        ]
    },
    "Session": {
	"Photons":  1000000,
	"RNGSeed":  29012392,
	"ID":       "qtest"
    },
    "Forward": {
	"T0": 0.0e+00,
	"T1": 5.0e-09,
	"Dt": 5.0e-09
    },
    "Optode": {
	"Source": {
	    "Pos": [29.0, 29.0, 0.0],
	    "Dir": [0.0, 0.0, 1.0],
	    "Type": "pencil",
	    "Param1": [0.0, 0.0, 0.0, 0.0],
	    "Param2": [0.0, 0.0, 0.0, 0.0]
	},
	"Detector": [
	    {
		"Pos": [29.0,  19.0,  0.0],
		"R": 1.0
	    },
            {
                "Pos": [29.0,  39.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [19.0,  29.0,  0.0],
                "R": 1.0
            },
            {
                "Pos": [39.0,  29.0,  0.0],
                "R": 1.0
            }
	]
    }
}
```

A JSON input file requires several root objects, namely `Domain`, 
`Session`, `Forward` and `Optode`. Other root sections, like 
`Help`, will be ignored. Each object is a data structure providing 
information indicated by its name. Each object can contain various sub-fields. 
The orders of the fields in the same level are flexible. For each field, you 
can always find the equivalent fields in the `*.inp` input files. For example, 
The `VolumeFile` field under the `Domain` object is the same as Line\#6 
in `qtest.inp`; the `RNGSeed` under `Session` is the same as Line\#2; the 
`Optode.Source.Pos` is the same as the triplet in Line\#3; the 
`Forward.T0` is the same as the first number in Line\#5, etc.

An MCX JSON input file must be a valid JSON text file. You can validate your 
input file by running a JSON validator, for example <http://jsonlint.com/> You 
should always use "" to quote a “name” and separate parallel items by 
“,”.

MCX accepts an alternative form of JSON input, but using it is not recommended. 
In the alternative format, you can use “`rootobj_name.field_name`”`: value` 
to represent any parameter directly in the root level. For example

    {
        "Domain.VolumeFile": "semi60x60x60.json",
        "Session.Photons": 10000000,
        ...
    }

You can even mix the alternative format with the standard format. If any input 
parameter has values in both formats in a single input file, the 
standard-formatted value has higher priority.

To invoke the JSON-formatted input file in your simulations, you can use the 
`-f` command line option with MCX, just like using an `.inp` file. For 
example:

      mcx -A 1 -n 20 -f onecube.json -s onecubejson

The input file must have a `.json` suffix in order for MCX to recognize. If 
the input information is set in both command line, and input file, the command 
line value has higher priority (this is the same for `.inp` input files). For 
example, when using `-n 20`, the value set in `Session`/`Photons`
is overwritten to 20; when using `-s onecubejson`, the 
`Session`/`ID` value is modified. If your JSON input file is invalid, 
MCX will quit and point out where the format is incorrect.


Using JSON-formatted shape description files
-----------------------------------------------

Starting from v0.7.9, MCX can also use a shape description file in the place of 
the volume file. Using a shape-description file can save you from making a 
binary `.bin` volume. A shape file uses more descriptive syntax and can be easily 
understood and shared with others.

Samples on how to use the shape files are included under the example/shapetest 
folder.

The sample shape file, `shapes.json`, is shown below:
```
{
  "MCX_Shape_Command_Help":{
     "Shapes::Common Rules": "Shapes is an array object. The Tag field sets the voxel value for each
         region; if Tag is missing, use 0. Tag must be smaller than the maximum media number in the
         input file.Most parameters are in floating-point (FP). If a parameter is a coordinate, it
         assumes the origin is defined at the lowest corner of the first voxel, unless user overwrite
         with an Origin object. The default origin of all shapes is initialized by user's --srcfrom0
         setting: if srcfrom0=1, the lowest corner of the 1st voxel is [0,0,0]; otherwise, it is [1,1,1]",
     "Shapes::Name": "Just for documentation purposes, not parsed in MCX",
     "Shapes::Origin": "A floating-point (FP) triplet, set coordinate origin for the subsequent objects",
     "Shapes::Grid": "Recreate the background grid with the given dimension (Size) and fill-value (Tag)",
     "Shapes::Sphere": "A 3D sphere, centered at C0 with radius R, both have FP values",
     "Shapes::Box": "A 3D box, with lower corner O and edge length Size, both have FP values",
     "Shapes::SubGrid": "A sub-section of the grid, integer O- and Size-triplet, inclusive of both ends",
     "Shapes::XLayers/YLayers/ZLayers": "Layered structures, defined by an array of integer triples:
          [start,end,tag]. Ends are inclusive in MATLAB array indices. XLayers are perpendicular to x-axis, and so on",
     "Shapes::XSlabs/YSlabs/ZSlabs": "Slab structures, consisted of a list of FP pairs [start,end]
          both ends are inclusive in MATLAB array indices, all XSlabs are perpendicular to x-axis, and so on",
     "Shapes::Cylinder": "A finite cylinder, defined by the two ends, C0 and C1, along the axis and a radius R",
     "Shapes::UpperSpace": "A semi-space defined by inequality A*x+B*y+C*z>D, Coef is required, but not Equ"
  },
  "Shapes": [
     {"Name":     "Test"},
     {"Origin":   [0,0,0]},
     {"Grid":     {"Tag":1, "Size":[40,60,50]}},
     {"Sphere":   {"Tag":2, "O":[30,30,30],"R":20}},
     {"Box":      {"Tag":0, "O":[10,10,10],"Size":[10,10,10]}},
     {"Subgrid":  {"Tag":1, "O":[13,13,13],"Size":[5,5,5]}},
     {"UpperSpace":{"Tag":3,"Coef":[1,-1,0,0],"Equ":"A*x+B*y+C*z>D"}},
     {"XSlabs":   {"Tag":4, "Bound":[[5,15],[35,40]]}},
     {"Cylinder": {"Tag":2, "C0": [0.0,0.0,0.0], "C1": [15.0,8.0,10.0], "R": 4.0}},
     {"ZLayers":  [[1,10,1],[11,30,2],[31,50,3]]}
  ]
 }
```
A shape file must contain a `Shapes` object in the root level. Other 
root-level fields are ignored. The `Shapes` object is a JSON array, with 
each element representing a 3D object or setting. The object-class commands 
include `Grid`, `Sphere`, `Box` etc. Each of these object include a 
number of sub-fields to specify the parameters of the object. For example, the 
`Sphere` object has 3 subfields, `O`, `R` and `Tag`. Field 
`O` has a value of 1x3 array, representing the center of the sphere; 
`R` is a scalar for the radius; `Tag` is the voxel values. The most 
useful command is `[XYZ]Layers`. It contains a series of integer 
triplets, specifying the starting index, ending index and voxel value of a 
layered structure. If multiple objects are included, the subsequent objects 
always overwrite the overlapping regions covered by the previous objects.

There are a few ways for you to use shape description records in your MCX 
simulations. You can save it to a JSON shape file, and put the file name in 
Line\#6 of your `.inp` file, or set as the value for Domain.VolumeFile field in a 
`.json` input file. In these cases, a shape file must have a suffix of `.json`.

You can also merge the Shapes section with a `.json` input file by simply 
appending the Shapes section to the root-level object. You can find an example, 
`jsonshape_allinone.json`, under example/shapetest. In this case, you no longer 
need to define the `VolumeFile` field in the input.

Another way to use Shapes is to specify it using the `-P` (or `--shapes`) command 
line flag. For example:
```
     mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
```
This will first initialize a volume based on the settings in the input `.json` 
file, and then rasterize new objects to the domain and overwrite regions that 
are overlapping.

For both JSON-formatted input and shape files, you can use the JSONlab toolbox 
[4] to load and process in MATLAB.


Output data formats
------------------------------------

MCX may produces several output files depending user's simulation settings.
Overall, MCX produces two types of outputs, 1) data accumulated within the 
3D volume of the domain (volumetric output), and 2) data stored for each detected
photon (detected photon data).

### Volumetric output

By default, MCX stores a 4D array denoting the fluence-rate at each voxel in 
the volume, with a dimension of Nx*Ny*Nz*Ng, where Nx/Ny/Nz are the voxel dimension
of the domain, and Ng is the total number of time gates. The output data are
stored in the format of single-precision floating point numbers. One may choose
to output different physical quantities by setting the `-O` option. When the
flag `-X/--saveref` is used, the output volume may contain the total diffuse
reflectance only along the background-voxels adjacent to non-zero voxels. 
A negative sign is added for the diffuse reflectance raw output to distinguish
it from the fluence data in the interior voxels.

When photon-sharing (simultaneous simulations of multiple patterns) or photon-replay
(the Jacobian of all source/detector pairs) is used, the output array may be extended
to a 5D array, with the left-most/fastest index being the number of patterns Ns (in the
case of photon-sharing) or src/det pairs (in replay), denoted as Ns.

Several data formats can be used to store the 3D/4D/5D volumetric output. 

#### mc2 files

Starting in MCX v2023, `.mc2` files are no longer the default output format for
MCX binary. Instead, JSON based JNIfTI (`.jnii`) files are used.

The `.mc2` format is simply a binary dump of the entire volumetric data output,
consisted of the voxel values (single-precision floating-point) of all voxels and
time gates. The file contains a continuous buffer of a single-precision (4-byte) 
5D array of dimension Ns\*Nx\*Ny\*Nz\*Ng, with the fastest index being the left-most 
dimension (i.e. column-major, similar to MATLAB/FORTRAN).

To load the mc2 file, one should call `loadmc2.m` and must provide explicitly
the dimensions of the data. This is because mc2 file does not contain the data
dimension information.

Saving to .mc2 volumetric file is deprecated as we are transitioning towards
JNIfTI/JData formatted outputs (.jnii). 

#### nii files

The NIfTI-1 (.nii) format is widely used in neuroimaging and MRI community to
store and exchange ND numerical arrays. It contains a 352 byte header, followed
by the raw binary stream of the output data. In the header, the data dimension
information as well as other metadata is stored. 

A .nii output file can be generated by using `-F nii` in the command line.

The .nii file is widely supported among data processing platforms, including
MATLAB and Python. For example
- niftiread.m/niftiwrite in MATLAB Image Processing Toolbox
- JNIfTI toolbox by Qianqian Fang (https://github.com/NeuroJSON/jnifti/tree/master/lib/matlab)
- PyNIfTI for Python http://niftilib.sourceforge.net/pynifti/intro.html

#### jnii files

Starting in MCX v2023, JSON based JNIfTI (`.jnii`) files are used as the default
volumetric data output format.

The JNIfTI format represents the next-generation scientific data storage 
and exchange standard and is part of the US NIH-funded NeuroJSON initiative (https://neurojson.org)
led by the MCX author Dr. Qianqian Fang. The NeuroJSON project aims at developing
easy-to-parse, human-readable and easy-to-reuse data storage formats based on
the ubiquitously supported JSON/binary JSON formats and portable JData data annotation
keywords. In short, .jnii file is simply a JSON file with capability of storing 
binary strongly-typed data with internal compression and built in metadata.

The format standard (Draft 1) of the JNIfTI file can be found at

https://github.com/NeuroJSON/jnifti

A .jnii output file can be generated by using `-F jnii` in the command line.

The .jnii file can be potentially read in nearly all programming languages 
because it is 100% compatible to the JSON format. However, to properly decode
the ND array with built-in compression, one should call JData compatible
libraries, which can be found at https://neurojson.org/#software

Specifically, to parse/save .jnii files in MATLAB, you should use
- JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
- `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

To parse/save .jnii files in Python, you should use
- PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['NIFTIData']` 
is a NumPy `ndarray` object storing the volumetric data.


#### bnii files

The binary JNIfTI file is also part of the JNIfTI specification and the NeuroJSON
project. In comparison to text-based JSON format, .bnii files can be much smaller
and faster to parse. The .bnii format is also defined in the BJData specification

https://github.com/NeuroJSON/bjdata

and is the binary interface to .jnii. A .bnii output file can be generated by 
using `-F bnii` in the command line.

The .bnii file can be potentially read in nearly all programming languages 
because it was based on UBJSON (Universal Binary JSON). However, to properly decode
the ND array with built-in compression, one should call JData compatible
libraries, which can be found at https://neurojson.org/#software

Specifically, to parse/save .jnii files in MATLAB, you should use one of
- JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
- `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

To parse/save .jnii files in Python, you should use
- PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['NIFTIData']` 
is a NumPy `ndarray` object storing the volumetric data.

### Detected photon data

If one defines detectors, MCX is able to store a variety of photon data when a photon
is captured by these detectors. One can selectively store various supported data fields,
including partial pathlengths, exit position and direction, by using the `-w/--savedetflag`
flag. The storage of detected photon information is enabled by default, and can be
disabled using the `-d` flag.

The detected photon data are stored in a separate file from the volumetric output.
The supported data file formats are explained below.

#### mch files

The .mch file, or MC history file, is the legacy binary format, written when the
volumetric output format is set to `mc2`, `nii`, `hdr` or `tx3` by `-F`. When the
default `-F jnii` (or `-F bnii`) is used, the detected photons are saved in the
JSON/JData `.jdt` (or `.jdb`) format described below instead.

The .mch file contains a 256 byte binary header, followed by a 2-D numerical array
of dimensions `#savedphoton * #colcount` as recorded in the header.
```
typedef struct MCXHistoryHeader{
    char magic[4];                 // magic bits= 'M','C','X','H'
    unsigned int  version;         // version of the mch file format 
    unsigned int  maxmedia;        // number of media in the simulation 
    unsigned int  detnum;          // number of detectors in the simulation 
    unsigned int  colcount;        // how many output files per detected photon 
    unsigned int  totalphoton;     // how many total photon simulated 
    unsigned int  detected;        // how many photons are detected (not necessarily all saved) 
    unsigned int  savedphoton;     // how many detected photons are saved in this file 
    float unitinmm;                // what is the voxel size of the simulation
    unsigned int  seedbyte;        // how many bytes per RNG seed
    float normalizer;              // what is the normalization factor
    int respin;                    // if positive, repeat count so total photon=totalphoton*respin; if negative, total number is processed in respin subset 
    unsigned int  srcnum;          // number of sources for simultaneous pattern sources 
    unsigned int  savedetflag;     // number of sources for simultaneous pattern sources 
    unsigned int  totalsource;     // total source number when multiple sources are defined
    int reserved[1];               // reserved fields for future extension 
} History;
```
When the `-q` flag is set to 1, the detected photon initial seeds are also stored
following the detected photon data, consisting of a 2-D byte array of `#savedphoton * #seedbyte`.

To load the mch file, one should call `loadmch.m` in MATLAB/Octave.

Saving to .mch history file is deprecated as we are transitioning towards
JSON/JData formatted outputs (`.jdt`/`.jdb`).

#### jdt and jdb files

When `-F jnii` (the default) is used, instead of saving the detected photons into the
legacy .mch format, a `session_detp.jdt` file is written, which is a pure JSON file.
With `-F bnii`, the same data are saved in the binary JData (BJData) format as
`session_detp.jdb`. Prior to v2026, this file used the `.jdat` suffix; `.jdat` files
can still be loaded, for example, for replay (`-E file.jdat`). This file contains a
hierarchical data record of the following JSON structure

```
{
   "MCXData":{
       "Info":{
           "Version":
           "MediaNum":
           "DetNum":
           ...
           "Media":{
               ...
           }
       },
       "PhotonData":{
           "detid":
           "srcid":
           "nscat":
           "ppath":
           "mom":
           "p":
           "v":
           "w0":
       },
       "Seed":[
           ...
       ]
   }
}
```

where "Info" is required, and other subfields are optional depending on users' input;
"srcid" is only present in multi-source simulations, and "Seed" is only present
when `-q 1` is used. Each subfield in this file may contain JData 1-D or 2-D array
constructs to allow storing binary and compressed data.

Although .jdt and .jnii have different suffixes, they are both JSON/JData files and
can be opened/written by the same JData compatible libraries mentioned above, i.e.

For MATLAB
- JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
- `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

For Python
- PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['MCXData']['PhotonData']` 
stores the photon data, and `data['MCXData']['Seed']` stores the RNG seeds.


### Photon trajectory data

For debugging and plotting purposes, MCX can output photon trajectories, as polylines,
when `-D M` flag is attached, or mcxlab is asked for the 5th output. Such information
can be stored in one of the following formats.

#### mct files

When the legacy `-F mc2`, `nii`, `hdr` or `tx3` output formats are used, MCX stores the
photon trajectory data in a .mct (MC trajectory) file, which uses the same binary format
as .mch. This file can be loaded to MATLAB using the same `loadmch.m` function.

Using .mct file is deprecated and users are encouraged to migrate to the .jdt file
as described below.

#### jdt and jdb files

When `-F jnii` (the default) is used, the trajectory data are saved in a separate
JSON file named `session_traj.jdt` (or `session_traj.jdb` with `-F bnii`), with a
"Trajectory" object storing the "photonid", "p" (position), "w0" (weight) and
"srcid" of each trajectory point, as well as "iquv" (Stokes vector) when
`--trajstokes 1` is used in polarized simulations. The relevant parsers are listed
in the above section. In MATLAB/Octave, `mcxplotphotons.m` can plot the trajectories.


Using MCXLAB in MATLAB and Octave
------------------------------------

MCXLAB is the native MEX version of MCX for MATLAB and GNU Octave. It includes 
the entire MCX code in a MEX function which can be called directly inside 
MATLAB or Octave. The input and output files in MCX are replaced by convenient 
in-memory struct variables in MCXLAB, thus, making it much easier to use and 
interact. MATLAB/Octave also provides convenient plotting and data analysis 
functions. With MCXLAB, your analysis can be streamlined and simplified without 
involving disk files.

Please read the `mcxlab/README.txt` file for more details on how to install and 
use MCXLAB.

Please also browse this interactive [Jupyter Notebook based MCXLAB tutorial](https://colab.research.google.com/github/fangq/mcx/blob/master/mcxlab/tutorials/mcxlab_getting_started.ipynb)
to see a suite of examples showing the key functionalities of MCXLAB (using GNU Octave).


Using PMCX in Python
------------------------------------

PMCX is the native binary binding of MCX for Python 3.6 or newer. Similar to
MCXLAB, PMCX can run GPU-based simulations inside a Python environment with
efficient in-memory inputs and outputs. It can be installed with

```
python3 -m pip install pmcx
```

On macOS, where CUDA is not available, pip installs a pure-python pmcx that
provides the utility functions (`detweight`, `meanpath`, `loadmc2` etc.); to run
simulations there, install `pmcxcl`, which uses these utilities and the OpenCL
version of MCX.

Please read the `pmcx/README.md` file for more details on how to install and 
use PMCX.

Please also browse this interactive [Jupyter Notebook based PMCX tutorial](https://colab.research.google.com/github/fangq/mcx/blob/master/pmcx/tutorials/pmcx_getting_started.ipynb)
to see a suite of examples showing the key functionalities of PMCX.


Using MCX Studio GUI
-----------------------

Starting from v2026, the `mcxstudio` GUI shipped with MCX is MCX Studio 2, a
rewrite of the original MCX Studio, with source code in the `mcxstudio2/` folder.
It is a cross-platform graphical front end for MCX, MCX-CL and MMC, written in
Object Pascal with Lazarus, and runs on Windows, Linux and macOS.

In MCX Studio 2, a simulation is simply the `.json` input file that mcx, mcxcl and
mmc read; there is no separate project format. The settings are grouped in the order
a simulation is described, and pressing F1 over any setting shows what it does and
which mcx option it corresponds to. The Command tab shows the command line that will
be run and the JSON tab shows the file that will be written. The built-in 3-D view
renders the domain, shapes, mmc meshes, sources and detectors, as well as the
fluence (`.jnii`, `.bnii`, `.nii`, `.mc2`) and photon trajectories (`_traj.jdt`)
written by the simulators. It uses OpenGL 3.3 and falls back to a software renderer
when OpenGL 3.3 is not available, for example, over a remote X display.

MCX Studio does not perform simulations by itself. Depending on the selected processor
(`nvidia`, `amd`, `intel` or `cpu`) and the domain (voxel grid or mesh), it runs mcx,
mcx built with ROCm/HIP, mcxcl or mmc. The simulators are searched in
`MCXSuite/<simulator>/bin` beside mcxstudio, beside mcxstudio, `~/MCXStudio`, and then
in the `PATH`. Using the all-in-one MCX Studio package (see Requirement and
Installation) is the easiest way to have all of them in place.

MCX Studio can also be used from the command line, for example

```
mcxstudio benchmark1.json benchmark1.jnii      # open an input and its result
mcxstudio --run --processor intel qtest.json   # run the input with mcxcl
xvfb-run -a mcxstudio --snapshot qtest.png qtest.json qtest.jnii
```

Projects (`.mcxp` files) created by the original MCX Studio can be opened using
`mcxstudio -p project.mcxp`, or converted to `.json` files using `utils/mcxp2json.py`.
Please read `mcxstudio2/README.md` for more details, including how to build it
from the source code.

The source code of the original MCX Studio, as well as the `mcxshow` and `mcxviewer`
utilities, are still kept in the `mcxstudio/` folder.


Interpreting the Output
--------------------------

MCX output consists of two parts: the output data files and the messages printed
on the screen.

### Output files

The volumetric output file (`.jnii` by default, or `.bnii`, `.nii` or `.mc2` with
`-F`) contains the fluence-rate distribution from the simulation in the given
medium. By default, this fluence rate is a normalized solution (as opposed to the
raw probability), therefore, one can compare it directly to analytical solutions
(i.e. Green's functions). The order of storage in the output file is the same as
in the input volume: i.e., if the input is row-major, the output is row-major, and
so on. The dimensions of the data are Nx, Ny, Nz and Ng, where Ng is the total
number of time gates.

By default, MCX produces the **Green's function** of the **fluence rate** for 
the given domain and source. It is sometimes also known as the time-domain 
“two-point” function. If you run MCX with the following command

      mcx -f input.inp -s output ....

the fluence-rate data will be saved in a file named “output.jnii” under the
current folder. If you run MCX without `-s output`, the output file is named
after the session ID given in the input file.

To understand this further, you need to know that a **fluence-rate (Phi(r,t))** 
is measured by number of particles passing through an infinitesimal spherical 
surface per **unit time** at **a given location** regardless of directions. The 
unit of the MCX output is “W/mm<sup>2</sup> = J/(mm<sup>2</sup>s)”, if it is 
interpreted as the “energy fluence-rate” [6], or 
“1/(mm<sup>2</sup>s)”, if the output is interpreted as the “particle 
fluence-rate” [6].

The Green's function of the fluence-rate means that it is produced by a 
**unitary source**. In simple terms, this represents the fraction of 
particles/energy that arrives at a location per second under **the radiation of 1 
unit (packet or J) of particle or energy at time t=0**. The Green's function is 
calculated by a process referred to as the “normalization” in the MCX code 
and is detailed in the MCX paper [6] (MCX and MMC outputs share the same 
meanings).

Please be aware that the output flux is calculated at each time-window defined 
in the input file. For example, if you type

     0.e+00 5.e-09 1e-10  # time-gates(s): start, end, step

in the 5th row of a legacy `.inp` input file (or `"T0": 0, "T1": 5e-9, "Dt": 1e-10`
in the `Forward` section of a JSON input file), MCX will produce 50 fluence-rate snapshots,  
corresponding to the time-windows at [0 0.1] ns, [0.1 0.2]ns ... and 
[4.9,5.0] ns. To convert the fluence rate to the fluence for each 
time-window, you just need to multiply each solution by the width of the 
window, 0.1 ns in this case. To convert the time-dependent fluence-rate to 
continuous-wave (CW) fluence (fluence in short), you need to integrate the 
fluence-rate along the time dimension. Assuming the fluence-rate after 5 ns is 
negligible, then the CW fluence is simply `sum(flux_i*0.1 ns, i=1,50)`. You can 
read `mcx/example/validation/plotsimudata.m` and
`mcx/example/sphbox/plotresults.m` for examples to compare an MCX output with  
the analytical fluence-rate/fluence solutions.

One can load an `.mc2` output file into Matlab or Octave using the loadmc2 
function in the `{mcx root}/utils` folder.

To get a continuous-wave solution, run a simulation with a sufficiently long 
time window, and sum the flux along the time dimension, for example

       mcx=loadmc2('output.mc2',[60 60 60 10],'float');
       cw_mcx=sum(mcx,4);

Note that for time-resolved simulations, the corresponding solution in the 
results approximates the flux at the center point of each time window. For 
example, if the simulation time window setting is 
`[t0,t0+dt,t0+2dt,t0+3dt...,t1]`, the time points for the snapshots stored in 
the solution file are located at `[t0+dt/2, t0+3*dt/2, t0+5*dt/2, ... ,t1-dt/2]`

A more detailed interpretation of the output data can be found at 
https://mcx.space/wiki/?MMC/Doc/FAQ#How_do_I_interpret_MMC_s_output_data

MCX can also output “current density” (J(r,t), unit W/mm^2, same as  
Phi(r,t)) - referring to the expected number of photons or Joule of energy 
flowing through a unit area pointing towards a particular direction per unit 
time. The current density can be calculated at the boundary of the domain by 
two means:

1.  using the detected photon partial path output (i.e. the second output of 
  mcxlab.m),
  one can compute the total energy E received by a detector, then one can divide 
  E by the area/aperture of the detector to obtain the J(r) at a detector (E 
  should be calculated as a function of t by using the time-of-flight of detected  
  photons, the E(t)/A gives J(r,t); if you integrate all time gates, the total 
  E/A gives the current I(r), instead of the current density).
2.  use `-X 1` or `--saveref/cfg.issaveref` option in mcx to enable the
  diffuse reflectance recordings on the boundary. The diffuse reflectance is  
  represented by the current density J(r) flowing outward from the domain.

The current density has, as mentioned, the same unit as fluence rate, but the 
difference is that `J(r,t)` is a vector, and Phi(r,t) is a scalar. Both measure
the energy flow across a small area (the area has a direction in the case of J)
per unit time.

You can find more rigorous definitions of these quantities in Lihong Wang's 
Biomedical Optics book, Chapter 5.

### Console print messages

Timing information is printed on the screen (stdout). The clock starts (at time 
T0) right before the initialization data is copied from CPU to GPU. For each 
simulation, the elapsed time from T0 is printed (in ms). Also the accumulated 
elapsed time is printed for all memory transactions from GPU to CPU.

When a user specifies `-D P` in the command line, or sets  
`cfg.debuglevel='P'`, MCX or MCXLAB prints a progress bar showing the percentage 
of completion.



Best practices guide
-----------------------

To maximize MCX's performance on your hardware, you should follow the best 
practices guide listed below:

### Use a mid-range or high-end GPU, and multiple GPUs if possible

MCX is highly scalable, providing a linear speedup as long as you provide the
GPU cores it can use. As a result, the better the GPU you use, the higher the speed
you can get. An enthusiast-grade GPU, such as the RTX 4070Ti (~$700), can be 12x
faster than a low-end laptop RTX 4050 GPU, even within the same generation.

MCX can readily take advantage of multiple GPUs if you have them installed. The
MCX simulation speed scales nearly linearly with the number of GPUs.
So, to maximize MCX performance, get at least a mid-range or high-end consumer
grade GPU; if you need more speed, adding more GPUs will cut down the runtime.

### Launch as many threads as possible

It has been shown that MCX's speed is related to the thread number (-t). 
Generally, the more threads, the better speed, until all GPU resources are 
fully occupied. For higher-end GPUs, a thread number over 10,000 is 
recommended. Please use the autopilot mode, `-A`, to let MCX determine the 
“optimal” thread number when you are not sure what to use. In autopilot mode,
photons are also dispatched to the threads dynamically, so that threads that
finish early take on more photons.

### Use -A 0 to reproduce a simulation exactly

Because of the dynamic photon dispatch, two autopilot runs with the same seed
(`-E`) agree statistically but do not simulate the identical set of photons. If you
need to reproduce a simulation exactly, for example to debug or to compare outputs
bit-by-bit, disable the autopilot with `-A 0` (`cfg.autopilot=0` in MCXLAB/PMCX) and
keep the seed, the thread number (`-t`) and the block size (`-T`) unchanged. Photon
replay (`-E file_detp.jdt`) always uses the static dispatch.



Acknowledgement
------------------

MCX contains modified versions of the below source codes from other 
open-source projects (with a compatible license).

### cJSON library by Dave Gamble

- Files: src/cjson folder
- Copyright (c) 2009 Dave Gamble
- URL: https://github.com/DaveGamble/cJSON
- License: MIT License, https://github.com/DaveGamble/cJSON/blob/master/LICENSE

### GLScene library for Lazarus by GLScene developers

- Files: mcxstudio/glscene/*
- Copyright (c) GLScene developers
- URL: http://glscene.org, https://sourceforge.net/p/glscene/code/HEAD/tree/branches/GLSceneLCL/
- License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
- Comment: 
  A subset of the GLSceneLCL branch is included as part of the MCX source code tree
  to allow compilation of the MCX Studio binary on various platforms without
  needing to install the full package.

### Texture3D sample project by Jürgen Abel

- Files: mcx/src/mcxstudio/mcxview.pas
- Copyright (c) 2003 Jürgen Abel
- License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
- Comment: 
  The MCX volume renderer (mcxviewer) was adapted based on the Texture3D Example 
  provided by the GLScene Project (http://glscene.org). The original author of 
  this example is Jürgen Abel. 

### Synapse communication library for Lazarus

- Files: mcxstudio/synapse/*
- Copyright (c) 1999-2017, Lukas Gebauer
- URL: http://www.ararat.cz/synapse/
- License: MIT License or LGPL version 2 or later or GPL version 2 or later
- Comment:
  A subset of the Synapse units is included as part of the MCX source code tree
  to allow compilation of the MCX Studio binary on various platforms without
  needing to install the full package.

### ZMat data compression unit

- Files: src/zmat/*
- Copyright: 2019-2023 Qianqian Fang
- URL: https://github.com/fangq/zmat
- License: GPL version 3 or later, https://github.com/fangq/zmat/blob/master/LICENSE.txt

### LZ4 data compression library

- Files: src/zmat/lz4/*
- Copyright: 2011-2020, Yann Collet
- URL: https://github.com/lz4/lz4
- License: BSD-2-clause, https://github.com/lz4/lz4/blob/dev/lib/LICENSE

### LZMA/Easylzma data compression library

- Files: src/zmat/easylzma/*
- Copyright: 2009, Lloyd Hilaiel, 2008, Igor Pavlov
- License: public-domain
- Comment:
 All the cruft you find here is public domain.  You don't have to
 credit anyone to use this code, but my personal request is that you mention
 Igor Pavlov for his hard, high quality work.

### myslicer toolbox by Anders Brun

- Files: utils/{islicer.m, slice3i.m, image3i.m}
- Copyright (c) 2009 Anders Brun, anders@cb.uu.se
- URL: https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume
- License: BSD-3-clause License, https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume#license_modal

### MCX Filter submodule

- Files: filter/*
- Copyright (c) 2018 Yaoshen Yuan, 2018 Qianqian Fang
- URL: https://github.com/fangq/GPU-ANLM/
- License: MIT License, https://github.com/fangq/GPU-ANLM/blob/master/LICENSE.txt

### pymcx Python module

- Files: pymcx/*
- Copyright (c) 2020  Maxime Baillot <maxime.baillot.1 at ulaval.ca>
- URL: https://github.com/fangq/GPU-ANLM/
- License: GPL version 3 or later, https://github.com/4D42/pymcx/blob/master/LICENSE.txt

### Pybind11
- Files: src/pybind11/*
- Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
- URL: https://github.com/pybind/pybind11/
- License: BSD-style license, https://github.com/pybind/pybind11/blob/master/LICENSE

Reference
------------

- [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of 
  Photon Migration in 3D Turbid Media Accelerated by Graphics Processing Units," 
  Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

- [Yu2018] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, 
  “Scalable and massively parallel Monte Carlo photon transport simulations 
  for heterogeneous computing platforms,” J. Biomed. Opt. 23(1), 010504 (2018).

- [Yan2020] Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel 
  based Monte Carlo algorithm for accurate and efficient photon transport 
  modeling in complex bio-tissues," Biomed. Opt. Express, 11(11) pp. 6262-6270.
  https://opg.optica.org/boe/abstract.cfm?uri=boe-11-11-6262

If you use MCX in your research, the author of this software would like you to 
cite the above papers in your related publications.

Links:

- [1] https://developer.nvidia.com/cuda-downloads
- [2] https://developer.nvidia.com/cuda-gpus
- [3] https://en.wikipedia.org/wiki/Row-major_order
- [4] https://neurojson.org/jsonlab
- [5] http://science.jrank.org/pages/60024/particle-fluence.html
- [6] https://opg.optica.org/oe/abstract.cfm?uri=oe-17-22-20178
