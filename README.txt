---------------------------------------------------------------------
=                   Monte Carlo eXtreme  (MCX)                      =
                          CUDA Edition
---------------------------------------------------------------------

*Author:  Qianqian Fang <q.fang at neu.edu>
*License: GNU General Public License version 3 (GPLv3)
*Version: 1.9.7 (v2022.10, Heroic Hexagon)
*Website: http://mcx.space

---------------------------------------------------------------------

Table of Content:

<toc>

---------------------------------------------------------------------

== #  What's New ==

MCX v2022.10 introduces a major new feature - split-voxel MC (SVMC), published in 
Biomedical Optics Express recently by Shijie Yan and Qianqian Fang, see 
[Yan2020](https://www.osapublishing.org/boe/abstract.cfm?uri=boe-11-11-6262) for details.
Shortly, SVMC provides a level of accuracy close to mesh-based MC (MMC) in modeling
curved boundaries but it is 4x to 6x faster than MMC. Several demo scripts of SVMC 
can be found in the MCXLAB package under examples/demo_svmc_\*. In addition, MCX v2022.10
supports GPU-based polarized light simulation, see our JBO paper
[Yan2022](https://doi.org/10.1117/1.JBO.27.8.083015). This release also includes
both the web-client and server scripts for MCX Cloud - an in-browser MC simulator
as we reported in [Fang2022](https://doi.org/10.1117/1.JBO.27.8.083008). Moreover, MCX
v2022.10 provides an official pymcx module to run stream-lined MCX simulations in
Python, offering mcxlab-like interface.

*   Introduced Split voxel MC (SVMC) to accurately model curved boundaries
*   GPU polarized light modeling (Stokes) with 900x speedup
*   Web-based MCX Cloud platform including web-client and server scripts
*   pymcx - an mcxlab-like Python module for running MCX simulations in Python
*   Added Debian/Ubuntu packages for easy installation
*   Added a unified command line interface, photon, to call mcx/mcxcl/mmc
*   Fine-tuned Windows installer
*   Support CMake in Travis-CI automation

A detailed list of updates is summarized below (key features marked with “*”):

* 2022-10-05 [dc42951] prevent nan where log(rand) is calculated
* 2022-10-05 [63ffc1e] fix nan in detected photon data when using hyperboloid src, see https://groups.google.com/g/mcx-users/c/vyfHbzEO-0M/m/JzlpEZ3TBwAJ
* 2022-09-07 [e281f3e] allow to preview continuously varying medium (4D cfg.vol)
* 2022-08-19 [10330ef] fix windows compilation error
* 2022-08-17 [bbb4425] prevent zero-valued mus creating nan, #133
* 2022-08-12 [51f42f5] fix mcxlab log printing caused by commit f3beb75a
* 2022-08-12 [7058785] Lambertian launch for all focusable sources when focal-length is -inf
* 2022-07-28 [6d64c0b] fix incorrect flag for skipvoid
* 2022-06-27 [3d4fb26] partially fix rf replay
* 2022-06-04 [8af3631] fix line source
* 2022-05-22 [149b1ef] make code compile on windows
* 2022-05-20 [e87bb49] use consistent file naming convention, remove outdated units
* 2022-05-20 [45d84d3] complete reformat source code using astyle, always run make pretty before committing
* 2022-05-20 [aff8ff0] add source code formatting option
* 2022-05-20 [f3beb75] use MATLAB_MEX_FILE to determine container environment
* 2022-05-18 [1295024] fix incorrect trajectory id, fix #147
* 2022-05-18 [ccd2deb] fix macro condition
* 2022-05-18 [6f4ee88] use MCX_CONTAINER and env macros to replace extra PYMCX_CONTAINER
* 2022-05-16 [6fa1580] avoid using clear all and ~ in return value
* 2022-05-16 [21f9bd7] merge changes with @ShijieYan's svmc fix
* 2022-05-16 [8b2740f] debugging svmc crashes
* 2022-05-16 [7582a6e] fix svmc issue after patch f8da832f11b751c07d33c77dd7d428a2c75a888b
* 2022-05-15 [188ac2a] Added Pybind11's license info to README.md.
* 2022-05-15 [86529cf] Added PYMCX_CONTAINER compilation macro. Added support for extracting MCX_ERRORs like Mex + author fix.
* 2022-05-15 [b58ad88] Renamed gpu_info to gpuinfo for consistency.
* 2022-05-15 [0582974] changed issaveref to accept ints.
* 2022-05-15 [1cf1b3b] Added py::value_error handling + additional error checking for volume assurance.
* 2022-05-15 [7df8938] Added better + more informative exception handling for pymcx.
* 2022-05-15 [6a741e8] Changed reinterpret_casts to direct object construction + added the stat dict to the output dict + defined PYBIND11_DETAILED_ERROR_MESSAGES for easier debugging.
* 2022-05-15 [e4547ba] add pybind11 as submodule to build pymcx
* 2022-05-13 [f8da832] fix cyclic bc demo and srctype demo error, svmc still not working
* 2022-05-13 [4bd3974] report register counts on sm_60 using nvcc --ptxas-options=-v
* 2022-05-13 [e8f6a2d] fix cfg.unitinmm does not exist error
* 2022-05-13 [447975f] complete dda ray-marching, cube60 benchmark gain 40% speed, others are similar
* 2022-05-12 [b873f90] add integer voxel indices to avoid nextafter
* 2022-05-11 [32d46cd] merge additional updates from mcxcl version of json2mcx, #139
* 2022-05-11 [3d6d7df] fix bugs in json2mcx, #139
* 2022-05-08 [3b1c320] Removed nlhs argument left from Matlab.
* 2022-05-08 [61cc994] Fixed issue with std::cout and std::cerr flush.
* 2022-05-06 [d9793e9] Added working setup.py.
* 2022-05-03 [a3e47c8] Minor fix.
* 2022-05-02 [c9bedd6] Moved validate_config.
* 2022-05-02 [5427ece] Working prototype with different volume configs.
* 2022-05-02 [739b7ea] Moved some stuff to interface-common.cpp.
* 2022-05-02 [d852a87] Minor bug fixes.
* 2022-05-01 [f4cd3c3] Added kwargs version of mcx.
* 2022-05-01 [baa5fdd] Skeleton is done.
* 2022-04-24 [e919716] Prints GPU info.
* 2022-04-23 [0c7b6a7] Working interface and CMakeLists.txt file.
* 2022-04-19 [c710c3c] update ubj parser, update jnifti metadata
* 2022-04-15 [4033c54] mcxlab bug fix: digimouse atlas voxel size is 0.2mm, not 0.8mm
* 2022-04-15 [9b17eee] critical bug fix: digimouse atlas voxel size is 0.2mm, not 0.8mm
* 2022-03-31 [df6c311] Add viewing axis selection
* 2022-03-25 [04f1565] Add optimized isosurface rendering; add the ability to view cross-sectional slices of geometry
* 2022-03-23 [200decb] Remove testing files
* 2022-03-23 [e434553] Remove unnecessary variable
* 2022-03-22 [ebd5bee] update ubj to support BJData Draft 2, add JNIfTI _DataInfo_, fix #142
* 2022-03-21 [6de0855] Enable event-based repainting; re-add shader refinement, remove animation frame bugs; remove unnecessary shader branches and discards
* 2022-03-19 [7ef65a8] Added event-based repainting; shader optimizations
* 2022-03-05 [a93f6fa] save user info in local storage to avoid retyping
* 2022-03-05 [34d9afa] fix SaveDataFlag invisible bug
* 2022-02-18 [f051314] add missing voxel unit
* 2022-02-03 [23bf5b2] lowering default photon number so it can be launched on most gpus
* 2022-01-31 [220d9c2] fix incorrect type for gsmatrix
* 2022-01-31 [28e20d6] fix windows vs warning
* 2022-01-29 [6a9ad2f] update mcx_utils to use the Mie interface
* 2022-01-29 [13679e9] fix compilation issue of mcx_mie.cpp using MSVC, close #138
* 2022-01-28 [d7daf57] manually resolve complaint in CUDA 9
* 2022-01-28 [e99edb2] update .travis.yml
* 2022-01-28 [533c8ce] manually add mcx_mie in Makefile
* 2022-01-28 [e56b5cb] improve complex arithmetic compatablity with MSVC
* 2022-01-27 [a0ed0e7] add Mie function modules into cmake
* 2022-01-27 [c350c67] seperate Mie scattering functions from mcx_utils.h
* 2022-01-27 [0d51bb7] add missing i detflag in command line
* 2022-01-27 [9b74e4b] fix: add save detector flag for Stokes vectors
* 2022-01-26 [077060a] use static_cast in mcxlab so that cfg.vol can be realloc in mcx_shapes
* 2022-01-26 [8503125] do not reset cfg.vol when rasterizing cfg.shapes
* 2022-01-26 [3f22070] fix normalization in multiple detector RF replay
* 2022-01-26 [cdfd468] apply normalization to both real and imaginary jacobain for RF replay
* 2022-01-26 [87a310e] one can use ~ to ignore fluence output in octave, not in matlab
* 2022-01-26 [d45f084] allow users to explicitly disable fluence output by accepting cfg.issave2pt
* 2022-01-25 [376a730] partial fix to RF Jacobian calculation, need verification
* 2022-01-25 [d6e2b9e] NaN value in mua_float and muamus_float signifies a 0-value voxel
* 2022-01-24 [c9f2ad9] force threejs version to avoid breaking update in R136
* 2022-01-14 [51483eb] add template specialization for polarized mode
* 2022-01-12 [3487dfe] update the example for the polarized MCX
* 2021-12-15 [b9e046a] fix out of bounds error due to limited precision of R_PI
* 2021-12-15 [3b10265] fix the built-in example to match the update in e5dfd78f28f31d710e02206cb2835aabcd4d5508
* 2021-12-15 [dbe17af] no Stoke vector output for unpolarized MCX simulation
* 2021-12-15 [99293dd] add sanity check for incident Stokes vector
* 2021-12-13 [f1537bd] no need to check constant memory usage in polarized mode
* 2021-12-13 [61281ae] use prop.g to return the anisotropy computed from Mie
* 2021-12-12 [3b0ecc0] fix #133, handling underflowing denorms in float to half conversion for muamus_float
* 2021-12-11 [979f691] Move scattering matrix from constant memory to global memory
* 2021-11-29 [5c13f4b] avoid divided by zero on windows cygwin gcc
* 2021-11-29 [ef57f4b] allow make double to compile
* 2021-11-28 [0c96fe8] accept JData styled NaN in the JSON input for srcdir
* 2021-11-26 [a4545a4] fix #131, mcxplotshapes now plots shapes with correct scale
* 2021-11-04 [2585471] making svmc matlab demos compatible with Octave 5
* 2021-11-03 [5976811] replace matlab-only function with more portable code
* 2021-11-01 [37e121c] update preprint version url
* 2021-10-21 [7a77bf7] display rendering frame rate
* 2021-10-18 [99592c1] fix: #114 improve robustness to unknown boundry conditions
* 2021-10-14 [1aa2922] feat: Add one-sheet hyperboloid gaussian beam
* 2021-10-07 [86d56c2] feat: output prop. along with det. photon profile
* 2021-10-07 [24f4698] fix: ensure the largest grid to be accumulated
* 2021-10-06 [e5dfd78] feat: Support target mus or musp in polarized MCX
* 2021-10-06 [ae9216d] remove old det photon data after a new simulation
* 2021-10-05 [8cb21b5] support downloading detected photon data in mcxcloud
* 2021-10-04 [81ff4b1] fix rf replay bugs reported by Pauliina Hirvi
* 2021-09-24 [833bf6a] Reorganize some kernel code to optimize SVMC speed
* 2021-09-20 [605c15f] Fix numerical error of intersection test in SVMC
* 2021-09-20 [392ee87] Reorder code to fix photon detection for SVMC
* 2021-09-07 [5c44c6e] fix trajectory saving buffer length bug
* 2021-08-20 [99ea2b6] avoid continuous mua/mus inputs be treated as 0-label if mua=mus=0
* 2021-08-01 [943197a] Reorder preprocessing code to allow detector in SVMC Mask detector voxels only after the volume has been fully prepared!
* 2021-07-27 [65359f7] avoid extra level of square brackets for Optode.Detector
* 2021-07-27 [6b2f074] accept 3-element param1
* 2021-07-27 [2633bfb] avoid param1 missing error if not present
* 2021-07-23 [192613b] fix offset of cylinder along the axis direction, close #119
* 2021-07-17 [fc0465d] add tutorial 2 link
* 2021-07-17 [8c72d17] restore accidentally removed analytics tag
* 2021-07-13 [566df5e] provide flags to help access the detp.jdat file
* 2021-07-12 [b97c0f6] return metadata when loading simulation from library
* 2021-07-12 [3be6603] add default tab in the direct link,return mcx error if failed
* 2021-07-12 [70fb2a7] fix mixlabel byte order
* 2021-07-11 [24df0c3] add comment on raw voxel binary data layout
* 2021-07-07 [53d7ac0] fix shared mem misalignment error, close #118
* 2021-07-06 [7f8a2ac] allow ArrayZipSize to accept 1x2 array
* 2021-07-05 [9b00fa3] fix initial tab
* 2021-07-05 [2737853] deep copy data.options to avoid script error
* 2021-07-05 [0398b95] fix several bug while recording utorials
* 2021-07-04 [f5974ed] add preprint link
* 2021-07-04 [5d21a0d] create mcx cloud tutorial, add link
* 2021-07-03 [df0a48b] support tab param in url to open default tab
* 2021-07-03 [4c3f240] link json in direct link, remove schema
* 2021-07-03 [da6595e] add newline to json file download
* 2021-07-03 [cc3c06d] fix X/Y/ZSlabs parsing, restore original schemas
* 2021-07-03 [7c5054d] internal normalization of srcdir
* 2021-07-02 [877e9fe] get user id and group at runtime
* 2021-07-02 [22d6ffe] fix X/Y/ZSlabs schemas
* 2021-07-01 [270bb7c] prepare for beta testing
* 2021-06-30 [4b6df28] rename mcxone to mcxcloud, add help info
* 2021-06-30 [66b77fb] partial fix of json2mcx.m conversion issues
* 2021-06-25 [c399efa] enable negative g value support
* 2021-06-23 [3c642ce] set maximum characters to read for fscanf, fix #117
* 2021-06-23 [606b3d1] handle empty detector array in json2mcx
* 2021-06-22 [b537143] give a warning if the output type is not jacobian in replay
* 2021-06-17 [363d2d8] support reading .jdat file for replay
* 2021-06-04 [7191ca3] make thumbnail the same size when updating
* 2021-06-04 [5a2e13e] add tab overflow control
* 2021-06-04 [59e6be2] layout adjustments
* 2021-06-04 [4b55c88] minor polishing
* 2021-06-03 [1c29578] fix regression
* 2021-06-03 [3aedee6] add LengthUnit, MediaFormat in schema, support number and string for DebugFlag/SaveDataMask
* 2021-06-03 [64c5dd0] fix unnecessary shared memory allocation related to invcdf
* 2021-06-03 [ebf1ea1] support user-defined phase functions via cfg.invcdf, close #13
* 2021-06-03 [0731511] revert back to no restarting policy so that overtime jobs can be killed
* 2021-06-02 [4d2a891] process cache,fix fullname,fix job status,fix server-side limit,kill overtime job
* 2021-06-02 [a08d676] update the skinvessel benchmark
* 2021-06-02 [168db14] feat: save Mie function outputs mus, g to a file
* 2021-06-02 [57a44c5] feat: Add anisotropy g as an output of Mie func.
* 2021-06-02 [7387394] finally fix crossdomain post, change jsonp to json,test simu lib edit
* 2021-06-01 [95c6e1d] test:use default BC for all polarizedMC benchmarks
* 2021-06-01 [98697de] Add a three-layer slab demo for polarizedMC
* 2021-06-01 [7d86804] Add visualization for polarized MC example(MATLAB)
* 2021-05-31 [7d82a51] fix: resolve valgrind complaint: uninit. values
* 2021-05-31 [d6c9743] Add outputs in mcx2json.m to support polarizedMC
* 2021-05-31 [5088678] Add an example for polarized MC
* 2021-05-31 [d3054fd] Add document for polarized MC in mcxlab
* 2021-05-31 [44e0e9c] fea: extend loadmch.m to load output Stokes vector
* 2021-05-31 [a51cb52] feat: support polarized MC in command line (JSON)
* 2021-05-30 [d7921fe] skip checklimit if json is directly loaded from lib
* 2021-05-30 [65870f5] gui fine adjustment,use hash to update runcount,enable restart on fail,permit mcxpub update
* 2021-05-30 [692adfb] fix broken link
* 2021-05-30 [7a159ab] merge css
* 2021-05-30 [55bb1ea] initial drag and drop support, not working
* 2021-05-30 [fc9ca38] add meta headers, other minor adjustments
* 2021-05-29 [96cf071] support embedding src pattern in the all-in-one json/jdata file
* 2021-05-28 [450462c] Add document for functions used in polarized MC
* 2021-05-28 [cbc3340] Optimize Stokes vector computation
* 2021-05-28 [9bd2ce0] Remove redundant code in preprocessing
* 2021-05-28 [06a9c6b] fix: resolve nan results due to numerical error
* 2021-05-28 [d9d1d0a] rewrite some code to save computation
* 2021-05-28 [9195141] Add an example to show polarized photon simulation
* 2021-05-27 [2b87275] fix: rewrite code for better readability
* 2021-05-26 [d836c81] fix: correct formula for stokes parameter update
* 2021-05-25 [105d5a9] feat: Add stokes parameter output in MCXLAB
* 2021-05-25 [87d8847] feat: add polarized photon simulation
* 2021-05-23 [d398cc9] add simulation restrictions for initial public testing of mcx cloud
* 2021-05-23 [26536d3] feat: add preprocessing for polarized MC in mcxlab
* 2021-05-22 [f0975c5] support ring/annulus shaped source via disk source
* 2021-05-21 [6ed9727] support svmc in command line;add svmc example
* 2021-05-21 [3d0a793] reading 8-byte svmc volume format from input file
* 2021-05-20 [4010d99] move svmc repacking to mcx_preprocess
* 2021-05-20 [3214c1b] remove duplicated preprocessing codes in mcx and mcxlab,fix detbc bug in command line
* 2021-05-20 [54b0602] run batch of jobs in each call to fill all GPU devices
* 2021-05-20 [de9850c] add -K short option and svmc mediatype
* 2021-05-19 [660a8b8] relocate db and workspace folder to non www-data accessible paths
* 2021-05-19 [64f3008] update acknowledgement list
* 2021-05-19 [c168a87] can update thumbnail, add credit links
* 2021-05-19 [b9361a1] update to mcxcloud scripts
* 2021-05-15 [cef630b] save volume in jdata format by default
* 2021-05-15 [c50f871] define Frequency in json file instead of Omega
* 2021-05-15 [b4e7b57] initial support RF mua Jacobian in replay, thx Ilkka Nissilä, formula based on Juha Heiskala PhD thesis p45
* 2021-05-10 [073b168] mcxcloud initial preview at JBO hot-topics
* 2021-05-05 [5732e6a] update front and backends
* 2021-05-01 [ee3f88d] update and rename mcxcloudd and mcxserver.cgi
* 2021-05-01 [eac952d] fix cylinder schema, add footer
* 2021-04-14 [aaa1eab] add download, fix jsonp callback, render output volume
* 2021-04-10 [aaef1f3] draw 3d fluence,use orth camera,add cancel
* 2021-04-04 [4ab8105] add src rendering, fix material color
* 2021-04-02 [1ed5272] fix cylinder and layer object drawing bug
* 2021-04-02 [f4ba0b4] add md5 digest for each submitted json for cache lookup
* 2021-03-31 [2c55c3b] change basic tab name
* 2021-03-31 [3c466a1] now mcxcloud can render 3D volumes, float32 buffer only
* 2021-03-29 [b379b2b] initial support in rendering 3d volume, add schema to support jdata ND array
* 2021-03-28 [31345a1] support Domain.Volume to encode JData-formatted 3D array
* 2021-03-28 [9c2e8c7] rendering all shape types, bbx as dashed box,add tag based material color
* 2021-03-27 [9f6e82c] avoid repainting preview
* 2021-03-26 [5109d29] add normal material, add box, subgrid and cylinder
* 2021-03-25 [ad0b814] draw grid from Domain.Dim
* 2021-03-25 [9b0cf95] fine tune fonts, add big tab initial screen, add svg background, add funding info
* 2021-03-25 [77f8f7a] add three.js for 3d preview
* 2021-03-24 [4274c77] rename mcxcloud.txt to mcxcloud
* 2021-03-24 [dc25a87] add mcx cloud service server and client files, partially working
* 2021-03-22 [f9bc07c] use tabs in mcxone, add jquery by default
* 2021-03-18 [d8b88e1] fix unwanted double-precision math functions
* 2021-03-11 [f6ce5bd] update variable and function name to follow the convention
* 2021-03-11 [ca2ce60] add example: comparison of surface diffuse reflectance between MC and Diffusion
* 2021-03-05 [bcbb324] change window sizes using 96dpi default setting
* 2021-03-05 [5c8d27f] fix Name shape object schema
* 2021-03-03 [02add69] MCX json schema and json editor are working, added more Shapes objects
* 2021-03-01 [940d725] wrapping up json input import feature in mcxstudio
* 2021-02-28 [64d629c] parse src/detector, media and shape
* 2021-02-27 [a3b8457]*open/import JSON input file in MCX Studio
* 2021-01-07 [9811c83] reorder the input data layout to match the change in preprocessing
* 2020-10-22 [991910e] add function comment and revert unnecessary changes
* 2020-10-22 [3343338]*add benchmarks from SVMC paper to mcxlab
* 2020-10-19 [de87cbf] resolve code alignment issue
* 2020-10-18 [5acd287] fix photon detection issue for SVMC mode (by Shijie Yan)
* 2020-10-18 [61dbf63] fix ray-tracing issue after the initial template implementation
* 2020-10-17 [fbb4f8c] initial implementation of template for SVMC mode (by Shijie Yan)
* 2020-10-08 [dad83c6] resolve conflict between two branches to elimate mismatch in demo_focus_mirror_bc.m
* 2020-10-08 [fb61782]*sync master branch into nuvox(SVMC) branch (by Shijie Yan)
* 2020-09-20 [75f08c5] remove empty depends
* 2020-09-20 [fa98229] fix incorrect dependency
* 2020-09-20 [d748d29] add octave package files for mcxlab and mcxtools
* 2020-09-16 [cf3b1f0] fix typo, change default exe path
* 2020-09-16 [15e9946]*fix warnings found by debian packaging at https://mentors.debian.net/package/mcx/
* 2020-09-16 [04bb0e7] add man pages for other binaries
* 2020-09-14 [aca9f97] remove additional debian packging warnings
* 2020-09-14 [ce4e341] add desktop icon files
* 2020-09-14 [eb0aa9f] allow new lines in string values in json
* 2020-09-14 [4b1301a] set default exe folder to /usr/libexec, fall back to ~/bin/
* 2020-09-14 [643e4a1]*add photon as unified cmd for mcx/mcxcl/mmc,polish for debian packaging
* 2020-09-14 [a67bc6d] updates to ease debian packaging
* 2020-09-08 [8983305] Inno Installer Setup paths and file details fixed
* 2020-09-07 [a6bc5a9] another attempt to fix #105
* 2020-09-07 [ca303dd] change default shortcut group name, fix #105
* 2020-09-06 [0313d4c] install mcxstudio to 64bit folder, close #105
* 2020-09-04 [37b4914] add demo script for mirror bc
* 2020-09-04 [e561890] make mcxplotvol work in matlab 2010 or earlier
* 2020-09-04 [9518cfa] handle mirror bc correctly, close #104
* 2020-09-04 [64896aa]*reset pattern center position following a failed launch, fix #103
* 2020-09-02 [5af2e76] fix -geometry 0x0 error, see https://forum.lazarus.freepascal.org/index.php?topic=40593.0
* 2020-09-01 [dd4be78] add cubesph60b to match example/benchmark2
* 2020-08-30 [971ffac] fix extended ascii letters
* 2020-08-29 [6eb9596] update mcxcreate.m, add mcxplotshapes.m to render json shapes
* 2020-08-29 [0199dad] clean up code and add comments for SVMC
* 2020-08-29 [94d55a7]*add mcxcreate, force mcxlab return one output
* 2020-08-28 [d917751] give an error for unsupported single dash option
* 2020-08-28 [093c9ba]*add pre-processing for SVMC mode
* 2020-08-28 [a79e116] add mode delphi in carbon unit
* 2020-08-27 [63e5a5f] handle det radii less than or equal to 0.5, fix #101
* 2020-08-27 [8f93ee2] fix make mex link error
* 2020-08-26 [65f0fe4] fix issrcfrom0 offset
* 2020-08-26 [79f9d70]*multiply voxelsize with det radius
* 2020-08-26 [d5c3c11] fix mcxpreview det radis issue, require srcpos and tend in mcxlab
* 2020-08-24 [1af5507] avoid error on mac
* 2020-08-24 [2fce8e5] add missing carbon unit for mac
* 2020-08-24 [6f11857] add command line option cheatsheet
* 2020-08-24 [5046de0] fix cmake command
* 2020-08-24 [cea663b] test cmake in travis
* 2020-08-24 [782b4a3] massive update of documentation
* 2020-08-24 [041e386] massive update to README to describe all output formats

Between 2020 and 2022, three new journal papers have been published as the 
result of this project, including [Yan2020]. Please see the full list at 
http://mcx.space/#publication

* [Yan2020] Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel based Monte Carlo
 algorithm for accurate and efficient photon transport modeling in complex bio-tissues," 
 Biomed. Opt. Express, 11(11) pp. 6262-6270.
* [Fang2022] Qianqian Fang, Shijie Yan, "MCX Cloud—a modern, scalable, high-performance
 and in-browser Monte Carlo simulation platform with cloud computing," J. Biomed. Opt.
 27(8) 083008, 2022
* [Yan2022] Shijie Yan, Steven L. Jacques, Jessica C. Ramella-Roman, Qianqian Fang, 
 "Graphics processing unit-accelerated Monte Carlo simulation of polarized light in 
 complex three-dimensional media," J. of Biomedical Optics, 27(8), 083015 (2022)

---------------------------------------------------------------------

== # Introduction ==

Monte Carlo eXtreme (MCX) is a fast photon transport simulation software for 3D 
heterogeneous turbid media. By taking advantage of the massively parallel 
threads and extremely low memory latency in a modern graphics processing unit 
(GPU), MCX is capable of performing Monte Carlo (MC) photon simulations at a 
blazing speed, typically hundreds to a thousand times faster than a fully 
optimized CPU-based MC implementation.

The algorithm of this software is detailed in the References 
[Fang2009,Yu2018,Yan2020]. A short summary of the main features includes:

* 3D heterogeneous media represented by voxelated array
* support over a dozen source forms, including wide-field and pattern illuminations
* boundary reflection support
* time-resolved photon transport simulations
* saving photon partial path lengths and trajectories
* optimized random number generators
* build-in flux/fluence normalization to output Green's functions
* user adjustable voxel resolution
* improved accuracy with atomic operations
* cross-platform graphical user interface
* native Matlab/Octave support for high usability
* flexible JSON interface for future extensions
* multi-GPU support

This software can be used on Windows, Linux and Mac OS. MCX is written in C/CUDA
and requires an NVIDIA GPU (support for AMD/Intel CPUs/GPUs via ROCm is still
under development). A more portable OpenCL implementation of MCX, i.e. MCXCL, 
was announced on July, 2012 and supports almost all NVIDIA/AMD/Intel CPU and GPU 
models. If your hardware does not support CUDA, please download MCXCL from the 
below URL:

  http://mcx.space/wiki/index.cgi?Learn#mcxcl

---------------------------------------------------------------------------
== # Requirement and Installation ==

Please read this section carefully. The majority of failures using MCX were 
found related to incorrect installation of NVIDIA GPU driver.

Please browse http://mcx.space/#documentation for step-by-step
instructions.

For MCX-CUDA, the requirements for using this software include

* a CUDA capable NVIDIA graphics card
* pre-installed NVIDIA graphics driver

You must install a CUDA capable NVIDIA graphics card in order to use
MCX. A list of CUDA capable cards can be found at [2]. The oldest 
graphics card that MCX supports is the Fermi series (circa 2010).
Using the latest NVIDIA card is expected to produce the best
speed. You must have a fermi (GTX 4xx) or newer 
(5xx/6xx/7xx/9xx/10xx/20xx series) graphics card. The default release 
of MCX supports atomic operations and photon detection. 
In the below webpage, we summarized the speed differences
between different generations of NVIDIA GPUs

http://mcx.space/gpubench/

For simulations with large volumes, sufficient graphics memory 
is also required to perform the simulation. The minimum amount of 
graphics memory required for a MC simulation is Nx*Ny*Nz
bytes for the input tissue data plus Nx*Ny*Nz*Ng*4 bytes for 
the output flux/fluence data - where Nx,Ny,Nz are the dimensions of the 
tissue volume, Ng is the number of concurrent time gates, 4 is 
the size of a single-precision floating-point number.
MCX does not require double-precision capability in your hardware.

To install MCX, you need to download the binary executable compiled for your 
computer architecture (32 or 64bit) and platform, extract the package 
and run the executable under the <mcx root>/bin directory. 

For Windows users, you must make sure you have installed the appropriate
NVIDIA driver for your GPU. You should also configure your OS to run
CUDA simulations. This requires you to open the mcx/setup/win64 folder
using your file explorer, right-click on the "apply_timeout_registry_fix.bat"
file and select "Run as administrator". After confirmation, you should 
see a windows command window with message 
<pre>
  Patching your registry
  Done
  Press any key to continue ...
</pre>

You MUST REBOOT your Windows computer to make this setting effective.
The above patch modifies your driver settings so that you can run MCX 
simulations for longer than a few seconds. Otherwise, when running MCX
for over a few seconds, you will get a CUDA error: "unspecified error".

Please see the below link for details

http://mcx.space/wiki/index.cgi?Doc/FAQ#I_am_getting_a_kernel_launch_timed_out_error_what_is_that

If you use Linux, you may enable Intel integrated GPU (iGPU) for display while
leaving your NVIDIA GPU dedicated for computing using `nvidia-prime`, see

https://forums.developer.nvidia.com/t/solved-run-cuda-on-dedicated-nvidia-gpu-while-connecting-monitors-to-intel-hd-graphics-is-this-possible/47690/6

or choose one of the 4 other approaches in this blog post

https://nvidia.custhelp.com/app/answers/detail/a_id/3029/~/using-cuda-and-x


== # Running Simulations ==

To run a simulation, the minimum input is a configuration (text) file, and, if
the input file does not contain built-in domain shape descriptions, an external
volume file (a binary file with a specified voxel format via `-K/--mediabyte`). 
Typing `mcx` without any parameters prints the help information and a list of 
supported parameters, as shown below:

<pre>###############################################################################
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #
#          Copyright (c) 2009-2021 Qianqian Fang <q.fang at neu.edu>          #
#                             http://mcx.space/                               #
#                                                                             #
# Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.org #
#   Department of Bioengineering, Northeastern University, Boston, MA, USA    #
###############################################################################
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #
###############################################################################
$Rev::e8fdb3$v2021.2$Date::2021-02-07 16:04:46 -05$ by $Author::Qianqian Fang $
###############################################################################

usage: mcx <param1> <param2> ...
where possible parameters include (the first value in [*|*] is the default)

== Required option ==
 -f config     (--input)       read an input file in .json or .inp format
                               if the string starts with '{', it is parsed as
			       an inline JSON input file
      or
 --bench ['cube60','skinvessel',..] run a buint-in benchmark specified by name
                               run --bench without parameter to get a list

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
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw
 -E [0|int|mch](--seed)        set random-number-generator seed, -1 to generate
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
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto
      or
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum
 -I            (--printgpu)    print GPU information and run program
 --atomic [1|0]                1: use atomic operations to avoid thread racing
                               0: do not use atomic operation (not recommended)

== Input options ==
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid.
                               only the root object named 'Shapes' is parsed 
			       and added to the existing domain defined via -f 
			       or --bench
 -j '{...}'    (--json)        a JSON string for modifying all input settings.
                               this input can be used to modify all existing 
			       settings defined by -f or --bench
 -K [1|int|str](--mediabyte)   volume data format, use either a number or a str
                               1 or byte: 0-128 tissue labels
			       2 or short: 0-65535 (max to 4000) tissue labels
			       4 or integer: integer tissue labels 
			      99 or labelplus: 32bit composite voxel format
                             100 or muamus_float: 2x 32bit floats for mua/mus
                             101 or mua_float: 1 float per voxel for mua
			     102 or muamus_half: 2x 16bit float for mua/mus
			     103 or asgn_byte: 4x byte gray-levels for mua/s/g/n
			     104 or muamus_short: 2x short gray-levels for mua/s
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array

== Output options ==
 -s sessionid  (--session)     a string to label all output file names
 -O [X|XFEJPM] (--outputtype)  X - output flux, F - fluence, E - energy deposit
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, 
			       event counts at each voxel (replay mode only)
                               M - momentum transfer; 
 -d [1|0]      (--savedet)     1 to save photon info at detectors; 0 not save
 -w [DP|DSPMXVW](--savedetflag)a string controlling detected photon data fields
    /case insensitive/         1 D  output detector ID (1)
                               2 S  output partial scat. even counts (#media)
                               4 P  output partial path-lengths (#media)
			       8 M  output momentum transfer (#media)
			      16 X  output exit position (3)
			      32 V  output exit direction (3)
			      64 W  output initial weight (1)
      combine multiple items by using a string, or add selected numbers together
      by default, mcx only saves detector ID and partial-path data
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions
                               setting -x to 1 also implies setting '-d' to 1.
			       same as adding 'XV' to -w.
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels
                               right outside of the domain; if non-zero voxels
			       appear at the boundary, pad 0s before using -X
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save.
                               same as adding 'M' to the -w flag
 -q [0|1]      (--saveseed)    1 to save photon RNG seed for replay; 0 not save
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save
 -H [1000000] (--maxdetphoton) max number of detected photons
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save
 -F [mc2|...] (--outputformat) fluence data output format:
                               mc2 - MCX mc2 format (binary 32bit float)
                               jnii - JNIfTI format (http://openjdata.org)
                               bnii - Binary JNIfTI (http://openjdata.org)
                               nii - NIfTI format
                               hdr - Analyze 7.5 hdr/img format
                               tx3 - GL texture data for rendering (GL_RGBA32F)
	the bnii/jnii formats support compression (-Z) and generate small files
	load jnii (JSON) and bnii (UBJSON) files using below lightweight libs:
	  MATLAB/Octave: JNIfTI toolbox   https://github.com/fangq/jnifti, 
	  MATLAB/Octave: JSONLab toolbox  https://github.com/fangq/jsonlab, 
	  Python:        PyJData:         https://pypi.org/project/jdata
	  JavaScript:    JSData:          https://github.com/fangq/jsdata
 -Z [zlib|...] (--zip)         set compression method if -F jnii or --dumpjson
                               is used (when saving data to JSON/JNIfTI format)
			       0 zlib: zip format (moderate compression,fast) 
			       1 gzip: gzip format (compatible with *.gz)
			       2 base64: base64 encoding with no compression
			       3 lzip: lzip format (high compression,very slow)
			       4 lzma: lzma format (high compression,very slow)
			       5 lz4: LZ4 format (low compression,extrem. fast)
			       6 lz4hc: LZ4HC format (moderate compression,fast)
 --dumpjson [-,0,1,'file.json']  export all settings, including volume data using
                               JSON/JData (http://openjdata.org) format for 
			       easy sharing; can be reused using -f
			       if followed by nothing or '-', mcx will print
			       the JSON to the console; write to a file if file
			       name is specified; by default, prints settings
			       after pre-processing; '--dumpjson 2' prints 
			       raw inputs before pre-processing

== User IO options ==
 -h            (--help)        print this message
 -v            (--version)     print MCX revision number
 -l            (--log)         print messages to a log file instead
 -i 	       (--interactive) interactive mode

== Debug options ==
 -D [0|int]    (--debug)       print debug information (you can use an integer
  or                           or a string by combining the following flags)
 -D [''|RMP]                   1 R  debug RNG
    /case insensitive/         2 M  store photon trajectory info
                               4 P  print progress bar
      combine multiple items by using a string, or add selected numbers together

== Additional options ==
 --root         [''|string]    full path to the folder storing the input files
 --gscatter     [1e9|int]      after a photon completes the specified number of
                               scattering events, mcx then ignores anisotropy g
                               and only performs isotropic scattering for speed
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch
 --maxvoidstep  [1000|int]     maximum distance (in voxel unit) of a photon that
                               can travel before entering the domain, if 
                               launched outside (i.e. a widefield source)
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),
                               use this parameter to set the maximum positions
                               stored (default: 1e7)

== Example ==
example: (list built-in benchmarks)
       mcx --bench
or (list supported GPUs on the system)
       mcx -L
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)
       mcx --bench cube60b -n 1e7 -G 1101 -W 10,10,10
or (use inline domain definition)
       mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
or (use inline json setting modifier)
       mcx -f input.json -j '{"Optode":{"Source":{"Type":"isotropic"}}}'
or (dump simulation in a single json file)
       mcx --bench cube60planar --dumpjson
</pre>

To further illustrate the command line options, below one can find a sample command

  mcx -A 0 -t 16384 -T 64 -n 1e7 -G 1 -f input.json -r 2 -s test -g 10 -d 1 -w dpx -b 1

the command above asks mcx to manually (`-A 0`) set GPU threads, and launch 16384 
GPU threads (`-t`) with every 64 threads a block (`-T`); a total of 1e7 photons (`-n`)
are simulated by the first GPU (`-G 1`) and repeat twice (`-r`) - i.e. total 2e7 photons;
the media/source configuration will be read from a JSON file named `input.json` 
(`-f`) and the output will be labeled with the session id “test” (`-s`); the 
simulation will run 10 concurrent time gates (`-g`) if the GPU memory can not 
simulate all desired time gates at once. Photons passing through the defined 
detector positions are saved for later rescaling (`-d`), and the saved photon
data include detector id (letter 'd' in -w), partial path (letter 'p' in -w) 
and exit position (letter 'x' in -w); refractive index mismatch is considered 
at media boundaries (`-b`).

Historically, MCX supports an extended version of the input file format (.inp)
used by tMCimg. However, we are phasing out the .inp support and strongly 
encourage users to adopt JSON formatted (.json) input files. Many of the 
advanced MCX options are only supported in the JSON input format.

A legacy .inp MCX input file looks like this:

<pre>
1000000              # total photon, use -n to overwrite in the command line
29012392             # RNG seed, negative to generate, use -E to overwrite
30.0 30.0 0.0 1      # source position (in grid unit), the last num (optional) sets --srcfrom0 (-z)
0 0 1 0              # initial directional vector, 4th number is the focal-length, 0 for collimated beam, nan for isotropic
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
</pre>

Note that the scattering coefficient mus=musp/(1-g).

The volume file (semi60x60x60.bin in the above example),
can be read in two ways by MCX: row-major[3] or column-major
depending on the value of the user parameter "-a". If the volume file
was saved using matlab or fortran, the byte order is column-major,
and you should use "-a 0" or leave it out of the command line. 
If it was saved using the fwrite() in C, the order is row-major, 
and you can either use "-a 1".

You may replace the binary volume file by a JSON-formatted shape file.
Please refer to Section V for details.

The time gate parameter is specified by three numbers:
start time, end time and time step size (in seconds). In 
the above example, the configuration specifies a total time 
window of [0 1] ns, with a 0.1 ns resolution. That means the 
total number of time gates is 10. 

MCX provides an advanced option, -g, to run simulations when 
the GPU memory is limited. It specifies how many time gates to simulate 
concurrently (when the GPU does not have sufficient memory to simulate 
all desired time gates all together). Users may want to limit that number 
to less than the total number specified in the input file - and by default 
it runs one gate at a time in a single simulation. But if there's 
enough memory based on the memory requirement in Section II, you can 
simulate all 10 time gates (from the above example) concurrently by using 
"-g 10" in which case you have to make sure the video card has at least  
60*60*60*10*5=10MB of free memory.   If you do not include the -g, 
MCX will assume you want to simulate just 1 time gate at a time.. 
If you specify a time-gate number greater than the total number in the 
input file, (e.g, "-g 20") MCX will stop when the 10 time-gates are 
completed. If you use the autopilot mode (-A), then the time-gates
are automatically estimated for you.


---------------------------------------------------------------------------
== # Using JSON-formatted input files ==

Starting from version 0.7.9, MCX accepts a JSON-formatted input file in
addition to the legacy .inp input files. JSON (JavaScript Object Notation) 
is a portable, human-readable and "fat-free" text format to represent 
complex and hierarchical data. Using the JSON format makes a input file 
self-explanatory, extensible and easy-to-interface with other applications 
(like MATLAB and Python).

A sample JSON input file can be found under the examples/quicktest
folder. The same file, qtest.json, is also shown below:
<pre>
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
		   zgaussian,line,slit,pencilarray,pattern3d",
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
</pre>

A JSON input file requiers several root objects, namely "Domain", "Session", 
"Forward" and "Optode". Other root sections, like "Help", will be ignored. 
Each object is a data structure providing information
indicated by its name. Each object can contain various sub-fields. 
The orders of the fields in the same level are flexible. For each field, 
you can always find the equivalent fields in the *.inp input files. 
For example, The "VolumeFile" field under the "Domain" object 
is the same as Line#6 in qtest.inp; the "RNGSeed" under "Session" is
the same as Line#2; the "Optode.Source.Pos" is the same as the 
triplet in Line#3; the "Forward.T0" is the same as the first number 
in Line#5, etc.

An MCX JSON input file must be a valid JSON text file. You can validate
your input file by running a JSON validator, for example http://jsonlint.com/
You should always use "" to quote a "name" and separate parallel
items by ",".

MCX accepts an alternative form of JSON input, but using it is not 
recommended. In the alternative format, you can use <tt>"rootobj_name.field_name": value</tt>
to represent any parameter directly in the root level. For example
<pre>
{
    "Domain.VolumeFile": "semi60x60x60.json",
    "Session.Photons": 10000000,
    ...
}
</pre>

You can even mix the alternative format with the standard format. 
If any input parameter has values in both formats in a single input 
file, the standard-formatted value has higher priority.

To invoke the JSON-formatted input file in your simulations, you 
can use the "-f" command line option with MCX, just like using an 
.inp file. For example:

  mcx -A 1 -n 20 -f onecube.json -s onecubejson

The input file must have a ".json" suffix in order for MCX to 
recognize. If the input information is set in both command line,
and input file, the command line value has higher priority
(this is the same for .inp input files). For example, when 
using "-n 20", the value set in "Session"/"Photons" is overwritten 
to 20; when using "-s onecubejson", the "Session"/"ID" value is modified.
If your JSON input file is invalid, MCX will quit and point out
where the format is incorrect.

---------------------------------------------------------------------------
== # Using JSON-formatted shape description files ==

Starting from v0.7.9, MCX can also use a shape 
description file in the place of the volume file.
Using a shape-description file can save you from making
a binary .bin volume. A shape file uses more descriptive 
syntax and can be easily understood and shared with others.

Samples on how to use the shape files are included under
the example/shapetest folder. 

The sample shape file, shapes.json, is shown below:
<pre>
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
</pre>

A shape file must contain a "Shapes" object in the root level.
Other root-level fields are ignored. The "Shapes" object is a
JSON array, with each element representing a 3D object or 
setting. The object-class commands include "Grid", "Sphere",
"Box" etc. Each of these object include a number of sub-fields
to specify the parameters of the object. For example, the 
"Sphere" object has 3 subfields, "O", "R" and "Tag". Field "O" 
has a value of 1x3 array, representing the center of the sphere; 
"R" is a scalar for the radius; "Tag" is the voxel values. 
The most useful command is "[XYZ]Layers". It contains a 
series of integer triplets, specifying the starting index, 
ending index and voxel value of a layered structure. If multiple
objects are included, the subsequent objects always overwrite 
the overlapping regions covered by the previous objects.

There are a few ways for you to use shape description records
in your MCX simulations. You can save it to a JSON shape file, and
put the file name in Line#6 of your .inp file, or set as the
value for Domain.VolumeFile field in a .json input file. 
In these cases, a shape file must have a suffix of .json.

You can also merge the Shapes section with a .json input file
by simply appending the Shapes section to the root-level object.
You can find an example, jsonshape_allinone.json, under 
examples/shapetest. In this case, you no longer need to define
the "VolumeFile" field in the input.

Another way to use Shapes is to specify it using the -P (or --shapes)
command line flag. For example:

 mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'

This will first initialize a volume based on the settings in the 
input .json file, and then rasterize new objects to the domain and 
overwrite regions that are overlapping.

For both JSON-formatted input and shape files, you can use
the JSONlab toolbox [4] to load and process in MATLAB.

---------------------------------------------------------------------------
== # Output data formats ==

MCX may produces several output files depending user's simulation settings.
Overall, MCX produces two types of outputs, 1) data accummulated within the 
3D volume of the domain (volumetric output), and 2) data stored for each detected
photon (detected photon data).

=== Volumetric output ===

By default, MCX stores a 4D array denoting the fluence-rate at each voxel in 
the volume, with a dimension of Nx*Ny*Nz*Ng, where Nx/Ny/Nz are the voxel dimension
of the domain, and Ng is the total number of time gates. The output data are
stored in the format of single-precision floating point numbers. One may choose
to output different physical quantities by setting the `-O` option. When the
flag `-X/--saveref` is used, the output volume may contain the total diffuse
reflectance only along the background-voxels adjacent to non-zero voxels. 
A negative sign is added for the diffuse reflectance raw output to distinguish
it from the fuence data in the interior voxels.

When photon-sharing (simultaneous simulations of multiple patterns) or photon-replay
(the Jacobian of all source/detector pairs) is used, the output array may be extended
to a 5D array, with the left-most/fastest index being the number of patterns Ns (in the
case of photon-sharing) or src/det pairs (in replay), denoted as Ns.

Several data formats can be used to store the 3D/4D/5D volumetric output. 

==== mc2 files ====

The `.mc2` format is simply a binary dump of the entire volumetric data output,
consisted of the voxel values (single-precision floating-point) of all voxels and
time gates. The file contains a continuous buffer of a single-precision (4-byte) 
5D array of dimension `Ns*Nx*Ny*Nz*Ng`, with the fastest index being the left-most 
dimension (i.e. column-major, similar to MATLAB/FORTRAN).

To load the mc2 file, one should call `loadmc2.m` and must provide explicitly
the dimensions of the data. This is because mc2 file does not contain the data
dimension information.

Saving to .mc2 volumetric file is depreciated as we are transitioning towards
JNIfTI/JData formatted outputs (.jnii). 

==== nii files ====

The NIfTI-1 (.nii) format is widely used in neuroimaging and MRI community to
store and exchange ND numerical arrays. It contains a 352 byte header, followed
by the raw binary stream of the output data. In the header, the data dimension
information as well as other metadata is stored. 

A .nii output file can be generated by using `-F nii` in the command line.

The .nii file is widely supported among data processing platforms, including
MATLAB and Python. For example
* niftiread.m/niftiwrite in MATLAB Image Processing Toolbox
* JNIfTI toolbox by Qianqian Fang (https://github.com/fangq/jnifti/tree/master/lib/matlab)
* PyNIfTI for Python http://niftilib.sourceforge.net/pynifti/intro.html

==== jnii files ====

The JNIfTI format represents the next-generation scientific data storage 
and exchange standard and is part of the OpenJData initiative (http://openjdata.org)
led by the MCX author Dr. Qianqian Fang. The OpenJData project aims at developing
easy-to-parse, human-readable and easy-to-reuse data storage formats based on
the ubiquitously supported JSON/binary JSON formats and portable JData data annotation
keywords. In short, .jnii file is simply a JSON file with capability of storing 
binary strongly-typed data with internal compression and built in metadata.

The format standard (Draft 1) of the JNIfTI file can be found at

https://github.com/fangq/jnifti

A .jnii output file can be generated by using `-F jnii` in the command line.

The .jnii file can be potentially read in nearly all programming languages 
because it is 100% comaptible to the JSON format. However, to properly decode
the ND array with built-in compression, one should call JData compatible
libraries, which can be found at http://openjdata.org/wiki

Specifically, to parse/save .jnii files in MATLAB, you should use
* JSONLab for MATLAB (https://github.com/fangq/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://github.com/fangq/jsonlab)

To parse/save .jnii files in Python, you should use
* PyJData module (https://pypi.org/project/jdata/) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['NIFTIData']` 
is a NumPy `ndarray` object storing the volumetric data.


==== bnii files ====

The binary JNIfTI file is also part of the JNIfTI specification and the OpenJData
project. In comparison to text-based JSON format, .bnii files can be much smaller
and faster to parse. The .bnii format is also defined in the BJData specification

https://github.com/fangq/bjdata

and is the binary interface to .jnii. A .bnii output file can be generated by 
using `-F bnii` in the command line.

The .bnii file can be potentially read in nearly all programming languages 
because it was based on UBJSON (Universal Binary JSON). However, to properly decode
the ND array with built-in compression, one should call JData compatible
libraries, which can be found at http://openjdata.org/wiki

Specifically, to parse/save .jnii files in MATLAB, you should use one of
* JSONLab for MATLAB (https://github.com/fangq/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://github.com/fangq/jsonlab)

To parse/save .jnii files in Python, you should use
* PyJData module (https://pypi.org/project/jdata/) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['NIFTIData']` 
is a NumPy `ndarray` object storing the volumetric data.

=== Detected photon data ===

If one defines detectors, MCX is able to store a variety of photon data when a photon
is captured by these detectors. One can selectively store various supported data fields,
including partial pathlengths, exit position and direction, by using the `-w/--savedetflag`
flag. The storage of detected photon information is enabled by default, and can be
disabled using the `-d` flag.

The detected photon data are stored in a separate file from the volumetric output.
The supported data file formats are explained below.

==== mch files ====

The .mch file, or MC history file, is stored by default, but we strongly encourage users
to adpot the newly implemented JSON/.jdat format for easy data sharing. 

The .mch file contains a 256 byte binary header, followed by a 2-D numerical array
of dimensions #savedphoton * #colcount as recorded in the header.

 typedef struct MCXHistoryHeader{
	char magic[4];                 /**< magic bits= 'M','C','X','H' */
	unsigned int  version;         /**< version of the mch file format */
	unsigned int  maxmedia;        /**< number of media in the simulation */
	unsigned int  detnum;          /**< number of detectors in the simulation */
	unsigned int  colcount;        /**< how many output files per detected photon */
	unsigned int  totalphoton;     /**< how many total photon simulated */
	unsigned int  detected;        /**< how many photons are detected (not necessarily all saved) */
	unsigned int  savedphoton;     /**< how many detected photons are saved in this file */
	float unitinmm;                /**< what is the voxel size of the simulation */
	unsigned int  seedbyte;        /**< how many bytes per RNG seed */
        float normalizer;              /**< what is the normalization factor */
	int respin;                    /**< if positive, repeat count so total photon=totalphoton*respin; if negative, total number is processed in respin subset */
	unsigned int  srcnum;          /**< number of sources for simultaneous pattern sources */
	unsigned int  savedetflag;     /**< number of sources for simultaneous pattern sources */
	int reserved[2];               /**< reserved fields for future extension */
 } History;

When the `-q` flag is set to 1, the detected photon initial seeds are also stored
following the detected photon data, consisting of a 2-D byte array of #savedphoton * #seedbyte.

To load the mch file, one should call `loadmch.m` in MATLAB/Octave.

Saving to .mch history file is depreciated as we are transitioning towards
JSON/JData formatted outputs (.jdat).

==== jdat files ====

When `-F jnii` is specified, instead of saving the detected photon into the legacy .mch format,
a .jdat file is written, which is a pure JSON file. This file contains a hierachical data
record of the following JSON structure

 {
   "MCXData": {
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
	   "nscat":
	   "ppath":
	   "mom":
	   "p":
	   "v":
	   "w0":
       },
       "Trajectory":{
           "photonid":
	   "p":
	   "w0":
       },
       "Seed":[
           ...
       ]
   }
 }

where "Info" is required, and other subfields are optional depends on users' input.
Each subfield in this file may contain JData 1-D or 2-D array constructs to allow 
storing binary and compressed data.

Although .jdat and .jnii have different suffix, they are both JSON/JData files and
can be opened/written by the same JData compatible libraries mentioned above, i.e.

For MATLAB
* JSONLab for MATLAB (https://github.com/fangq/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://github.com/fangq/jsonlab)

For Python
* PyJData module (https://pypi.org/project/jdata/) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['MCXData']['PhotonData']` 
stores the photon data, `data['MCXData']['Trajectory']` stores the trajectory data etc.


=== Photon trajectory data ===

For debugging and plotting purposes, MCX can output photon trajectories, as polylines,
when `-D M` flag is attached, or mcxlab is asked for the 5th output. Such information
can be stored in one of the following formats.

==== mct files ====

By default, MCX stores the photon trajectory data in to a .mct file MC trajectory, which
uses the same binary format as .mch but renamed as .mct. This file can be loaded to
MATLAB using the same `loadmch.m` function. 

Using .mct file is depreciated and users are encouraged to migrate to .jdat file
as described below.

==== jdat files ====

When `-F jnii` is used, MCX merges the trajectory data with the detected photon and
seed data and saved as a JSON-compatible .jdat file. The overall structure of the
.jdat file as well as the relevant parsers can be found in the above section.

---------------------------------------------------------------------------
== # Using MCXLAB in MATLAB and Octave ==

MCXLAB is the native MEX version of MCX for Matlab and GNU Octave. It includes
the entire MCX code in a MEX function which can be called directly inside
Matlab or Octave. The input and output files in MCX are replaced by convenient
in-memory struct variables in MCXLAB, thus, making it much easier to use
and interact. Matlab/Octave also provides convenient plotting and data
analysis functions. With MCXLAB, your analysis can be streamlined and speed-
up without involving disk files.

Please read the mcxlab/README.txt file for more details on how to
install and use MCXLAB.


---------------------------------------------------------------------------
== # Using MCX Studio GUI ==

MCX Studio is a graphics user interface (GUI) for MCX. It gives users
a straightforward way to set the command line options and simulation
parameters. It also allows users to create different simulation tasks 
and organize them into a project and save for later use.
MCX Studio can be run on many platforms such as Windows,
GNU Linux and Mac OS.

To use MCX Studio, it is suggested to put the mcxstudio binary
in the same directory as the mcx command; alternatively, you can
also add the path to mcx command to your PATH environment variable.

Once launched, MCX Studio will automatically check if mcx
binary is in the search path, if so, the "GPU" button in the 
toolbar will be enabled. It is suggested to click on this button
once, and see if you can see a list of GPUs and their parameters 
printed in the output field at the bottom part of the window. 
If you are able to see this information, your system is ready
to run MCX simulations. If you get error messages or not able
to see any usable GPU, please check the following:

* are you running MCX Studio/MCX on a computer with a supported card?
* have you installed the CUDA/NVIDIA drivers correctly?
* did you put mcx in the same folder as mcxstudio or add its path to PATH?

If your system has been properly configured, you can now add new simulations 
by clicking the "New" button. MCX Studio will ask you to give a session
ID string for this new simulation. Then you are allowed to adjust the parameters
based on your needs. Once you finish the adjustment, you should click the 
"Verify" button to see if there are missing settings. If everything looks
fine, the "Run" button will be activated. Click on it once will start your
simulation. If you want to abort the current simulation, you can click
the "Stop" button.

You can create multiple tasks with MCX Studio by hitting the "New"
button again. The information for all session configurations can
be saved as a project file (with .mcxp extension) by clicking the
"Save" button. You can load a previously saved project file back
to MCX Studio by clicking the "Load" button.


---------------------------------------------------------------------------
== # Interpreting the Output ==

MCX's output consists of two parts, the fluence volume file 
(.mc2, .nii, .jnii etc) and the detected photon data (.mch, .jdat etc).

=== Output files ===

An mc2/nii/jnii file contains the fluence-rate distributions from the simulation in 
the given medium. By default, this fluence-rate is a normalized solution 
(as opposed to the raw probability) therefore, one can compare this directly 
to analytical solutions (i.e. Green's function) of RTE/DE. The dimensions of the 
volume contained in this file are Nx, Ny, Nz, and Ng where Ng is the total number 
of time gates.

By default, MCX produces the '''Green's function''' of the 
'''fluence rate'''  for the given domain and source. Sometime it is also 
known as the time-domain "two-point" function. If you run MCX with the following command

  mcx -f input.json -s output ....

the fluence-rate data will be saved in a file named "output.mc2" under
the current folder. If you run MCX without "-s output", the
output file will be named as "input.json.dat".

To understand this further, you need to know that a '''fluence-rate (Phi(r,t))''' is
measured by number of particles passing through an infinitesimal 
spherical surface per '''unit time''' at '''a given location''' regardless of directions.
The unit of the MCX output is "W/mm<sup>2 = J/(mm<sup>2</sup>s)", if it is interpreted as the 
"energy fluence-rate" [6], or "1/(mm<sup>2</sup>s)", if the output is interpreted as the 
"particle fluence-rate" [6].

The Green's function of the fluence-rate means that it is produced
by a '''unitary source'''. In simple terms, this represents the 
fraction of particles/energy that arrives a location per second 
under '''the radiation of 1 unit (packet or J) of particle or energy 
at time t=0'''. The Green's function is calculated by a process referred
to as the "normalization" in the MCX code and is detailed in the 
MCX paper [6] (MCX and MMC outputs share the same meanings).

Please be aware that the output flux is calculated at each time-window 
defined in the input file. For example, if you type 

  0.e+00 5.e-09 1e-10  # time-gates(s): start, end, step

in the 5th row in the input file, MCX will produce 50 fluence-rate
snapshots, corresponding to the time-windows at [0 0.1] ns, 
[0.1 0.2]ns ... and [4.9,5.0] ns. To convert the fluence rate
to the fluence for each time-window, you just need to
multiply each solution by the width of the window, 0.1 ns in this case. 

To convert the time-dependent fluence-rate to continuous-wave (CW) 
fluence (fluence in short), you need to integrate the
fluence-rate along the time dimension. Assuming the fluence-rate after 
5 ns is negligible, then the CW fluence is simply sum(flux_i*0.1 ns, i=1,50). 
You can read <tt>mcx/examples/validation/plotsimudata.m</tt>
and <tt>mcx/examples/sphbox/plotresults.m</tt> for examples 
to compare an MCX output with the analytical fluence-rate/fluence solutions.

One can load an mc2 output file into Matlab or Octave using the
loadmc2 function in the <mcx root>/utils folder. 

To get a continuous-wave solution, run a simulation with a sufficiently 
long time window, and sum the flux along the time dimension, for 
example

   mcx=loadmc2('output.mc2',[60 60 60 10],'float');
   cw_mcx=sum(mcx,4);

Note that for time-resolved simulations, the corresponding solution
in the results approximates the flux at the center point
of each time window. For example, if the simulation time window 
setting is [t0,t0+dt,t0+2dt,t0+3dt...,t1], the time points for the 
snapshots stored in the solution file is located at 
[t0+dt/2, t0+3*dt/2, t0+5*dt/2, ... ,t1-dt/2]

A more detailed interpretation of the output data can be found at 
http://mcx.space/wiki/index.cgi?MMC/Doc/FAQ#How_do_I_interpret_MMC_s_output_data

MCX can also output "current density" (J(r,t), unit W/m^2, same as Phi(r,t)) -
referring to the expected number of photons or Joule of energy flowing
through a unit area pointing towards a particular direction per unit time.
The current density can be calculated at the boundary of the domain by two means:

# using the detected photon partial path output (i.e. the second output of mcxlab.m),
one can compute the total energy E received by a detector, then one can
divide E by the area/aperture of the detector to obtain the J(r) at a detector
(E should be calculated as a function of t by using the time-of-fly of detected
photons, the E(t)/A gives J(r,t); if you integrate all time gates, the total E/A
gives the current I(r), instead of the current density).

# use -X 1 or --saveref/cfg.issaveref option in mcx to enable the
diffuse reflectance recordings on the boundary. the diffuse reflectance
is represented by the current density J(r) flowing outward from the domain.

The current density has, as mentioned, the same unit as fluence rate,
but the difference is that J(r,t) is a vector, and Phi(r,t) is a scalar. Both measuring
the energy flow across a small area (the are has direction in the case of J) per unit
time.

You can find more rigorous definitions of these quantities in Lihong Wang's
Biomedical Optics book, Chapter 5.

=== Console print messages ===

Timing information is printed on the screen (stdout). The 
clock starts (at time T0) right before the initialization data is copied 
from CPU to GPU. For each simulation, the elapsed time from T0
is printed (in ms). Also the accumulated elapsed time is printed for 
all memory transaction from GPU to CPU.

When a user specifies "-D P" in the command line, or set `cfg.debuglevel='P'`,
MCX or MCXLAB prints a progress bar showing the percentage of completition.

---------------------------------------------------------------------------
== # Best practices guide ==

To maximize MCX's performance on your hardware, you should follow the
best practices guide listed below:

=== Use dedicated GPUs ===
A dedicated GPU is a GPU that is not connected to a monitor. If you use
a non-dedicated GPU, any kernel (GPU function) can not run more than a
few seconds. This greatly limits the efficiency of MCX. To set up a 
dedicated GPU, it is suggested to install two graphics cards on your 
computer, one is set up for displays, the other one is used for GPU 
computation only. If you have a dual-GPU card, you can also connect 
one GPU to a single monitor, and use the other GPU for computation
(selected by -G in mcx). If you have to use a non-dedicated GPU, you
can either use the pure command-line mode (for Linux, you need to 
stop X server), or use the "-r" flag to divide the total simulation 
into a set of simulations with less photons, so that each simulation 
only lasts a few seconds.

=== Launch as many threads as possible ===
It has been shown that MCX's speed is related to the thread number (-t).
Generally, the more threads, the better speed, until all GPU resources
are fully occupied. For higher-end GPUs, a thread number over 10,000 
is recommended. Please use the autopilot mode, "-A", to let MCX determine
the "optimal" thread number when you are not sure what to use.

---------------------------------------------------------------------------
== # Acknowledgement ==

=== cJSON library by Dave Gamble ===

* Files: src/cJSON folder
* Copyright (c) 2009 Dave Gamble
* URL: https://github.com/DaveGamble/cJSON
* License: MIT License, https://github.com/DaveGamble/cJSON/blob/master/LICENSE

=== GLScene library for Lazarus by GLScene developers ===

* Files: mcxstudio/glscene/*
* Copyright (c) GLScene developers
* URL: http://glscene.org, https://sourceforge.net/p/glscene/code/HEAD/tree/branches/GLSceneLCL/
* License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
* Comment: \
  A subset of the GLSceneLCL branch is included as part of the MCX source code tree \
  to allow compilation of the MCX Studio binary on various platforms without\
  needing to install the full package.

=== Texture3D sample project by Jürgen Abel ===

* Files: mcx/src/mcxstudio/mcxview.pas
* Copyright (c) 2003 Jürgen Abel
* License: Mozilla Public License 2.0 (MPL-2), https://sourceforge.net/p/glscene/code/HEAD/tree/trunk/LICENSE
* Comment: \
  The MCX volume renderer (mcxviewer) was adapted based on the Texture3D Example \
  provided by the GLScene Project (http://glscene.org). The original author of \
  this example is Jürgen Abel. 

=== Synapse communication library for Lazarus ===

* Files: mcxstudio/synapse/*
* Copyright (c) 1999-2017, Lukas Gebauer
* URL: http://www.ararat.cz/synapse/
* License: MIT License or LGPL version 2 or later or GPL version 2 or later
* Comment:\
  A subset of the Synapse units is included as part of the MCX source code tree \
  to allow compilation of the MCX Studio binary on various platforms without\
  needing to install the full package.

=== ZMat data compression unit ===

* Files: src/zmat/*
* Copyright: 2019-2020 Qianqian Fang
* URL: https://github.com/fangq/zmat
* License: GPL version 3 or later, https://github.com/fangq/zmat/blob/master/LICENSE.txt

=== LZ4 data compression library ===

* Files: src/zmat/lz4/*
* Copyright: 2011-2020, Yann Collet
* URL: https://github.com/lz4/lz4
* License: BSD-2-clause, https://github.com/lz4/lz4/blob/dev/lib/LICENSE

=== LZMA/Easylzma data compression library ===

* Files: src/zmat/easylzma/*
* Copyright: 2009, Lloyd Hilaiel, 2008, Igor Pavlov
* License: public-domain
* Comment: \
 All the cruft you find here is public domain.  You don't have to \
 credit anyone to use this code, but my personal request is that you mention \
 Igor Pavlov for his hard, high quality work.

=== myslicer toolbox by Anders Brun ===

* Files: utils/{islicer.m, slice3i.m, image3i.m}
* Copyright (c) 2009 Anders Brun, anders@cb.uu.se
* URL: https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume
* License: BSD-3-clause License, https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume#license_modal

=== MCX Filter submodule ===

* Files: filter/*
* Copyright (c) 2018 Yaoshen Yuan, 2018 Qianqian Fang
* URL: https://github.com/fangq/GPU-ANLM/
* License: MIT License, https://github.com/fangq/GPU-ANLM/blob/master/LICENSE.txt

=== pymcx Python module ===

* Files: pymcx/*
* Copyright (c) 2020  Maxime Baillot <maxime.baillot.1 at ulaval.ca>
* URL: https://github.com/fangq/GPU-ANLM/
* License: GPL version 3 or later, https://github.com/4D42/pymcx/blob/master/LICENSE.txt



---------------------------------------------------------------------------
== # Reference ==

* [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

* [Yu2018] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, \
"Scalable and massively parallel Monte Carlo photon transport simulations \
for heterogeneous computing platforms," J. Biomed. Opt. 23(1), 010504 (2018).

* [Yan2020] Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel based Monte Carlo \
algorithm for accurate and efficient photon transport modeling in complex bio-tissues," \
Biomed. Opt. Express, 11(11) pp. 6262-6270.

If you use MCX in your research, the author of this software would like
you to cite the above papers in your related publications.

Links: 

[1] http://developer.nvidia.com/cuda-downloads
[2] http://www.nvidia.com/object/cuda_gpus.html
[3] http://en.wikipedia.org/wiki/Row-major_order
[4] http://iso2mesh.sourceforge.net/cgi-bin/index.cgi?jsonlab
[5] http://science.jrank.org/pages/60024/particle-fluence.html
[6] http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-17-22-20178
