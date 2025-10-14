=                   Monte Carlo eXtreme  (MCX)  -- CUDA Edition                    =

*Author:  Qianqian Fang <q.fang at neu.edu>
*License: GNU General Public License version 3 (GPLv3)
*Version: 2.8 (v2025.10, Kilo-Kelvin)
*Website: https://mcx.space
*Download: https://mcx.space/wiki/?Get
---------------------------------------------------------------------

Table of Content:

<toc>

---------------------------------------------------------------------

== #  What's New ==

MCX v2025.10 is a maintenance release with multiple bug fixes and minor new features. It is highly
recommended to upgrade for all users.

Notable new features include:
* initial international language support (i18n), including Chinese (`zh_CN` and `zh_TW`),\
  French (`fr_CA`), Spanish (`es_MX`), German (`de_DE`), Japanese (`ja_JP`), Korean (`ko_KR`),\
  Hindi (`hi_IN`) and Portugues (`pt_BR`); use `--lang` in the command line or `cfg.lang`\
  or `cfg['lang']`, or use environment variable `MCX_LANG` to set output language;
* ported additional mcx utility functions from mcxlab to pmcx, including `cwdiffusion`,\
  `cwfluxdiffusion`, `cwfluencediffusion`, `dcsg1`, `mcxcreate`, `rfreplay`, `rfmusreplay`,\
  `loadmc2`, `loadmch`, `loadfile`, `mcx2json`, `json2mcx`, `loadnii`, `preview`,\
  `plotshapes`, `plotphotons`, `plotvol`; all new functions are unit-tested
* make mcx data processing functions in `pmcx` Python module pip-installable for Mac \
  users running on Apple silicon. The GPU simulation binary module (`_pmcx`) is not\
  supported on Apple silicon as it does not support CUDA.
* use replay to produce frequency-domain Jacobians - contributed by Paulliina Hirvi and Jarno Rantaharju

This release also contains a few bug fixes, including
* ensure time gate can not exceed gcfg->maxgate, fix #242 
* fix typos in pmcx functions
* package DLL files in the github action build script for mcxlab

We would like to give our special thank the following contributors
* @HirviP and @rantahar for their contributions of the new RF-replay feature added in this release (#241, #245)
* @ShawnSun1031 and @kuilef for the typo fixes


The detailed updates can be found in the below change log

* 2025-10-12 [4d6ca7f] [doc] update README, tag v2025.10
* 2025-10-12 [66128fd] [ci] fix opencl generic cpu error when testing packages
* 2025-10-11 [aa62f2b] [pmcx] bump pmcx to 0.6.1
* 2025-10-10 [d20953c] [inno] installer adds paths for iso2mesh or redbird
* 2025-10-10 [02f1b75] [ci] update release test actions
* 2025-10-06 [c285125] [pmcx] bump version to 0.6.0
* 2025-10-06 [b1bd7a8] [release] update release version to v2025.10
* 2025-10-05 [334ca2a] [mcxstudio] force enabling cthreads in linux
* 2025-10-05 [7bd0c04] [mcxstudio] correctly handle _ArrayOrder_=c, fix --lang for mcxcl/mmc
* 2025-10-01 [e4ed8f0] [feat] use USE_MCXCL>0 to set OpenCL device, negative to set CUDA device
* 2025-09-30 [da51af2] [bug] fix bug when individually specifying srcid>0 and replaydet>0, #250
* 2025-09-30 [96a728d] [bug] output 6D array when replaying multi-src w/ srcid=-1,replaydet=-1, fix #250
* 2025-09-30 [037366e] [feat] multiple source replay
* 2025-09-28 [76f8b4d] [i18n] use --lang=ID instead of --lang
* 2025-09-28 [c9b1439] [i18n] set default lang to en
* 2025-09-28 [5adace3] [i18n] set installer language tag
* 2025-09-28 [9fa04fd] [i18n] use locale folder that is already at root
* 2025-09-28 [66afb5c] [i18n] installer also setup locale files
* 2025-09-28 [4bab8ba] [typo] fix typo
* 2025-09-28 [975329c] [i18n] support French, German, Spanish, Italian, Japanese, Korean, Portuguese and Chinese
* 2025-09-28 [dbca123] [i18n] add GUI translations
* 2025-09-27 [fda913a] [bug] support more than 256 media count in output JSON file
* 2025-09-27 [2489d22] [inno] remove unsupported language
* 2025-09-27 [0756947] [inno] enable language selection
* 2025-09-26 [3d67580] [lang] support lang in the inno installer
* 2025-09-25 [e230514] [bug] when srctype is unknown, the program can run successfully
* 2025-08-26 [84adbfc] [pmcx] bump version to 0.5.1, some fixes after new unit tests
* 2025-08-26 [e12bbef]*[pmcx] add additional unit tests
* 2025-08-25 [521fcf0] [pmcx] pmcx v0.5.0
* 2025-08-25 [501d6b3] [pmcx] bump version 0.4.6 to update macos apple silicon src
* 2025-08-24 [37c1205]*[pmcx] initial port of all remaining utils matlab functions to python
* 2025-08-24 [101f3c7] [doc] fix inaccurate descrption of the 'energy' output type
* 2025-08-24 [196d1de] [pmcx] bump pmcx version to 0.4.3
* 2025-07-20 [fbf5942]*[ci] build none-any pmcx package for Apple silicon macos
* 2025-06-01 [6752c76] [i18n] updated es_MX translations verified by Edgar Guevara
* 2025-06-01 [bdc9a79] [lang] fix some octave warnings, improve translations
* 2025-06-01 [3ebff5b] [bug] use consistent message
* 2025-06-01 [8097604] [feat] read env variable MCX_LANG to set language
* 2025-06-01 [a2c8ef5] [bug] fix typos in translations, there could be more
* 2025-06-01 [fc7c1cf] [pmcx] bump version to 0.4.2 to support cfg['lang'], print lang name
* 2025-06-01 [8ece5d6] [pmcx] fix ci error for windows, support locale in pmcx, pass test
* 2025-06-01 [70958b9] [ci] fix windows vs error, make mcx_lang const visible in c++
* 2025-06-01 [b99a33f] [ci] fix macos build error, add missing new unit in cmake
* 2025-06-01 [729fc37]*[feat] initial i18n support for international languages
* 2025-05-12 [8663b80] Fix photontof index in WPTOF calculation
* 2025-04-21 [174e35c]*[feat] further polish rf replay for mus Jacobian, close #241
* 2025-04-21 [346a638] [feat] dynamically detect the lowest supported gpu arch by nvcc
* 2025-04-18 [0b3c364] Create demo script for computing any or all of the available Jacobian types in replay mode
* 2025-04-16 [512ea26] [pmcx] fix ci build errors
* 2025-04-16 [9ec9be1] Fix typo in pmcx.cpp
* 2025-04-15 [3e8b730] fix typo cfg.issaveexit to cfg["issaveexit"]
* 2025-04-12 [0bf148c]*[bug] ensure tshift can not exceed gcfg->maxgate, fix #242
* 2025-03-18 [98c92ea] Fix undefined variable cfg, should be mcx_config.
* 2025-03-18 [c1bc350] Format updated code and simplify example
* 2025-03-18 [d94e4f4] Add terminal output types for new features.
* 2025-03-17 [0d4c892] Fix typo in cfg struct name.
* 2025-03-17 [ded9aba]*Implement computation of the total time-of-flight (TOF) and final detected weight
* 2025-03-17 [eb4b171] Implement computation of total time-of-flight (TOF)- and final detected weight
* 2025-03-14 [aa7bbdd] [doc] fix typo in mcxlab, fix mcxpreview format
* 2025-03-12 [6d60f81]*[feat] accelerate mcxpreview using isosurface, must transpose
* 2025-03-10 [2eacfb8] [example] fix rf replay script to use .jdat output, fix #239
* 2025-03-06 [4e02a5b] Add object files to gitignore
* 2025-03-06 [c7df8d9] Fix memory error by unifying handling of otRF and otRFmus
* 2025-03-06 [7e7d88b] Fix function name
* 2025-03-06 [5f7d9c6] Fix compliation errors
* 2025-02-27 [7f05c3b] Remove redundant ifdef
* 2025-02-27 [c1b7c3c] Implement RFmus calculation
* 2025-02-27 [40882be] Add RFmus output type and start implementing calculation
* 2025-02-27 [7f09134] rename new example
* 2025-02-27 [6599a1c] Added new example functions for RF scattering Jacobians.
* 2025-02-23 [986ec46] [doc] post v2025 release action, close #238

---------------------------------------------------------------------

== # Introduction ==

Monte Carlo eXtreme (MCX) is a fast physically-accurate photon simulation 
software for 3D heterogeneous complex media. By taking advantage of 
the massively parallel threads and extremely low memory latency in a 
modern graphics processing unit (GPU), this program is able to perform Monte 
Carlo (MC) simulations at a blazing speed, typically hundreds to
a thousand times faster than a single-threaded CPU-based MC implementation.

MCX is written in C and NVIDIA CUDA. It only be executed on NVIDIA GPUs.
If you want to run hardware-accelerated MCX simulations on AMD/Intel GPUs
or CPUs, please download MCX-CL (MCX for OpenCL), which is written in OpenCL.
MCX and MCX-CL are highly compatible.

Due to the nature of the underlying MC algorithms, MCX and MCX-CL are
ray-tracing/ray-casting software under-the-hood. Compared to commonly
seen ray-tracing libraries used in computer graphics or gaming
engines, MCX-CL and MCX have many unique characteristics. The most
important difference is that MCX/MCX-CL are rigorously based on physical
laws. They are numerical solvers to the underlying radiative transfer equation
(RTE) and their solutions have been validated across many publications
using optical instruments and experimental measurements. In comparison,
most graphics-oriented ray-tracers have to make many approximations in
order to achieve fast rendering, enable to provide quantitatively accurate
light simulation results. Because of this, MCX/MCX-CL have been extensively
used by biophotonics research communities to obtain reference solutions and
guide the development of novel medical imaging systems or clinical
applications. Additionally, MCX/MCX-CL are volumetric ray-tracers; they
traverse photon-rays throughout complex 3-D domains and computes physically
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
* advanced features: photon-replay, photon-sharing, and more

This software can be used on Windows, Linux and Mac OS. MCX is written in C/CUDA
and requires NVIDIA GPUs (support for AMD/Intel CPUs/GPUs via ROCm is still
under development). A more portable OpenCL implementation of MCX, i.e. MCXCL, 
was announced on July, 2012 and supports almost all NVIDIA/AMD/Intel CPU and GPU 
models. If your hardware does not support CUDA, please download MCXCL from the 
below URL:

  https://mcx.space/wiki/index.cgi?Learn#mcxcl

---------------------------------------------------------------------------
== # Requirement and Installation ==

Please read this section carefully. The majority of failures using MCX were 
found related to incorrect installation of NVIDIA GPU driver.

Please browse https://mcx.space/#documentation for step-by-step
instructions.

For MCX-CUDA, the requirements for using this software include

* a CUDA capable NVIDIA graphics card
* pre-installed NVIDIA graphics driver

You must make sure that your NVIDIA graphics driver was installed properly.
A list of CUDA capable cards can be found at [2]. The oldest 
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
MC simulation is Nx*Ny*Nz bytes for the input tissue data plus 
Nx*Ny*Nz*Ng*4*2 bytes for the output flux/fluence data - where Nx,Ny,Nz are 
the dimensions of the tissue volume, Ng is the number of concurrent time gates, 
4 is the size of a single-precision floating-point number, 2 is for the extra memory
needed to ensure output accuracy (https://github.com/fangq/mcx/issues/41). MCX does not require 
double-precision support in your hardware.

MCX stores optical properties and detector positions in the constant memory.
Usually, NVIDIA GPUs provides about 64 kB constant memory. As a result, we can only
the total number of optical properties plus the number of detectors can not
exceed 4000 (4000 * 4 * 4 = 64 k).

In addition, MCX stores detected photon data inside the shared memory, which also ranges
between 42 kB to 100 kB per stream processor across different GPU generations. 
If your domain contains many medium types, it is possible that the allocation of
the shared memory can exceed the limit. You will also receive an "out of memory" error.

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

  sudo prime-select nvidia

to achieve the same goal. Otherwise, the simulation may hang your system
after running for a few seconds. A hybrid GPU laptop combing an NVIDIA GPU 
with an AMD iGPU does not seem to have this issue if using Linux.

In addition, NVIDIA drirver (520 or newer) has a known glitch running on Linux kernel
6.x (such as those in Ubuntu 22.04). See

https://forums.developer.nvidia.com/t/dev-nvidia-uvm-io-error-on-ubuntu-22-04-520-to-535-driver-versions/262153

When the laptop is in the "performance" mode and wakes up from suspension, MCX or any
CUDA program fails to run with an error


  MCX ERROR(-999):unknown error in unit mcx_core.cu:2523


This is because the kernel module `nvida-uvm` fails to be reloaded after suspension.
If you had an open MATLAB session, you must close MATLAB first, and
run the below commands (if MATLAB is open, you will see `rmmod: ERROR: Module nvidia_uvm is in use`)


  sudo rmmod nvidia-uvm
  sudo modprobe nvidia-uvm


after the above command, MCX should be able to run again.

New generations of Mac computers no longer support NVIDIA or AMD GPUs. you will
have to use the OpenCL version of MCX, MCX-CL by downloading it from

https://mcx.space/wiki/?Learn#mcxcl


== # Running Simulations ==

To run a simulation, the minimum input is a configuration (text) file, and, if
the input file does not contain built-in domain shape descriptions, an external
volume file (a binary file with a specified voxel format via `-K/--mediabyte`). 
Typing `mcx` without any parameters prints the help information and a list of 
supported parameters, as shown below:

<pre>###############################################################################
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #
#          Copyright (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>          #
#                https://mcx.space/  &  https://neurojson.io                  #
#                                                                             #
# Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.org #
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
$Rev::d20953$v2025.10$Date::2025-10-10 23:08:18 -04$ by $Author::Qianqian Fang$
###############################################################################

usage: mcx <param1> <param2> ...
where possible parameters include (the first value in [*|*] is the default)

== Required option ==
 -f config     (--input)       read an input file in the .json format,if config
                               string starts with '{',it is parsed as an inline
                               JSON input file; if -f is followed by nothing or
                               a single '-', it reads input from stdin via pipe
      or
 -Q/--bench [cube60, skinvessel,...] run a buint-in benchmark specified by name
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
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw
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
 -O [X|XFEJPMRLSTB](--outputtype) X - output flux, F - fluence, E - energy
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, 
                               event counts at each voxel (replay mode only)
                               M - momentum transfer; R - RF/FD Jacobian
                               L - total pathlength; S - RF/FD mus Jacobian
                               T - time-of-flight*nscat;B - time-of-flight*path
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
    load jnii (JSON) and bnii (UBJSON) files using below lightweight libs:
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

Another way to use Shapes is to specify it using the `-P` (or `--shapes`)
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

Starting in MCX v2023, mc2 files are no longer the default output format for
MCX binary. Instead, JSON based JNIfTI (.jnii) files are used.

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
* JNIfTI toolbox by Qianqian Fang (https://github.com/NeuroJSON/jnifti/tree/master/lib/matlab)
* PyNIfTI for Python http://niftilib.sourceforge.net/pynifti/intro.html

==== jnii files ====

Starting in MCX v2023, JSON based JNIfTI (.jnii) files are used as the default
volumetric data output format.

The JNIfTI format represents the next-generation scientific data storage 
and exchange standard and is part of the US NIH-funded NeuroJSON project (https://neurojson.org)
led by the MCX author Dr. Qianqian Fang. The NeuroJSON project aims at developing
easy-to-parse, human-readable and easy-to-reuse data storage formats based on
the ubiquitously supported JSON/binary JSON formats and portable JData data annotation
keywords. In short, .jnii file is simply a JSON file with capability of storing 
binary strongly-typed data with internal compression and built-in metadata.

The format standard (Draft 1) of the JNIfTI file can be found at

https://github.com/NeuroJSON/jnifti

A .jnii output file can be generated by using `-F jnii` in the command line.

The .jnii file can be potentially read in nearly all programming languages 
because it is 100% comaptible to the JSON format. However, to properly decode
the ND array with built-in compression, one should call JData compatible
libraries, which can be found at https://neurojson.org/#software

Specifically, to parse/save .jnii files in MATLAB, you should use
* JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

To parse/save .jnii files in Python, you should use
* PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

In Python, the volumetric data is loaded as a `dict` object where `data['NIFTIData']` 
is a NumPy `ndarray` object storing the volumetric data.


==== bnii files ====

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
* JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

To parse/save .jnii files in Python, you should use
* PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

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
* JSONLab for MATLAB (https://neurojson.org/download/jsonlab) or install `octave-jsonlab` on Fedora/Debian/Ubuntu
* `jsonencode/jsondecode` in MATLAB + `jdataencode/jdatadecode` from JSONLab (https://neurojson.org/download/jsonlab)

For Python
* PyJData module (https://neurojson.org/download/pyjdata) or install `python3-jdata` on Debian/Ubuntu

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

MCXLAB is the native MEX version of MCX for MATLAB and GNU Octave. It includes 
the entire MCX code in a MEX function which can be called directly inside 
MATLAB or Octave. The input and output files in MCX are replaced by convenient 
in-memory struct variables in MCXLAB, thus, making it much easier to use and 
interact. MATLAB/Octave also provides convenient plotting and data analysis 
functions. With MCXLAB, your analysis can be streamlined and simplified without 
involving disk files.

Please read the `mcxlab/README.txt` file for more details on how to install and 
use MCXLAB.

Please also browse this interactive [https://colab.research.google.com/github/fangq/mcx/blob/master/mcxlab/tutorials/mcxlab_getting_started.ipynb Jupyter Notebook based MCXLAB tutorial]
to see a suite of examples showing the key functionalities of MCXLAB (using GNU Octave).

---------------------------------------------------------------------------
== # Using PMCX in Python ==

PMCX is the native binary binding of MCX for Python 3.6 or newer. Similar to
MCXLAB, PMCX can run GPU-based simulations inside Python environment with
efficient in-memory inputs and outputs. 

Please read the `pmcx/README.txt` file for more details on how to install and 
use PMCX.

Please also browse this interactive [https://colab.research.google.com/github/fangq/mcx/blob/master/pmcx/tutorials/pmcx_getting_started.ipynb Jupyter Notebook based PMCX tutorial]
to see a suite of examples showing the key functionalities of PMCX.


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
You can read `mcx/examples/validation/plotsimudata.m`
and `mcx/examples/sphbox/plotresults.m` for examples 
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
https://mcx.space/wiki/index.cgi?MMC/Doc/FAQ#How_do_I_interpret_MMC_s_output_data

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

# use `-X 1` or `--saveref/cfg.issaveref` option in mcx to enable the
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

=== Use a middle-range or enthusiastic-grade GPU, use multiple of them if possible ===

MCX is highly scalable, providing linear-speedup as long as you provide the
GPU cores it can use. As a result, the better the GPU you use, the higher the speed
you can get. An enthusiastic-grade GPU, such as RTX 4070Ti (~$700), can be 12x
faster than an low-end laptop RTX 4050 GPU even within the same generation.

MCX can readily take advantage of multiple GPUs if you have it installed. The
MCX simulation speed scales nearly linearly as the number of GPUs increases.
So, to maximize MCX performance, get at least a middle-level or high-end consumer
grade GPU; if you need more speed, throw in more GPUs will cut down the runtime.

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
* Copyright: 2019-2023 Qianqian Fang
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

=== Pybind11 ===
* Files: src/pybind11/*
* Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
* URL: https://github.com/pybind/pybind11/
* License: BSD-style license, https://github.com/pybind/pybind11/blob/master/LICENSE

---------------------------------------------------------------------------
== # Reference ==

* [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units," \
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

* [1] http://developer.nvidia.com/cuda-downloads
* [2] http://www.nvidia.com/object/cuda_gpus.html
* [3] http://en.wikipedia.org/wiki/Row-major_order
* [4] https://neurojson.org/jsonlab
* [5] http://science.jrank.org/pages/60024/particle-fluence.html
* [6] http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-17-22-20178
