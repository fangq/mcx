= MCXLAB: MCX for MATLAB and GNU Octave =

Author: Qianqian Fang <q.fang at neu.edu>
License: GNU General Public License version 3 (GPLv3)
Version: this package is part of Monte Carlo eXtreme (MCX) v2026

<toc>


== # Introduction ==

MCXLAB is the native MEX version of MCX for MATLAB and GNU Octave. It compiles
the entire MCX code into a MEX function which can be called directly inside
MATLAB or Octave. The input and output files in MCX are replaced by convenient
in-memory struct variables in MCXLAB, thus, making it much easier to use 
and interact. MATLAB/Octave also provides convenient plotting and data
analysis functions. With MCXLAB, your analysis can be streamlined and sped
up without involving disk files.

Because MCXLAB contains the exact computational codes for the GPU calculations
as in the MCX binaries, MCXLAB is expected to have identical performance when
running simulations compared to the standalone version of MCX.


== # Installation ==

To download MCXLAB, please visit [https://mcx.space/wiki/index.cgi?Download#Download_the_Latest_Release this link]. 
If you choose to [https://mcx.space/wiki/index.cgi?register/mcx register],
you will have an option to be notified for any future updates including
critical bug fixes and new milestone releases.

The system requirements for MCXLAB are the same as MCX: you have to make
sure that you have a CUDA-capable graphics card with properly configured 
GPU driver (you can run the standard MCX binary first to test if your 
system is capable to run MCXLAB). Of course, you need to have either MATLAB
or Octave installed.

The released mex files are statically linked with the CUDA runtime, so you
only need an up-to-date NVIDIA driver; the CUDA toolkit is not needed. The
Octave mex files released for Windows and macOS are linked against liboctmex
and require Octave 10 or newer; the Linux Octave mex file works with any
recent Octave version. On computers without an NVIDIA GPU, mcxlab can run
the simulation with MCXLAB-CL (mcxlabcl, the OpenCL version) if it is
installed, using mcxlab(cfg,'opencl') or by setting USE_MCXCL=1 (see below).

Once the NVIDIA driver is set up, you can then add the 
"mcxlab" directory to your MATLAB/Octave search path using the addpath command.
If you want to add this path permanently, please use the "pathtool" 
command, or edit your `startup.m` (`~/.octaverc` for Octave).

If everything works ok, typing `mcxlab('gpuinfo')` in MATLAB/Octave will print the
supported GPUs. If you see any error, particularly any missing libraries,
please make sure you have downloaded the matching version built for your
platform.


== # How to use MCXLAB in MATLAB/Octave ==

To learn the basic usage of MCXLAB, you can type

  help mcxlab

in MATLAB/Octave to see the help information regarding how to use this 
function. The help information is listed below. You can find the input/output 
formats and examples. The input cfg structure has very similar field names as
the verbose command line options in MCX.

<pre> 
  ====================================================================
       MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave
  --------------------------------------------------------------------
  Copyright (c) 2011-2026 Qianqian Fang <q.fang at neu.edu>
                       URL: https://mcx.space
  ====================================================================
 
  Format:
     fluence=mcxlab(cfg);
        or
     [fluence,detphoton,vol,seed,trajectory]=mcxlab(cfg);
     [fluence,detphoton,vol,seed,trajectory]=mcxlab(cfg, option);
 
  Input:
     cfg: a struct, or struct array. Each element of cfg defines
          the parameters associated with a simulation.
          if cfg='gpuinfo': return the supported GPUs and their parameters,
          if cfg='version': return the version of MCXLAB as a string,
          see sample script at the bottom
     option: (optional), options is a string, specifying additional options
          option='preview': this plots the domain configuration using mcxpreview(cfg)
          option='opencl':  force using mcxcl.mex* instead of mcx.mex* on NVIDIA/AMD/Intel hardware
          option='cuda':    force using mcx.mex* instead of mcxcl.mex* on NVIDIA GPUs
 
     if one defines USE_MCXCL=1 in MATLAB command line window, all following
     mcxlab and mcxlabcl calls will use mcxcl.mex; by setting option='cuda', one can
     force both mcxlab and mcxlabcl to use mcx (cuda version). Similarly, if
     USE_MCXCL=0, all mcxlabcl and mcxlab call will use mcx.mex by default, unless
     one sets option='opencl'. When USE_MCXCL is set to a positive integer
     or a string, it overwrites cfg.gpuid in mcxlabcl to specify the device
     to run the simulation.
 
     cfg may contain the following fields:
 
  == Required ==
      *cfg.nphoton:    the total number of photons to be simulated (integer)
                       maximum supported value is 2^63-1
      *cfg.vol:        a 3D array specifying the media index in the domain.
                       can be uint8, uint16, uint32, single or double
                       arrays.
                       2D simulations are supported if cfg.vol has a singleton
                       dimension (in x or y); srcpos/srcdir must belong to
                       the 2D plane in such case.
                       for 2D simulations, Example: <demo_mcxlab_2d.m>
 
                       MCXLAB also accepts 4D arrays to define continuously varying media.
                       The following formats are accepted
                         1 x Nx x Ny x Nz float32 array: "mua_float"/101 format: mua values for each voxel (must use permute to make 1st dimension singleton)
                         2 x Nx x Ny x Nz float32 array: "muamus_float"/100 format: mua/mus values for each voxel (g/n use prop(2,:))
                         3 x Nx x Ny x Nz float32 array: "labelplus"/99 format: {[float32: property_value][float32: property_index i, 0-3][float32: tissue label]},
                                          the optical properties are first read based on the label, then the i-th property is replaced by the property value
                         3 x Nx x Ny x Nz int16/uint16 array:  "mixlabel"/98 format: {[int16 label1][uint16 label2][0-65535 label1 percentage]}
                         4 x Nx x Ny x Nz uint8 array: "asgn_byte"/103 format: mua/mus/g/n gray-scale (0-255) interpolating between prop(2,:) and prop(3,:)
                         4 x Nx x Ny x Nz single array: "asgn_float"/96 format: per-voxel mua/mus/g/n floating point values (internally convert to half-precision)
                         2 x Nx x Ny x Nz uint16 array: "muamus_short"/104 format: mua/mus gray-scale (0-65535) interpolating between prop(2,:) and prop(3,:)
                         8 x Nx x Ny x Nz uint8 array: "svmc"/97 format: split-voxel MC (SVMC) hybrid domain, can be created by mcxsvmc.m
                         Example: <demo_continuous_mua_mus.m>. If voxel-based media are used, partial-path/momentum outputs are disabled
                         Example: <demo_svmc_*.m> for SVMC related examples
      *cfg.prop:       an N by 4 array, each row specifies [mua, mus, g, n] in order.
                       the first row corresponds to medium type 0
                       (background) which is typically [0 0 1 1]. The
                       second row is type 1, and so on. The background
                       medium (type 0) has special meanings: a photon is terminated
                       when moving from a non-zero to zero voxel.
      *cfg.tstart:     starting time of the simulation (in seconds)
      *cfg.tstep:      time-gate width of the simulation (in seconds)
      *cfg.tend:       ending time of the simulation (in second)
      *cfg.srcpos:     a 1 by 3 vector, the position of the source in grid unit; if a non-zero
                       4th element is given, it specifies the initial weight of each photon packet
                       (or a multiplier in the cases of pattern/pattern3d sources); this initial weight
                       can be negative - the output fluence is expected to be linearly proportional
                       to this initial weight (w0) before normalization.
      *cfg.srcdir:     a 1 by 3 vector, specifying the incident vector; if srcdir
                       contains a 4th element, it specifies the focal length of
                       the source (only valid for focusable src, such as planar, disk,
                       fourier, gaussian, pattern, slit, etc); if the focal length
                       is nan, all photons will be launched isotropically regardless
                       of the srcdir direction; if the focal length is -inf, the launch
                       angle will be computed based on the Lambertian (cosine) distribution,
                       over a hemisphere in 3D, or over a semicircle within the simulation
                       plane in 2D (when one dimension of cfg.vol is 1).
 
       Starting v2024, cfg.{srcpos,srcdir,srcparam1,srcparam2} accept multiple sources,
       with each source corresponding to a single row of the array. For all 4 components,
       srcpos/srcdir support 3 or 4 columns, and srcparam1/srcparam2 support 4 columns.
       If any of the 4 components are present, they should have the same number of rows.
 
  == MC simulation settings ==
       cfg.seed:       seed for the random number generator (integer) [default 1648335518]
                       setting seed to a negative integer or 0 uses system clock as seed;
                       a fixed seed reproduces a run only with cfg.autopilot=0;
                       if set to a uint8 array, the binary data in each column is used
                       to seed a photon (i.e. the "replay" mode)
                       Example: <demo_mcxlab_replay.m>
       cfg.respin:     repeat simulation for the given time (integer) [1]
                       if negative, divide the total photon number into respin subsets
       cfg.isreflect:  [1]-consider refractive index mismatch, 0-matched index
       cfg.bc          per-face boundary condition (BC), a string of 6 letters (case insensitive) for
                       bounding box faces at -x,-y,-z,+x,+y,+z axes;
                       overwrite cfg.isreflect if given.
                       each letter can be one of the following:
                       '_': undefined, fallback to cfg.isreflect
                       'r': like cfg.isreflect=1, Fresnel reflection BC
                       'a': like cfg.isreflect=0, total absorption BC
                       'm': mirror or total reflection BC
                       'c': cyclic BC, enter from opposite face
 
                       in addition, cfg.bc can contain up to 12 characters,
                       with the 7-12 characters indicating bounding box
                       facets -x,-y,-z,+x,+y,+z are used as a detector. The
                       acceptable characters for digits 7-12 include
                       '0': this face is not used to detector photons
                       '1': this face is used to capture photons (if output detphoton)
                       see <demo_bc_det.m>
       cfg.isnormalized:[1]-normalize the output fluence to unitary source, 0-no normalization.
                       setting isnormalized to 2 in the replay mode builds the Jacobian
                       with Born approximation instead of the default Rytov approximation
       cfg.isspecular: 1-calculate specular reflection if source is outside, [0] no specular reflection
       cfg.maxgate:    the num of time-gates per simulation
       cfg.minenergy:  terminate photon when weight less than this level (float) [0.0]
       cfg.unitinmm:   defines the length unit for a grid edge length [1.0]
                       Example: <demo_sphere_cube_subpixel.m>
       cfg.shapes:     a JSON string for additional shapes in the grid
                       Example: <demo_mcxyz_skinvessel.m>
       cfg.invcdf:     user-specified scattering phase function. To use this, one must define
                       a vector with monotonically increasing value between -1 and 1
                       defining the discretized inverse function of the cumulative density function (CDF)
                       of u=cos(theta), i.e. inv(CDF(u))=inv(CDF(cos(theta))), where theta [0-pi] is the
                       zenith angle of the scattering event, and u=cos(theta) is between -1 and 1.
                       Please note that the defined inv(CDF) is relative to u, not to theta. This is because
                       it is relatively easy to compute as P(u) is the primary form of phase function
                       see <demo_mcxlab_phasefun.m>
       cfg.angleinvcdf: user-specified launch angle distribution. To use this, one must define
                       a vector with monotonically increasing value between 0 and 1
                       defining the discretized inverse function of the
                       cumulative density function (CDF) of (theta/pi),
                       i.e. inv(CDF(theta/pi)), where theta in the range of
                       [0-pi] is the zenith angle of the launch angle relative to cfg.srcdir.
 
                       when source focal-length, cfg.srcdir(4), is set to 0
                       (default), the angleinvcdf is randomly sampled with
                       linear interpolation, i.e. every sample uses two
                       elements to interpolate the desired launch angle;
                       this is suited when the distribution is continuous.
 
                       when cfg.srcdir(4) is set to 1, the angleinvcdf
                       vector is randomly sampled without interpolation
                       (i.e. only return the theta/pi values specified
                       inside this array); this is best suited for discrete
                       angular distribution.
                       see <demo_mcxlab_launchangle.m>
       cfg.gscatter:   after a photon completes the specified number of
                       scattering events, mcx then ignores anisotropy g
                       and only performs isotropic scattering for speed [1e9]
       cfg.detphotons: detected photon data for replay. In the replay mode (cfg.seed
                       is set as the 4th output of the baseline simulation), cfg.detphotons
                       should be set to the 2nd output (detphoton) of the baseline simulation
                       or detphoton.data subfield (as a 2D array). cfg.detphotons can use
                       a subset of the detected photon selected by the user.
                       Example: <demo_mcxlab_replay.m>
       cfg.polprop:    an N by 5 array, each row specifies [mua, radius(micron), volume
                       density(1/micron^3), sphere refractive index, ambient medium
                       refractive index] in order. The first row is type 1,
                       and so on. The background medium (type 0) should be
                       defined in the first row of cfg.prop. For polprop type
                       i, if prop(i,2) is not zero: 1) if prop(i,3) == 1, the
                       density polprop(i,3) will be adjusted to achieve the target
                       mus prop(i,2); 2) if prop(i,3) < 1, polprop(i,3) will be
                       adjusted to achieve the target mus' prop(i,2)*(1-prop(i,3))
 
  == GPU settings ==
       cfg.autopilot:  1-automatically set threads and blocks, [0]-use nthread/nblocksize
                       with 1, photons are also dispatched to the threads dynamically,
                       so runs with the same cfg.seed agree statistically but not
                       bit-wise; set 0 to reproduce a run exactly
       cfg.nblocksize: how many CUDA thread blocks to be used [64]
       cfg.nthread:    the total CUDA thread number [2048]
       cfg.gpuid:      which GPU to use (run 'mcx -L' to list all GPUs) [1]
                       if set to an integer, gpuid specifies the index (starts at 1)
                       of the GPU for the simulation; if set to a binary string made
                       of 1s and 0s, it enables multiple GPUs. For example, '1101'
                       allows to use the 1st, 2nd and 4th GPUs together.
                       Example: <mcx_gpu_benchmarks.m>
       cfg.workload    an array denoting the relative loads of each selected GPU.
                       for example, [50,20,30] allocates 50%, 20% and 30% photons to the
                       3 selected GPUs, respectively; [10,10] evenly divides the load
                       between 2 active GPUs. A simple load balancing strategy is to
                       use the GPU core counts as the weight.
       cfg.isgpuinfo:  1-print GPU info, [0]-do not print
 
  == Source-detector parameters ==
       cfg.detpos:     an N by 4 array, each row specifying a detector: [x,y,z,radius]
       cfg.maxdetphoton:   maximum number of photons saved by the detectors [1000000]
       cfg.srctype:    source type, the parameters of the src are specified by cfg.srcparam{1,2}
                               Example: <demo_mcxlab_srctype.m>
                       'pencil' - default, pencil beam, no param needed
                       'isotropic' - isotropic source, no param needed
                       'cone' - uniform cone beam, srcparam1(1) is the half-angle in radian
                       'gaussian' [*] - a collimated gaussian beam, srcparam1(1) specifies the waist radius (in voxels)
                       'hyperboloid' [*] - a one-sheeted hyperboloid gaussian beam, srcparam1(1) specifies the waist
                                 radius (in voxels), srcparam1(2) specifies distance between launch plane and focus,
                                 srcparam1(3) specifies rayleigh range
                       'planar' [*] - a 3D quadrilateral uniform planar source, with three corners specified
                                 by srcpos, srcpos+srcparam1(1:3) and srcpos+srcparam2(1:3)
                       'pattern' [*] - a 3D quadrilateral pattern illumination, same as above, except
                                 srcparam1(4) and srcparam2(4) specify the pattern array x/y dimensions,
                                 and srcpattern is a floating-point pattern array, with values between [0-1].
                                 if cfg.srcnum>1, srcpattern must be a floating-point array with
                                 a dimension of [srcnum srcparam1(4) srcparam2(4)]
                                 Example: <demo_photon_sharing.m>
                       'pattern3d' [*] - a 3D illumination pattern. srcparam1{x,y,z} defines the dimensions,
                                 and srcpattern is a floating-point pattern array, with values between [0-1].
                       'fourier' [*] - spatial frequency domain source, similar to 'planar', except
                                 the integer parts of srcparam1(4) and srcparam2(4) represent
                                 the x/y frequencies; the fraction part of srcparam1(4) multiplies
                                 2*pi represents the phase shift (phi0); 1.0 minus the fraction part of
                                 srcparam2(4) is the modulation depth (M). Put in equations:
                                     S=0.5*[1+M*cos(2*pi*(fx*x+fy*y)+phi0)], (0<=x,y,M<=1)
                       'arcsine' - similar to isotropic, except the zenith angle is uniform
                                 distribution, rather than a sine distribution.
                       'disk' [*] - a uniform disk source pointing along srcdir; the radius is
                                set by srcparam1(1) (in grid unit); if srcparam1(2) is set to a non-zero
                                value, this source defines a ring (annulus) shaped source, with
                                srcparam1(2) denoting the inner circle's radius, here srcparam1(1)>=srcparam1(2)
                                in 2D (one dimension of cfg.vol is 1), the disk becomes a line segment
                                of half-length srcparam1(1) centered at srcpos, perpendicular to srcdir
                                within the simulation plane (srcparam1(2) is not used)
                       'ring' [*] - a uniform ring/ring-sector source pointing along srcdir; the outer radius
                                of the ring is srcparam1(1) (in grid unit) and the inner radius is srcparam1(2);
                                srcparam1(3) and srcparam1(4) can optionally set the lower and upper angular
                                bound (in positive rad) of a ring sector if at least one of those is positive.
                       'fourierx' [*] - a general Fourier source, the parameters are
                                srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phi0,M]
                                normalized vectors satisfy: srcdir cross v1=v2
                                the phase shift is phi0*2*pi
                       'fourierx2d' [*] - a general 2D Fourier basis, parameters
                                srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phix,phiy]
                                the phase shift is phi{x,y}*2*pi
                       'zgaussian' - an angular gaussian beam, srcparam1(1) specifies the variance in the zenith angle
                       'line' - a line source, emitting from the line segment between
                                cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), radiating
                                uniformly in the perpendicular direction;
                                when cfg.srcparam2.x is non-zero, it
                                creates a wedge-source (or fan-beam), where
                                the launch directions are still
                                perpendicular to the source line segment,
                                but rotated by the wedge half-angle
                                cfg.srcparam2.x (in radian) from the plane
                                spanned by cfg.srcdir and the source line
                       'slit' [*] - a collimated slit beam emitting from the line segment between
                                cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), with the initial
                                dir specified by cfg.srcdir; when user defines positive values for srcparam2.x or .y,
                                the slit source is broadened in a Guassian profile controlled by
                                srcparam2.x: width of Gaussian broadening in the direction perpendicular to both slit and srcdir
                                srcparam2.y: width of Gaussian broadening in the direction of the slit line: cfg.srcparam(1:3)
                       'pencilarray' - a rectangular array of pencil beams. The srcparam1 and srcparam2
                                are defined similarly to 'fourier', except that srcparam1(4) and srcparam2(4)
                                are both integers, denoting the element counts in the x/y dimensions, respectively.
                                For exp., srcparam1=[10 0 0 4] and srcparam2[0 20 0 5] represent a 4x5 pencil beam array
                                spanning 10 grids in the x-axis and 20 grids in the y-axis (5-voxel spacing)
                       source types marked with [*] can be focused using the
                       focal length parameter (4th element of cfg.srcdir)
       cfg.{srcparam1,srcparam2}: 1x4 vectors, see cfg.srctype for details
       cfg.srcpattern: see cfg.srctype for details
       cfg.srcnum:     the number of source patterns that are
                       simultaneously simulated; only works for 'pattern'
                       source, see cfg.srctype='pattern' for details
                       Example <demo_photon_sharing.m>
       cfg.srcid:      when multiple sources are defined, if srcid is -1, each source is separately
                       simulated; if set to 0, all source solution are summed; if set to a positive
                       number starting from 1, only the specified source is simulated; default to 0
       cfg.omega: source modulation angular frequency (rad/s), i.e. 2*pi*f; used for
                       RF replay (cfg.seed from file, outputtype 'rf'/'rfmus') and
                       RF forward (no cfg.seed, omega>0 triggers RF forward automatically)
       cfg.srciquv: 1x4 vector [I,Q,U,V], Stokes vector of the incident light
                    I: total light intensity (I >= 0)
                    Q: balance between horizontal and vertical linearly
                    polarized light (-1 <= Q <= 1)
                    U: balance between +45° and -45° linearly polarized
                    light (-1 <= Q <= 1)
                    V: balance between right and left circularly polarized
                    light (-1 <= Q <= 1)
       cfg.lambda: source light wavelength (nm) for polarized MC
       cfg.issrcfrom0: 1-first voxel is [0 0 0], [0]- first voxel is [1 1 1]
       cfg.replaydet:  only works when cfg.outputtype is 'jacobian', 'wl', 'nscat', 'wp' or 'rf' and cfg.seed is an array
                       -1 replay all detectors and save in separate volumes (output has 5 dimensions)
                        0 replay all detectors and sum all Jacobians into one volume
                        a positive number: the index of the detector to replay and obtain Jacobians
       cfg.voidtime:   for wide-field sources, [1]-start timer at launch, or 0-when entering
                       the first non-zero voxel
 
  == Output control ==
       cfg.savedetflag: ['dp'] - a string (case insensitive) controlling the output detected photon data fields
                           1 d  output detector ID (1)
                           2 s  output partial scat. even counts (#media)
                           4 p  output partial path-lengths (#media)
                           8 m  output momentum transfer (#media)
                          16 x  output exit position (3)
                          32 v  output exit direction (3)
                          64 w  output initial weight (1)
                         128 i  output stokes vector (4)
                       combine multiple items by using a string, or add selected numbers together
                       by default, mcx only saves detector ID (d) and partial-path data (p)
       cfg.issaveexit: [0]-save the position (x,y,z) and (vx,vy,vz) for a detected photon
                       same as adding 'xv' to cfg.savedetflag. Example: <demo_lambertian_exit_angle.m>
       cfg.ismomentum: 1 to save photon momentum transfer,[0] not to save.
                       save as adding 'M' to cfg.savedetflag string
       cfg.issaveref:  [0]-save diffuse reflectance/transmittance in the non-zero voxels
                       next to a boundary voxel. The reflectance data are stored as
                       negative values; must pad zeros next to boundaries
                       Example: see the demo script at the bottom
       cfg.issave2pt:  [1]-save volumetric output in the first output fluence.data; user can disable this output
                       by explicitly setting cfg.issave2pt=0, this way, even the first output fluence presents
                       in mcxlab call, volume data will not be saved, this can speed up simulation when only
                       detphoton is needed
       cfg.issavedet:  if the 2nd output is requested, this will be set to 1; in such case, user can force
                       setting it to 3 to enable early termination of simulation if the detected photon
                       buffer (length controlled by cfg.maxdetphoton) is filled; if the 2nd output is not
                       present, this will be set to 0 regardless user input.
       cfg.outputtype: 'flux' - fluence-rate, (default value)
                       'fluence' - fluence integrated over each time gate,
                       'energy' - energy deposit per voxel
                       'jacobian' or 'wl' - mua Jacobian (replay mode),
                       'nscat' or 'wp' - weighted scattering counts for computing Jacobian for mus (replay mode)
                       'wm' - weighted momentum transfer for a source/detector pair (replay mode)
                       'rf' - dual-mode depending on cfg.seed:
                            replay mode (cfg.seed from .mch file, cfg.omega set):
                              frequency-domain mua Jacobian; output has dim6=2
                              ([real; imag] of complex Jacobian)
                            forward mode (no cfg.seed file, cfg.omega set):
                              complex frequency-domain fluence rate; output is
                              complex-valued (same size as 'flux'); multiply by
                              tstep to convert to complex fluence
                       'length' total pathlengths accumulated per voxel,
                       'rfmus' frequency domain/RF mus Jacobian (replay mode only),
                       'wltof' weighted average of time-of-flight x total path length in each voxel,
                       'wptof' weighted average of time-of-flight x total scattering count in each voxel
 
                       == Adjoint (forward-mode) Jacobians - requires cfg.detpos and cfg.detdir ==
                       'adjoint' - mua Jacobian (forward mode, no replay):
                              J_mua[r] = -phi_src[r]*phi_det[r]*dV [mm^-1]
                              output: real array [Nx,Ny,Nz,Nch] (CW) or
                              complex [Nx,Ny,Nz,Nch] (RF, when cfg.omega>0), where
                              Nch=Ns*Nd (all source-detector pairs, detector index
                              varying the fastest), or the number of rows of cfg.sdmap
                       'adjoint_dcoeff' - diffusion coefficient D=1/(3*mus') Jacobian:
                              J_D[r] = grad(phi_src)[r] . grad(phi_det)[r] * dV [mm^-3]
                              gradients computed via 2nd-order finite differences
                              (central at interior, one-sided at edges) with spacing
                              cfg.unitinmm; normalization: kernel*unitinmm
                       'adjoint_mus' - scattering coeff mus Jacobian (D=1/(3*(1-g)*mus)):
                              J_mus[r] = J_D[r] * 3*D^2*(1-g) = J_D[r]/(3*(1-g)*mus^2)
                              uses each voxel's own mus and g from cfg.prop
                       'adjoint_musp' - reduced scattering coeff mus'=(1-g)*mus Jacobian:
                              J_musp[r] = J_D[r] * 3*D^2 = J_D[r]/(3*(1-g)^2*mus^2)
                              uses each voxel's own mus and g from cfg.prop
                       cfg.sdmap: an Nch x 2 (or wider) array of 1-based [srcid detid] rows
                              selecting the adjoint Jacobian channels, in that order
 
                       for types jacobian/wl/wp, example: <demo_mcxlab_replay.m>
                           and  <demo_replay_timedomain.m>
                       for types rf (replay), example: <demo_replay_frequencydomain.m>
                       for types rf (forward), example: <compare_redbird_mcx_rf.m>
                       for types rfmus/wltof/wptof, example: <demo_replay_all_jacobian.m>
                       for adjoint types, example: <demo_mcx_adjoint_jacobian.m>
       cfg.session:    a string for output file names (only used when no return variables)
       cfg.lang:       specify the language code for printing, supported languages include
                       zh_CN, zh_TW, ja_JP, fr_CA, es_MX, de_DE, ko_KR, hi_IN, pt_BR
 
  == Debug ==
       cfg.debuglevel:  debug flag string (case insensitive), one or a combination of ['R','M','P','T'], no space
                     'R':  debug RNG, output fluence.data is filled with 0-1 random numbers
                     'M':  return photon trajectory data as the 5th output
                     'P':  show progress bar
                     'T':  save photon trajectory data only, as the 1st output, disable flux/detp/seeds outputs
       cfg.flog: [2]  log printing control; if set to a string, it defines a file path
                      at which location the log will be printed in append mode; on Linux and Mac OS,
                      one can use special paths such as /dev/null; if set to an integer,
                      2 (default): stderr
                      1: stdout
                      0: stdout but suppress printing MCX banner
       cfg.istrajstokes [0]: if set to 1, traj.iquv output contains the Stokes IQUV vector along trajectories
       cfg.maxjumpdebug: [10000000|int] when trajectory is requested in the output,
                      use this parameter to set the maximum position stored. By default,
                      only the first 1e6 positions are stored.
 
       fields with * are required; options in [] are the default values
 
  Output:
       fluence: a struct array, with a length equals to that of cfg.
             For each element of fluence,
             fluence(i).data is a 4D array with
                  dimensions specified by [size(vol) total-time-gates].
                  The content of the array is the normalized fluence at
                  each voxel of each time-gate.
 
                  when cfg.debuglevel contains 'T', fluence(i).data stores trajectory
                  output, see below
             fluence(i).dref is a 4D array with the same dimension as fluence(i).data
                  if cfg.issaveref is set to 1, containing only non-zero values in the
                  layer of voxels immediately next to the non-zero voxels in cfg.vol,
                  storing the normalized total diffuse reflectance (summation of the weights
                  of all escaped photon to the background regardless of their direction);
                  it is an empty array [] when if cfg.issaveref is 0.
             fluence(i).stat is a structure storing additional information, including
                  runtime: total simulation run-time in millisecond
                  nphoton: total simulated photon number
                  energytot: total initial weight/energy of all launched photons
                  energyabs: total absorbed weight/energy of all photons
                  normalizer: normalization factor
                  unitinmm: same as cfg.unitinmm, voxel edge-length in mm
 
       detphoton: (optional) a struct array, with a length equals to that of cfg.
             Starting from v2018, the detphoton contains the below subfields:
               detphoton.detid: the ID(>0) of the detector that captures the photon
               detphoton.srcid: the ID(>0) of the source in a multi-source simulation
               detphoton.nscat: cumulative scattering event counts in each medium
               detphoton.ppath: cumulative path lengths in each medium (partial pathlength)
                    one need to multiply cfg.unitinmm with ppath to convert it to mm.
               detphoton.mom: cumulative cos_theta for momentum transfer in each medium
               detphoton.p or .v: exit position and direction, when cfg.issaveexit=1
               detphoton.w0: photon initial weight at launch time
               detphoton.s: exit Stokes parameters for polarized photon
               detphoton.prop: optical properties, a copy of cfg.prop
               detphoton.data: a concatenated and transposed array in the order of
                     [detid nscat ppath mom p v w0]'
               "data" is the is the only subfield in all MCXLAB before 2018
       vol: (optional) a struct array, each element is a preprocessed volume
             corresponding to each instance of cfg. Each volume is a 3D int32 array.
       seeds: (optional), if give, mcxlab returns the seeds, in the form of
             a byte array (uint8) for each detected photon. The column number
             of seed equals that of detphoton.
       trajectory: (optional), if given, mcxlab returns the trajectory data for
             each simulated photon. The output has 6 rows, the meanings are
                id:  1:    index of the photon packet
                pos: 2-4:  x/y/z/ of each trajectory position
                     5:    current photon packet weight
              srcid: 6:    source ID (>0) of the photon
              iquv:  7-10: Stokes IQUV vector (>0) of the photon
             By default, mcxlab only records the first 1e7 positions along all
             simulated photons; change cfg.maxjumpdebug to define a different limit.
 
 
  Example:
       % first query if you have supported GPU(s)
       info=mcxlab('gpuinfo')
 
       % define the simulation using a struct
       cfg.nphoton=1e7;
       cfg.vol=uint8(ones(60,60,60));
       cfg.vol(20:40,20:40,10:30)=2;    % add an inclusion
       cfg.prop=[0 0 1 1;0.005 1 0 1.37; 0.2 10 0.9 1.37]; % [mua,mus,g,n]
       cfg.issrcfrom0=1;
       cfg.srcpos=[30 30 1];
       cfg.srcdir=[0 0 1];
       cfg.detpos=[30 20 1 1;30 40 1 1;20 30 1 1;40 30 1 1];
       cfg.vol(:,:,1)=0;   % pad a layer of 0s to get diffuse reflectance
       cfg.issaveref=1;
       cfg.gpuid=1;
       cfg.autopilot=1;
       cfg.tstart=0;
       cfg.tend=5e-9;
       cfg.tstep=5e-10;
       % calculate the fluence distribution with the given config
       [fluence,detpt,vol,seeds,traj]=mcxlab(cfg);
 
       % integrate time-axis (4th dimension) to get CW solutions
       cwfluence=sum(fluence.data,4);  % fluence rate
       cwdref=sum(fluence.dref,4);     % diffuse reflectance
       % plot configuration and results
       subplot(231);
       mcxpreview(cfg);title('domain preview');
       subplot(232);
       imagesc(squeeze(log(cwfluence(:,30,:))));title('fluence at y=30');
       subplot(233);
       hist(detpt.ppath(:,1),50); title('partial path tissue#1');
       subplot(234);
       plot(squeeze(fluence.data(30,30,30,:)),'-o');title('TPSF at [30,30,30]');
       subplot(235);
       newtraj=mcxplotphotons(traj);title('photon trajectories')
       subplot(236);
       imagesc(squeeze(log(cwdref(:,:,1))));title('diffuse refle. at z=1');
 
  This function is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
 
  License: GNU General Public License version 3, please read LICENSE.txt for details
</pre>

== # Examples ==

We provided several examples to demonstrate the basic usage of MCXLAB,
as well as to perform validations of MCX algorithm using both simple 
homogeneous and heterogeneous domains. These examples are explained below:

==== demo_mcxlab_basic.m ====

In this example, we show the most basic usage of MCXLAB. This include
how to define the input configuration structure, launch MCX simulations
and interpret and plotting the resulting data.

==== demo_validation_homogeneous.m ====

In this example, we validate MCXLAB with a homogeneous medium in a 
cubic domain. This is exactly the example shown in Fig.5 of [Fang2009].

You can also use the alternative optical properties that has a high g
value to observe the similarity between the two scattering/g configurations.

==== demo_validation_heterogeneous.m ====

In this example, we validate the MCXLAB solver with a heterogeneous
domain and the analytical solution of the diffusion model. We also 
demonstrate how to use sub-pixel resolution to refine the representation
of heterogeneities. The domain is consisted of a 6x6x6 cm box with a 
2cm diameter sphere embedded at the center. 

This test is identical to that used for Fig. 3 in [Fang2010].

==== demo_fullhead_atlas.m ====
In this example, we demonstrate light transport simulation in a full-head 
atlas template(USC 19.5 year group[Sanchez2012]). 
This demo is identical to the MCX simulation used for Fig.9(a) in
[TranYan2019].

==== demo_mcxyz_skinvessel.m ====
In this example, we compare between MCX and mcxyz written by Dr. Steve Jacques.
The same benchmark can be found at https://omlc.org/software/mc/mcxyz/index.html

==== demo_digimouse_sfdi.m ====
This simulates a widefield SFDI source using the Digimouse atlas. There are
21 tissue types in the atlas.

==== demo_4layer_head.m ====

In this example, we simulate a 4-layer brain model using MCXLAB.
We will investigate the differences between the solutions with and 
without boundary reflections (both external and internal) and show
you how to display and analyze the resulting data.

==== demo_mcxlab_srctype.m ====
This demo script shows how to use 9 different types sources in your
simulations. These 9 source types include pencil beam, isotropic source,
Gaussian beam, uniform planar source, uniform disk source, Fourier 
pattern illumination (spatial frequency domain sources), arcsine 
distribution beam, uniform cone beam, and an arbitrary light pattern 
(defined by a 2D image).

==== demo_mcxlab_2d.m ====
In this example, we show how to use MCX to run a 2D simulation.
You must define a 3D array with one singleton dimension (with length 1).
unfortunately if you define z as singleton, matlab will make the array 2D
instead of 3D, so we have to permute it to make the 1st dimension
singleton.

==== demo_photon_sharing.m ====
This script demonstrates the "photon sharing" feature (Yao&Yan et al, 2019, 
Photonics West) to simultaneously create forward solutions of multiple
patterned source.

==== demo_replay_timedomain.m ====
In this example, we show how to use replay to obtain time-resolved
Jacobians - setting cfg.replaydet to -1 to replay all detectors.

==== demo_replay_vs_pmc_timedomain.m ====
In this example, we compare perturbation MC and replay in predicting
time-resolved measurement change with respect to mua change in a layer.

==== demo_sphere_cube_subpixel.m ====

In this example, we demonstrate how to use sub-pixel resolution 
to represent the problem domain. The domain is consisted of a 
6x6x6 cm box with a 2cm diameter sphere embedded at the center.


== # How to compile MCXLAB ==

To compile MCXLAB for MATLAB, you need to cd mcx/src directory, and type 

 make mex

from a shell window. You need to make sure your MATLAB is installed and 
the command <tt>mex</tt> is included in your PATH environment variable. Similarly, 
to compile MCXLAB for Octave, you type

 make oct

The command <tt>mkoctfile</tt> must be accessible from your command line;
on Ubuntu/Debian, it is provided by the package <tt>liboctave-dev</tt>.

The default targets compile for the oldest supported GPU architecture. To
compile for a specific GPU generation, use the corresponding target, such as
<tt>make turingmex</tt>, <tt>make ampereoct</tt> etc. MCXLAB can also be built with
CMake, see <tt>src/CMakeLists.txt</tt>.

== # Screenshots ==

Screenshot for using MCXLAB in MATLAB:
  https://mcx.space/wiki/upload/matlab_mcxlab.png

Screenshot for using MCXLAB in GNU Octave:
  https://mcx.space/wiki/upload/octave_mcxlab.png


== # Reference ==

 [Yan2020] Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel 
  based Monte Carlo algorithm for accurate and efficient photon transport 
  modeling in complex bio-tissues," Biomed. Opt. Express, 11(11) pp. 6262-6270.
  https://opg.optica.org/boe/abstract.cfm?uri=boe-11-11-6262
 
 [Yu2018] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, 
 "Scalable and massively parallel Monte Carlo photon transport simulations 
 for heterogeneous computing platforms," J. Biomed. Opt. 23(1), 010504 (2018).

 [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation 
  of photon migration in 3D turbid media accelerated by graphics processing 
  units," Opt. Express 17, 20178-20190 (2009)

 [TranYan2020] A.P.Tran, S.Yan and Q.Fang, "Improving model-based fNIRS
 analysis using mesh-based anatomical and light-transport models", 
 Neurophotonics, 7(1), 015008

