function varargout = mcxlab(varargin)
%
% ====================================================================
%      MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave
% --------------------------------------------------------------------
% Copyright (c) 2011-2026 Qianqian Fang <q.fang at neu.edu>
%                      URL: https://mcx.space
% ====================================================================
%
% Format:
%    fluence = mcxlab(cfg);
%    [fluence, detphoton, vol, seeds, traj] = mcxlab(cfg);
%    [fluence, detphoton, vol, seeds, traj] = mcxlab(cfg, option);
%    info = mcxlab('gpuinfo');
%    ver  = mcxlab('version');
%
%    Request only the outputs you need: each extra output costs memory and
%    time. The detphoton, seeds and traj outputs are only produced when they
%    are requested, and [~, detphoton] = mcxlab(cfg) skips the fluence volume
%    in Octave (set cfg.issave2pt=0 to do the same in MATLAB).
%
% Input:
%    cfg: a struct, or a struct array; each element is one simulation.
%         cfg='gpuinfo' returns a struct array describing the supported GPUs;
%         cfg='version' returns the MCX version string.
%    option: (optional) a string
%         'preview': plot the domain, sources and detectors with mcxpreview(cfg)
%                    instead of running the simulation
%         'opencl':  run with mcxcl.mex* (MCX-CL, any OpenCL device) instead of mcx.mex*
%         'cuda':    run with mcx.mex* (CUDA, NVIDIA GPUs), the default
%
%    A variable USE_MCXCL in the base workspace changes the default for all
%    following mcxlab/mcxlabcl calls: USE_MCXCL=1 makes both use mcxcl.mex*,
%    USE_MCXCL=0 makes both use mcx.mex*; option='opencl'/'cuda' still
%    overrides it. USE_MCXCL set to an integer N>1 (or a device string such
%    as '0101') also selects mcxcl.mex* and replaces cfg.gpuid by N; set to
%    -N, it selects mcx.mex* and replaces cfg.gpuid by N.
%
%    Fields marked with * are required; values in [] are the defaults.
%    Lengths are in grid (voxel) units unless stated otherwise; cfg.unitinmm
%    sets the voxel edge length in mm.
%
% == Domain and optical properties ==
%     *cfg.nphoton:    total number of photons to simulate (up to 2^63-1)
%     *cfg.vol:        a 3D array of tissue labels (uint8, uint16, uint32, single
%                      or double), with label 0 being the background (outside of
%                      the domain). A singleton x or y dimension makes a 2D
%                      simulation; srcpos/srcdir must then lie in that plane.
%                      Example: <demo_mcxlab_2d.m>
%
%                      a 4D array, whose first dimension holds per-voxel data,
%                      describes continuously varying media:
%                        1xNxxNyxNz single:  "mua_float" (101), per-voxel mua;
%                                            mus/g/n from cfg.prop(2,:)
%                        2xNxxNyxNz single:  "muamus_half" (102), per-voxel mua/mus,
%                                            stored as half floats; g/n from prop(2,:)
%                        2xNxxNyxNz uint16:  "muamus_short" (104), mua/mus as 0-65535
%                                            levels between prop(2,:) and prop(3,:)
%                        3xNxxNyxNz single:  "labelplus" (99): [value, property
%                                            index 0-3, label]; the property of
%                                            the label is replaced by the value
%                        3xNxxNyxNz int16/uint16: "mixlabel" (98): [label1, label2,
%                                            label1 percentage 0-65535]
%                        4xNxxNyxNz uint8:   "asgn_byte" (103), mua/mus/g/n as 0-255
%                                            levels between prop(2,:) and prop(3,:)
%                        4xNxxNyxNz single:  "asgn_float" (96), per-voxel mua/mus/g/n
%                                            (stored as half floats)
%                        8xNxxNyxNz uint8:   "svmc" (97), split-voxel MC; create it
%                                            with mcxsvmc.m
%                      Examples: <demo_continuous_mua_mus.m>, <demo_svmc_*.m>
%                      With per-voxel properties (all but svmc), detected photons
%                      carry no per-medium data (partial path, nscat, momentum).
%     *cfg.prop:       an Nx4 array, one row per medium, [mua(1/mm), mus(1/mm), g, n];
%                      the first row is label 0 (background, normally [0 0 1 1]),
%                      the second row is label 1, and so on. A photon is terminated
%                      when it moves from a non-zero voxel into a 0 voxel.
%      cfg.polprop:    an Nx5 array for polarized MC, one row per medium starting
%                      from label 1: [mua(1/mm), sphere radius(micron), sphere number
%                      density(1/micron^3), sphere refractive index, ambient
%                      refractive index]; the background stays in cfg.prop(1,:).
%                      If prop(i+1,2) is non-zero, the density is adjusted to give
%                      mus=prop(i+1,2) when prop(i+1,3)==1, or mus'=prop(i+1,2)*
%                      (1-prop(i+1,3)) when prop(i+1,3)<1. Requires cfg.lambda.
%                      Example: <demo_polarized_photon.m>
%      cfg.unitinmm:   voxel edge length in mm [1.0]
%                      Example: <demo_sphere_cube_subpixel.m>
%      cfg.shapes:     a JSON string of shapes (spheres, boxes, layers, ...) drawn
%                      into cfg.vol, see https://mcx.space/wiki/?Shapes
%                      Example: <demo_mcxyz_skinvessel.m>
%      cfg.issrcfrom0: 1: the first voxel spans [0,1] in each axis; [0]: [1,2]
%
% == Time gates ==
%     *cfg.tend:       end of the simulated time window (s)
%      cfg.tstart:     start of the time window (s) [0]
%      cfg.tstep:      width of each time gate (s) [cfg.tend-cfg.tstart, one gate]
%      cfg.maxgate:    the number of time gates simulated per run [0: all that fit
%                      in the GPU memory]
%
% == Source ==
%     *cfg.srcpos:     [x y z] position of the source; an optional 4th value sets
%                      the initial photon weight w0 [1] (for pattern sources, a
%                      multiplier); w0 may be negative, the output scales with it
%     *cfg.srcdir:     [vx vy vz] unit launch direction; an optional 4th value is
%                      the focal length (for the sources marked [*] below):
%                        0 [default]: collimated
%                        >0: converging to a focus at that distance along srcdir
%                        <0: diverging from a virtual point behind the source
%                      or, for every source type:
%                        nan: isotropic launch, regardless of srcdir
%                        -inf: Lambertian (cosine) launch around srcdir, over a
%                              hemisphere in 3D, or a semicircle within the plane
%                              of a 2D simulation
%                      several sources: give srcpos/srcdir (and srcparam1/2) one
%                      row per source, all with the same number of rows
%      cfg.srctype:    source type ['pencil']; cfg.srcparam1/srcparam2 hold the
%                      parameters. Example: <demo_mcxlab_srctype.m>
%                      'pencil'      a pencil beam
%                      'isotropic'   isotropic point source
%                      'arcsine'     like isotropic, with a uniform zenith angle
%                      'cone'        uniform cone; srcparam1(1): half-angle (rad)
%                      'zgaussian'   Gaussian zenith angle; srcparam1(1): variance
%                      'gaussian'    [*] collimated Gaussian beam; srcparam1(1): waist radius
%                      'hyperboloid' [*] one-sheet hyperboloid Gaussian beam; srcparam1:
%                                    [waist radius, distance from launch plane to
%                                    focus, Rayleigh range]
%                      'disk'        [*] uniform disk normal to srcdir; srcparam1(1):
%                                    radius, srcparam1(2): inner radius of an annulus
%                                    [0]; in 2D, a line segment of half-length
%                                    srcparam1(1) normal to srcdir
%                      'ring'        [*] ring or ring sector; srcparam1: [outer radius,
%                                    inner radius, start angle, end angle (rad)], a
%                                    sector when either angle is positive
%                      'planar'      [*] uniform quadrilateral with corners srcpos,
%                                    srcpos+srcparam1(1:3), srcpos+srcparam2(1:3)
%                      'pattern'     [*] the quadrilateral of 'planar' with intensity
%                                    cfg.srcpattern (0-1) of srcparam1(4) x srcparam2(4)
%                                    pixels; cfg.srcnum patterns at once are given as
%                                    a [srcnum, srcparam1(4), srcparam2(4)] array.
%                                    Example: <demo_photon_sharing.m>
%                      'pattern3d'   [*] a 3D intensity volume cfg.srcpattern of size
%                                    srcparam1(1:3), starting at srcpos
%                      'fourier'     [*] spatial-frequency source over the 'planar'
%                                    quadrilateral, S=0.5*[1+M*cos(2*pi*(fx*x+fy*y)+phi0)]:
%                                    integer parts of srcparam1(4)/srcparam2(4) are
%                                    fx/fy, fraction of srcparam1(4) is phi0/(2*pi),
%                                    1 minus the fraction of srcparam2(4) is M
%                      'fourierx'    [*] general Fourier source; srcparam1: [v1x v1y v1z
%                                    |v2|], srcparam2: [kx ky phi0 M], with
%                                    v2=srcdir x v1; the phase is phi0*2*pi
%                      'fourierx2d'  [*] general 2D Fourier basis; srcparam1 as for
%                                    'fourierx', srcparam2: [kx ky phix phiy]
%                      'pencilarray' an array of pencil beams over the 'planar'
%                                    quadrilateral; srcparam1(4)/srcparam2(4) are the
%                                    beam counts, e.g. srcparam1=[10 0 0 4],
%                                    srcparam2=[0 20 0 5] gives a 4x5 array
%                      'line'        line source from srcpos to srcpos+srcparam1(1:3),
%                                    emitting normal to the line; srcparam2(1)>0
%                                    turns it into a wedge (fan) beam tilted by that
%                                    half-angle (rad) from the srcdir-line plane
%                      'slit'        [*] collimated slit beam along srcdir from the same
%                                    segment; srcparam2(1)/(2): widths of a Gaussian
%                                    broadening across/along the slit
%      cfg.srcparam1, cfg.srcparam2: 1x4 vectors, see cfg.srctype
%      cfg.srcpattern: the pattern of 'pattern'/'pattern3d' sources
%      cfg.srcnum:     number of patterns simulated together by a 'pattern'
%                      source [1]; the output gains a last dimension of srcnum
%      cfg.srcid:      with several sources: [0] all together in one output,
%                      -1 each in its own output, N>0 only the N-th source
%      cfg.angleinvcdf: a user-defined launch-angle distribution: a vector,
%                      increasing within [0,1], of the inverse CDF of theta/pi,
%                      theta being the angle to srcdir. It is sampled with linear
%                      interpolation when srcdir(4)=0 (continuous distributions),
%                      without it when srcdir(4)=1 (discrete angles).
%                      Example: <demo_mcxlab_launchangle.m>
%      cfg.voidtime:   for a source outside of the domain, [1]: start the clock at
%                      launch, 0: when the photon enters the domain
%      cfg.maxvoidstep: the farthest (voxels) a photon launched outside travels
%                      to reach the domain [1000]
%      cfg.isspecular: 1: apply the specular reflection at entry for a source
%                      outside of the domain; [0]: do not
%      cfg.srciquv:    [I Q U V] Stokes vector of the source for polarized MC:
%                      I>=0 is the intensity; Q, U and V, each in [-1,1],
%                      balance horizontal/vertical, +45/-45 degree linear and
%                      right/left circular polarization
%      cfg.lambda:     wavelength (nm) for polarized MC
%      cfg.omega:      modulation angular frequency 2*pi*f (rad/s) for
%                      frequency-domain (RF) outputs [0]; see cfg.outputtype
%
% == Detectors ==
%      cfg.detpos:     an Nx4 array, one row per disk detector: [x y z radius]
%      cfg.detdir:     an Nx3 or Nx4 array, one row per detector: [vx vy vz] or
%                      [vx vy vz focal length], the inward direction of each
%                      detector, required by the adjoint outputs (the detectors
%                      are launched as sources); the focal length is used as in
%                      cfg.srcdir, and 0 [default, also when omitted] launches
%                      along [vx vy vz]
%      cfg.sdmap:      an Nch x 2 (or wider) array selecting the channels of the
%                      adjoint Jacobians: each row is [srcid detid], both 1-based
%                      (srcid counts the rows of cfg.srcpos, detid the rows of
%                      cfg.detpos); extra columns are ignored. The Jacobians then
%                      have Nch channels, in the row order of cfg.sdmap, instead
%                      of all Ns*Nd pairs. Default: all pairs, with the detector
%                      index varying the fastest, i.e. [srcid detid] =
%                      [1 1; 1 2; ... 1 Nd; 2 1; ...]. Only the sources and
%                      detectors in cfg.sdmap are simulated, and the forward
%                      solutions in the 1st output are only theirs: sources
%                      first, then detectors, each in increasing order
%      cfg.bc:         per-face boundary conditions, a string of up to 12 letters;
%                      letters 1-6 are the -x,-y,-z,+x,+y,+z bounding box faces:
%                        '_': use cfg.isreflect   'r': Fresnel reflection
%                        'a': total absorption    'm': mirror (total reflection)
%                        'c': cyclic, re-enter from the opposite face
%                      letters 7-12, '0' or '1', make the same faces detectors:
%                      e.g. '______001000' records photons leaving at z=0
%                      Example: <demo_bc_det.m>
%      cfg.maxdetphoton: the most detected photons kept [1000000]
%      cfg.savedetflag: which data to save per detected photon, a string of the
%                      letters below (case insensitive) ['dp']:
%                        'd' detector ID (1)        's' scattering counts (#media)
%                        'p' partial paths (#media) 'm' momentum transfer (#media)
%                        'x' exit position (3)      'v' exit direction (3)
%                        'w' initial weight (1)     'i' Stokes IQUV (4, polarized MC)
%      cfg.issaveexit: 1: also save exit positions and directions, same as
%                      adding 'xv' to cfg.savedetflag [0].
%                      Example: <demo_lambertian_exit_angle.m>
%      cfg.ismomentum: 1: also save the momentum transfer, same as adding 'm' [0]
%      cfg.issavedet:  set to 1 automatically when detphoton is requested; set it
%                      to 3 to stop the simulation once cfg.maxdetphoton photons
%                      are detected
%
% == Monte Carlo settings ==
%      cfg.seed:       RNG seed, an integer [1648335518]; <=0 seeds from the clock.
%                      A run is reproducible only with cfg.autopilot=0.
%                      a uint8 array (the seeds output of an earlier run) replays
%                      those detected photons, see "Replay" below
%      cfg.respin:     run the simulation this many times and add up the results
%                      [1]; a negative value -N splits nphoton into N runs instead
%      cfg.isreflect:  [1]: Fresnel reflection at the domain boundary;
%                      0: photons leave (index-matched boundary)
%      cfg.isnormalized: [1]: normalize the output to a unit source; 0: raw output;
%                      2: in replay, skip the normalization by the detected weight,
%                      i.e. Born instead of Rytov Jacobians
%      cfg.minenergy:  weight below which a photon enters Russian roulette [0]
%      cfg.gscatter:   after this many scattering events, scatter isotropically
%                      (ignore g) for speed [1e9]
%      cfg.invcdf:     a user-defined scattering phase function: a vector,
%                      increasing within [-1,1], of the inverse CDF of
%                      u=cos(theta) of the scattering angle; it replaces
%                      Henyey-Greenstein. Example: <demo_mcxlab_phasefun.m>
%
% == Output ==
%      cfg.outputtype: what the 1st output holds ['flux']
%                      'flux'      fluence rate (1/mm^2/s)
%                      'fluence'   fluence (1/mm^2), integrated over each time gate
%                      'energy'    energy deposited per voxel
%                      'length'    total path length per voxel
%                      replay outputs (see "Replay"):
%                      'jacobian' or 'wl'  mua Jacobian
%                      'nscat' or 'wp'     weighted scattering counts (mus Jacobian)
%                      'wm'        weighted momentum transfer
%                      'rf'        with cfg.omega>0: [real, imag] parts of the
%                                  frequency-domain mua Jacobian, in a 6th
%                                  dimension of size 2
%                      'rfmus'     frequency-domain mus Jacobian
%                      'wltof'     time of flight x path length, weighted
%                      'wptof'     time of flight x scattering count, weighted
%                      forward outputs, without replay:
%                      'rf'        with cfg.omega>0: complex fluence rate
%                                  (multiply by tstep for the complex fluence)
%                      adjoint Jacobians (need cfg.detpos and cfg.detdir; every
%                      detector is also run as a source, and the Jacobian of
%                      each source-detector pair is returned in its own field of
%                      the 1st output, [Nx,Ny,Nz,Ns*Nd], or [Nx,Ny,Nz,Nch] with
%                      cfg.sdmap, complex if cfg.omega>0):
%                      'adjoint'          mua Jacobian, fluence.jmua
%                      'adjoint_dcoeff'   diffusion coefficient D=1/(3mus')
%                                         Jacobian, fluence.jd
%                      'adjoint_mus'      mus Jacobian, fluence.jmus
%                      'adjoint_musp'     mus' Jacobian, fluence.jmusp
%                      'adjoint_mua_d'    both fluence.jmua and fluence.jd
%                      'adjoint_mua_musp' both fluence.jmua and fluence.jmusp
%                      Examples: <demo_mcxlab_replay.m>, <demo_replay_timedomain.m>,
%                      <demo_replay_frequencydomain.m>, <demo_replay_all_jacobian.m>,
%                      <demo_mcx_adjoint_jacobian.m>
%      cfg.issave2pt:  [1]: compute the 1st output; 0: skip it, which is faster
%                      when only detphoton is needed
%      cfg.issaveref:  1: save the diffuse reflectance in fluence.dref [0]; it is
%                      recorded in the background voxels next to the domain, so
%                      pad cfg.vol with a layer of 0s where it is wanted
%      cfg.session:    a name for the output files, used when mcxlab returns nothing
%      cfg.flog:       where the messages go - the echo of the cfg fields and the
%                      simulation log (banner, timings, progress bar, absorbed fraction)
%                      [2]: 2 the MATLAB/Octave console (in Python, sys.stdout), 1 the
%                      stdout of the process (the terminal MATLAB/Octave was started from),
%                      0 the same without the banner, or a file name (appended);
%                      '/dev/null' or 'NUL' (either, on any OS) silences everything,
%                      the progress bar included;
%                      errors are always shown
%      cfg.lang:       language of the log: zh_CN, zh_TW, ja_JP, fr_CA, es_MX, de_DE,
%                      ko_KR, hi_IN or pt_BR
%
% == Replay ==
%      Replaying the detected photons of a baseline run, with cfg.seed set to its
%      seeds output, gives Jacobians (sensitivity profiles) with cfg.outputtype.
%      cfg.detphotons: the detphoton output of the baseline run (or its .data)
%      cfg.replaydet:  [0]: replay all detectors into one output; N>0: only
%                      detector N; -1: all detectors, each in its own output
%                      (an extra 5th dimension)
%
% == GPU ==
%      cfg.gpuid:      which GPU to use, from 1 ('mcx -L' or mcxlab('gpuinfo')
%                      lists them) [1]; a string of 0s and 1s selects several,
%                      e.g. '1101' uses the 1st, 2nd and 4th GPUs.
%                      Example: <mcx_gpu_benchmarks.m>
%      cfg.workload:   the relative share of photons for each selected GPU, e.g.
%                      [50 20 30]
%      cfg.autopilot:  [1]: choose the thread and block counts automatically and
%                      hand out photons to the threads dynamically, so results
%                      with a fixed seed agree only statistically; 0: use
%                      cfg.nthread/cfg.nblocksize, bit-wise reproducible
%      cfg.nthread:    total thread count with autopilot=0 [16384]
%      cfg.nblocksize: threads per block with autopilot=0 [64]
%      cfg.isgpuinfo:  1: print the GPU information before running [0]
%
% == Debugging ==
%      cfg.debuglevel: a string of debug flags (case insensitive):
%                        'P' show a progress bar
%                        'M' record photon trajectories (5th output)
%                        'T' record trajectories only, returned as the 1st output,
%                            without fluence, detphoton or seeds
%                        'R' fill the 1st output with random numbers to test the RNG
%      cfg.maxjumpdebug: the most trajectory positions recorded [10000000]
%      cfg.istrajstokes: 1: also record the Stokes vector along trajectories [0]
%
%      accepted but without effect (legacy): cfg.isref3, cfg.isrefint,
%      cfg.sradius, cfg.crop0, cfg.crop1, cfg.faststep, cfg.steps (an alias
%      of cfg.unitinmm)
%
% Output:
%      fluence: a struct array, one element per cfg element, with the subfields
%            data:  the volume selected by cfg.outputtype, [size(vol), #gates],
%                   normalized as set by cfg.isnormalized; extra dimensions
%                   index the patterns (cfg.srcnum>1), the sources (cfg.srcid=-1),
%                   the detectors (cfg.replaydet=-1) or [real, imag] parts (RF);
%                   for the adjoint outputs, it is the forward fluence of the
%                   Ns sources and Nd detectors, [Nx,Ny,Nz,#gates,Ns+Nd]; with
%                   debuglevel 'T' it holds the trajectories instead (see traj)
%            dref:  the diffuse reflectance, same size as data, when
%                   cfg.issaveref=1, or []
%            stat:  run statistics: runtime (ms), nphoton, energytot (total
%                   launched weight), energyabs (total absorbed weight),
%                   normalizer, unitinmm and workload
%            prop:  for polarized MC, the optical properties computed from
%                   cfg.polprop
%            jmua, jd, jmus, jmusp: the adjoint Jacobians, see cfg.outputtype
%      detphoton: a struct array of the detected photons, one row per photon:
%            detid: the detector index (from 1)
%            srcid: the source index (from 1), with several sources
%            nscat: scattering counts in each medium
%            ppath: partial path lengths in each medium, in voxels (multiply by
%                   cfg.unitinmm for mm)
%            mom:   momentum transfer in each medium
%            p, v:  exit position and direction
%            w0:    initial weight
%            s:     exit Stokes vector (polarized MC)
%            prop, unitinmm: copies of cfg.prop and cfg.unitinmm
%            data:  all of the above stored, transposed, in one array
%      vol:   the preprocessed label volume (with cfg.shapes drawn in), uint32
%      seeds: the RNG seed of each detected photon, one uint8 column per
%            photon of detphoton, for replay
%      traj:  photon trajectories, sorted by photon, with subfields id (photon
%            index), pos (x/y/z), srcid, iquv (with cfg.istrajstokes=1) and
%            data, all of them as rows [id x y z weight srcid ...] of one array
%
% Example:
%      % list the supported GPUs
%      info = mcxlab('gpuinfo')
%
%      % a 60x60x60 voxel domain with a cubic inclusion
%      cfg.nphoton = 1e7;
%      cfg.vol = uint8(ones(60, 60, 60));
%      cfg.vol(20:40, 20:40, 10:30) = 2;
%      cfg.vol(:, :, 1) = 0;   % a layer of 0s to record the reflectance in
%      cfg.prop = [0 0 1 1; 0.005 1 0 1.37; 0.2 10 0.9 1.37]; % [mua, mus, g, n]
%      cfg.issrcfrom0 = 1;
%      cfg.srcpos = [30 30 1];
%      cfg.srcdir = [0 0 1];
%      cfg.detpos = [30 20 1 1; 30 40 1 1; 20 30 1 1; 40 30 1 1];
%      cfg.issaveref = 1;
%      cfg.gpuid = 1;
%      cfg.tstart = 0;
%      cfg.tend = 5e-9;
%      cfg.tstep = 5e-10;
%      [fluence, detp, vol, seeds, traj] = mcxlab(cfg);
%
%      % sum the time gates (4th dimension) for the CW solutions
%      cwfluence = sum(fluence.data, 4);  % fluence rate
%      cwdref = sum(fluence.dref, 4);     % diffuse reflectance
%      subplot(231);
%      mcxpreview(cfg); title('domain preview');
%      subplot(232);
%      imagesc(squeeze(log(cwfluence(:, 30, :)))); title('fluence at y=30');
%      subplot(233);
%      hist(detp.ppath(:, 1), 50); title('partial path tissue#1');
%      subplot(234);
%      plot(squeeze(fluence.data(30, 30, 30, :)), '-o'); title('TPSF at [30,30,30]');
%      subplot(235);
%      newtraj = mcxplotphotons(traj); title('photon trajectories');
%      subplot(236);
%      imagesc(squeeze(log(cwdref(:, :, 1)))); title('diffuse refle. at z=1');
%
% This function is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

try
    defaultocl = evalin('base', 'USE_MCXCL');
catch
    defaultocl = 0;
end

useopencl = defaultocl;

if (nargin == 2 && ischar(varargin{2}))
    if (strcmp(varargin{2}, 'preview'))
        [varargout{1:nargout}] = mcxpreview(varargin{1});
        return
    elseif (strcmp(varargin{2}, 'opencl'))
        useopencl = 1;
    elseif (strcmp(varargin{2}, 'cuda'))
        useopencl = 0;
    end
end

if (ischar(varargin{1}) && ~any(strcmp(varargin{1}, {'gpuinfo', 'version'})))
    error('unsupported command ''%s'': cfg must be a struct, ''gpuinfo'' or ''version''', varargin{1});
end

if (isstruct(varargin{1}))
    for i = 1:length(varargin{1})
        if (ischar(useopencl) || useopencl > 1)
            varargin{1}(i).gpuid = useopencl;
        elseif (useopencl < 0)
            varargin{1}(i).gpuid = -useopencl;
        end
        % the mex reads the numeric fields as double arrays: convert any other numeric or
        % logical class (int32(1), true, single ...) and MATLAB string scalars here; vol keeps
        % its class, which selects the media format, a logical vol becomes uint8 labels, seed
        % keeps uint8 (replay) and detphotons is converted to single below
        fields = fieldnames(varargin{1}(i));
        for j = 1:length(fields)
            val = varargin{1}(i).(fields{j});
            if (isa(val, 'string'))
                val = char(val);
            elseif (strcmp(fields{j}, 'vol'))
                if (islogical(val))
                    val = uint8(val);
                end
            elseif (strcmp(fields{j}, 'seed'))
                if (~isa(val, 'uint8'))
                    val = double(val);
                end
            elseif (~strcmp(fields{j}, 'detphotons') && (isnumeric(val) || islogical(val)) && ~isa(val, 'double'))
                val = double(val);
            end
            if (any(strcmp(fields{j}, {'srcpos', 'srcdir'})) && size(val, 2) == 1 && any(numel(val) == [3 4]))
                val = val(:)';  % a column vector is one source
            end
            varargin{1}(i).(fields{j}) = val;
        end
        if (~isfield(varargin{1}(i), 'tstart'))
            varargin{1}(i).tstart = 0;
        end
        if (~isfield(varargin{1}(i), 'tend'))
            error('you must define cfg.tend for the maximum time-of-flight of a photon in seconds');
        end
        if (~isfield(varargin{1}(i), 'tstep'))
            varargin{1}(i).tstep = varargin{1}(i).tend;
        end
        if (~isfield(varargin{1}(i), 'srcpos'))
            error('you must define cfg.srcpos to defin the x/y/z position of the source in voxel unit');
        end
        if (isfield(varargin{1}(i), 'detphotons') && isstruct(varargin{1}(i).detphotons))
            if (isfield(varargin{1}(i).detphotons, 'data'))
                varargin{1}(i).detphotons = varargin{1}(i).detphotons.data;
            else
                fulldetdata = {'detid', 'nscat', 'ppath', 'mom', 'p', 'v', 'w0'};
                detfields = ismember(fulldetdata, fieldnames(varargin{1}(i).detphotons));
                detdata = [];
                for j = 1:length(detfields)
                    if (detfields(j))
                        val = typecast(varargin{1}(i).detphotons.(fulldetdata{j})(:), 'single');
                        detdata = [detdata reshape(val, size(varargin{1}(i).detphotons.(fulldetdata{j})))];
                    end
                end
                varargin{1}(i).detphotons = detdata';
                varargin{1}(i).savedetflag = 'dspmxvw';
                varargin{1}(i).savedetflag(detfields == 0) = [];
            end
        end
        if (isfield(varargin{1}(i), 'detphotons') && isnumeric(varargin{1}(i).detphotons))
            varargin{1}(i).detphotons = single(varargin{1}(i).detphotons);  % the mex copies it as single
        end
    end
end

% detect [~,...]=mcxlab() in Octave. unfortunately matlab does not have isargout()
if (nargout >= 1 && exist('isargout', 'builtin') && isargout(1) == 0)
    for i = 1:length(varargin{1})
        varargin{1}(i).issave2pt = 0;
    end
end

if (useopencl <= 0)
    [varargout{1:max(1, nargout)}] = mcx(varargin{1});
else
    [varargout{1:max(1, nargout)}] = mcxcl(varargin{1});
end

if (nargin == 0)
    return
end

cfg = varargin{1};

if (~ischar(cfg))
    for i = 1:length(varargout{1})
        if (isfield(cfg(i), 'srcnum') && cfg(i).srcnum > 1)
            dim = size(varargout{1}(i).data);
            varargout{1}(i).data = reshape(varargout{1}(i).data, [cfg(i).srcnum, dim(1) / cfg(i).srcnum dim(2:end)]);
            varargout{1}(i).data = permute(varargout{1}(i).data, [2:(length(dim) + 1) 1]);
            if (isfield(varargout{1}(i), 'dref') && ~isempty(varargout{1}(i).dref))
                varargout{1}(i).dref = reshape(varargout{1}(i).dref, [cfg(i).srcnum, dim(1) / cfg(i).srcnum dim(2:end)]);
                varargout{1}(i).dref = permute(varargout{1}(i).dref, [2:(length(dim) + 1) 1]);
            end
        end
    end
end

if (nargout >= 2)
    for i = 1:length(varargout{2})
        if ((~isfield(cfg(i), 'savedetflag')) || ((isfield(cfg(i), 'savedetflag')) && isempty(cfg(i).savedetflag)))
            cfg(i).savedetflag = 'DP';
        end
        if (isfield(cfg(i), 'issaveexit') && cfg(i).issaveexit)
            cfg(i).savedetflag = [cfg(i).savedetflag, 'XV'];
        end
        if (isfield(cfg(i), 'ismomentum') && cfg(i).ismomentum)
            cfg(i).savedetflag = [cfg(i).savedetflag, 'M'];
        end
        if (isfield(cfg(i), 'polprop') && ~isempty(cfg(i).polprop))
            cfg(i).savedetflag = [cfg(i).savedetflag, 'PVWI'];
        else
            cfg(i).savedetflag(regexp(cfg(i).savedetflag, '[Ii]')) = [];
        end
        if (ndims(cfg(i).vol) == 4 && size(cfg(i).vol, 1) ~= 8)
            cfg(i).savedetflag = '';
        end
        if ((~isfield(cfg(i), 'issaveexit') || cfg(i).issaveexit ~= 2))
            medianum = size(cfg(i).prop, 1) - 1;
            detp = varargout{2}(i).data;
            if (isempty(detp))
                continue
            end
            if (isfield(cfg(i), 'polprop') && ~isempty(cfg(i).polprop))
                medianum = size(cfg(i).polprop, 1);
            end
            flags = {cfg(i).savedetflag};
            if (isfield(cfg(i), 'issaveref'))
                flags{end + 1} = cfg(i).issaveref;
            else
                flags{end + 1} = 0;
            end
            if (isfield(cfg(i), 'srcnum'))
                flags{end + 1} = cfg(i).srcnum;
            end
            newdetp = mcxdetphoton(detp, medianum, flags{:});
            newdetp.prop = cfg(i).prop;
            if (isfield(cfg(i), 'polprop') && ~isempty(cfg(i).polprop)) && isfield(varargout{1}(i), 'prop')
                newdetp.prop(2:end, :) = varargout{1}(i).prop(:, 2:end)';
            end
            if (isfield(cfg(i), 'unitinmm'))
                newdetp.unitinmm = cfg(i).unitinmm;
            end
            newdetp.data = detp;      % enable this line for compatibility
            newdetpstruct(i) = newdetp;
        else
            newdetpstruct(i) = varargout{2}(i);
        end
    end
    if (exist('newdetpstruct', 'var'))
        varargout{2} = newdetpstruct;
    end
end

if (nargout >= 5 || (~isempty(cfg) && isstruct(cfg) && isfield(cfg, 'debuglevel') && ~isempty(regexp(cfg(1).debuglevel, '[tT]', 'once'))))
    outputid = 5;
    if ((isfield(cfg, 'debuglevel') && ~isempty(regexp(cfg(1).debuglevel, '[tT]', 'once'))))
        outputid = 1;
    end
    for i = 1:length(varargout{outputid})
        data = varargout{outputid}.data;
        if (isempty(data))
            continue
        end
        traj.pos = data(2:4, :).';
        traj.id = typecast(data(1, :), 'uint32').';
        [traj.id, idx] = sort(traj.id);
        traj.pos = traj.pos(idx, :);
        traj.srcid = int32(data(6, :)');
        if (size(data, 1) >= 10)
            traj.iquv = data(7:10, :)';
            traj.iquv = traj.iquv(idx, :);
        end
        traj.data = [single(traj.id)'; data(2:end, idx)];
        newtraj(i) = traj;
    end
    if (exist('newtraj', 'var'))
        varargout{outputid} = newtraj;
    end
end
