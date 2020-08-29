function varargout=mcxlab(varargin)
%
%====================================================================
%      MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave
%--------------------------------------------------------------------
%Copyright (c) 2011-2020 Qianqian Fang <q.fang at neu.edu>
%                      URL: http://mcx.space
%====================================================================
%
% Format:
%    fluence=mcxlab(cfg);
%       or
%    [fluence,detphoton,vol,seed,trajectory]=mcxlab(cfg);
%    [fluence,detphoton,vol,seed,trajectory]=mcxlab(cfg, option);
%
% Input:
%    cfg: a struct, or struct array. Each element of cfg defines 
%         the parameters associated with a simulation. 
%         if cfg='gpuinfo': return the supported GPUs and their parameters,
%         see sample script at the bottom
%    option: (optional), options is a string, specifying additional options
%         option='preview': this plots the domain configuration using mcxpreview(cfg)
%         option='opencl':  force using mcxcl.mex* instead of mcx.mex* on NVIDIA/AMD/Intel hardware
%         option='cuda':    force using mcx.mex* instead of mcxcl.mex* on NVIDIA GPUs
%
%    if one defines USE_MCXCL=1 in MATLAB command line window, all following
%    mcxlab and mcxlabcl calls will use mcxcl.mex; by setting option='cuda', one can
%    force both mcxlab and mcxlabcl to use mcx (cuda version). Similarly, if
%    USE_MCXCL=0, all mcxlabcl and mcxlab call will use mcx.mex by default, unless
%    one set option='opencl'.
%
%    cfg may contain the following fields:
%
%== Required ==
%     *cfg.nphoton:    the total number of photons to be simulated (integer)
%                      maximum supported value is 2^63-1
%     *cfg.vol:        a 3D array specifying the media index in the domain.
%                      can be uint8, uint16, uint32, single or double
%                      arrays.
%                      2D simulations are supported if cfg.vol has a singleton
%                      dimension (in x or y); srcpos/srcdir must belong to
%                      the 2D plane in such case.
%                      for 2D simulations, Example: <demo_mcxlab_2d.m>
%
%                      MCXLAB also accepts 4D arrays to define continuously varying media. 
%                      The following formats are accepted
%                        1 x Nx x Ny x Nz float32 array: mua values for each voxel (must use permute to make 1st dimension singleton)
%                        2 x Nx x Ny x Nz float32 array: mua/mus values for each voxel (g/n use prop(2,:))
%                        4 x Nx x Ny x Nz uint8 array: mua/mus/g/n gray-scale (0-255) interpolating between prop(2,:) and prop(3,:)
%                        2 x Nx x Ny x Nz uint16 array: mua/mus gray-scale (0-65535) interpolating between prop(2,:) and prop(3,:)
%                        Example: <demo_continuous_mua_mus.m>. If voxel-based media are used, partial-path/momentum outputs are disabled
%     *cfg.prop:       an N by 4 array, each row specifies [mua, mus, g, n] in order.
%                      the first row corresponds to medium type 0
%                      (background) which is typically [0 0 1 1]. The
%                      second row is type 1, and so on. The background
%                      medium (type 0) has special meanings: a photon
%                      terminates when moving from a non-zero to zero voxel.
%     *cfg.tstart:     starting time of the simulation (in seconds)
%     *cfg.tstep:      time-gate width of the simulation (in seconds)
%     *cfg.tend:       ending time of the simulation (in second)
%     *cfg.srcpos:     a 1 by 3 vector, the position of the source in grid unit
%     *cfg.srcdir:     a 1 by 3 vector, specifying the incident vector; if srcdir
%                      contains a 4th element, it specifies the focal length of
%                      the source (only valid for focuable src, such as planar, disk,
%                      fourier, gaussian, pattern, slit, etc); if the focal length
%                      is nan, all photons will be launched isotropically regardless
%                      of the srcdir direction.
%
%== MC simulation settings ==
%      cfg.seed:       seed for the random number generator (integer) [0]
%                      if set to a uint8 array, the binary data in each column is used 
%                      to seed a photon (i.e. the "replay" mode)
%                      Example: <demo_mcxlab_replay.m>
%      cfg.respin:     repeat simulation for the given time (integer) [1]
%                      if negative, divide the total photon number into respin subsets
%      cfg.isreflect:  [1]-consider refractive index mismatch, 0-matched index
%      cfg.bc          per-face boundary condition (BC), a strig of 6 letters (case insensitive) for
%                      bounding box faces at -x,-y,-z,+x,+y,+z axes;
%		               overwrite cfg.isreflect if given.
%                      each letter can be one of the following:
%                      '_': undefined, fallback to cfg.isreflect
%                      'r': like cfg.isreflect=1, Fresnel reflection BC
%                      'a': like cfg.isreflect=0, total absorption BC
%                      'm': mirror or total reflection BC
%                      'c': cyclic BC, enter from opposite face
%
%                      in addition, cfg.bc can contain up to 12 characters,
%                      with the 7-12 characters indicating bounding box
%                      facets -x,-y,-z,+x,+y,+z are used as a detector. The 
%                      acceptable characters for digits 7-12 include
%                      '0': this face is not used to detector photons
%                      '1': this face is used to capture photons (if output detphoton)
%                      see <demo_bc_det.m>
%      cfg.isnormalized:[1]-normalize the output fluence to unitary source, 0-no reflection
%      cfg.isspecular: 1-calculate specular reflection if source is outside, [0] no specular reflection
%      cfg.maxgate:    the num of time-gates per simulation
%      cfg.minenergy:  terminate photon when weight less than this level (float) [0.0]
%      cfg.unitinmm:   defines the length unit for a grid edge length [1.0]
%                      Example: <demo_sphere_cube_subpixel.m>
%      cfg.shapes:     a JSON string for additional shapes in the grid
%                      Example: <demo_mcxyz_skinvessel.m>
%      cfg.gscatter:   after a photon completes the specified number of
%                      scattering events, mcx then ignores anisotropy g
%                      and only performs isotropic scattering for speed [1e9]
%      cfg.detphotons: detected photon data for replay. In the replay mode (cfg.seed 
%                      is set as the 4th output of the baseline simulation), cfg.detphotons
%                      should be set to the 2nd output (detphoton) of the baseline simulation
%                      or detphoton.data subfield (as a 2D array). cfg.detphotons can use
%                      a subset of the detected photon selected by the user.
%                      Example: <demo_mcxlab_replay.m>
%
%== GPU settings ==
%      cfg.autopilot:  1-automatically set threads and blocks, [0]-use nthread/nblocksize
%      cfg.nblocksize: how many CUDA thread blocks to be used [64]
%      cfg.nthread:    the total CUDA thread number [2048]
%      cfg.gpuid:      which GPU to use (run 'mcx -L' to list all GPUs) [1]
%                      if set to an integer, gpuid specifies the index (starts at 1)
%                      of the GPU for the simulation; if set to a binary string made
%                      of 1s and 0s, it enables multiple GPUs. For example, '1101'
%                      allows to use the 1st, 2nd and 4th GPUs together.
%                      Example: <mcx_gpu_benchmarks.m>
%      cfg.workload    an array denoting the relative loads of each selected GPU. 
%                      for example, [50,20,30] allocates 50%, 20% and 30% photons to the
%                      3 selected GPUs, respectively; [10,10] evenly divides the load 
%                      between 2 active GPUs. A simple load balancing strategy is to 
%                      use the GPU core counts as the weight.
%      cfg.isgpuinfo:  1-print GPU info, [0]-do not print
%
%== Source-detector parameters ==
%      cfg.detpos:     an N by 4 array, each row specifying a detector: [x,y,z,radius]
%      cfg.maxdetphoton:   maximum number of photons saved by the detectors [1000000]
%      cfg.srctype:    source type, the parameters of the src are specified by cfg.srcparam{1,2}
%                              Example: <demo_mcxlab_srctype.m>
%                      'pencil' - default, pencil beam, no param needed
%                      'isotropic' - isotropic source, no param needed
%                      'cone' - uniform cone beam, srcparam1(1) is the half-angle in radian
%                      'gaussian' [*] - a collimated gaussian beam, srcparam1(1) specifies the waist radius (in voxels)
%                      'planar' [*] - a 3D quadrilateral uniform planar source, with three corners specified 
%                                by srcpos, srcpos+srcparam1(1:3) and srcpos+srcparam2(1:3)
%                      'pattern' [*] - a 3D quadrilateral pattern illumination, same as above, except
%                                srcparam1(4) and srcparam2(4) specify the pattern array x/y dimensions,
%                                and srcpattern is a floating-point pattern array, with values between [0-1]. 
%                                if cfg.srcnum>1, srcpattern must be a floating-point array with 
%                                a dimension of [srcnum srcparam1(4) srcparam2(4)]
%                                Example: <demo_photon_sharing.m>
%                      'pattern3d' [*] - a 3D illumination pattern. srcparam1{x,y,z} defines the dimensions,
%                                and srcpattern is a floating-point pattern array, with values between [0-1]. 
%                      'fourier' [*] - spatial frequency domain source, similar to 'planar', except
%                                the integer parts of srcparam1(4) and srcparam2(4) represent
%                                the x/y frequencies; the fraction part of srcparam1(4) multiplies
%                                2*pi represents the phase shift (phi0); 1.0 minus the fraction part of
%                                srcparam2(4) is the modulation depth (M). Put in equations:
%                                    S=0.5*[1+M*cos(2*pi*(fx*x+fy*y)+phi0)], (0<=x,y,M<=1)
%                      'arcsine' - similar to isotropic, except the zenith angle is uniform
%                                distribution, rather than a sine distribution.
%                      'disk' [*] - a uniform disk source pointing along srcdir; the radius is 
%                               set by srcparam1(1) (in grid unit)
%                      'fourierx' [*] - a general Fourier source, the parameters are 
%                               srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phi0,M]
%                               normalized vectors satisfy: srcdir cross v1=v2
%                               the phase shift is phi0*2*pi
%                      'fourierx2d' [*] - a general 2D Fourier basis, parameters
%                               srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phix,phiy]
%                               the phase shift is phi{x,y}*2*pi
%                      'zgaussian' - an angular gaussian beam, srcparam1(0) specifies the variance in the zenith angle
%                      'line' - a line source, emitting from the line segment between 
%                               cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), radiating 
%                               uniformly in the perpendicular direction
%                      'slit' [*] - a colimated slit beam emitting from the line segment between 
%                               cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), with the initial  
%                               dir specified by cfg.srcdir
%                      'pencilarray' - a rectangular array of pencil beams. The srcparam1 and srcparam2
%                               are defined similarly to 'fourier', except that srcparam1(4) and srcparam2(4)
%                               are both integers, denoting the element counts in the x/y dimensions, respectively. 
%                               For exp., srcparam1=[10 0 0 4] and srcparam2[0 20 0 5] represent a 4x5 pencil beam array
%                               spanning 10 grids in the x-axis and 20 grids in the y-axis (5-voxel spacing)
%                      source types marked with [*] can be focused using the
%                      focal length parameter (4th element of cfg.srcdir)
%      cfg.{srcparam1,srcparam2}: 1x4 vectors, see cfg.srctype for details
%      cfg.srcpattern: see cfg.srctype for details
%      cfg.srcnum:     the number of source patterns that are
%                      simultaneously simulated; only works for 'pattern'
%                      source, see cfg.srctype='pattern' for details
%                      Example <demo_photon_sharing.m>
%      cfg.issrcfrom0: 1-first voxel is [0 0 0], [0]- first voxel is [1 1 1]
%      cfg.replaydet:  only works when cfg.outputtype is 'jacobian', 'wl', 'nscat', or 'wp' and cfg.seed is an array
%                      -1 replay all detectors and save in separate volumes (output has 5 dimensions)
%                       0 replay all detectors and sum all Jacobians into one volume
%                       a positive number: the index of the detector to replay and obtain Jacobians
%      cfg.voidtime:   for wide-field sources, [1]-start timer at launch, or 0-when entering 
%                      the first non-zero voxel
%
%== Output control ==
%      cfg.savedetflag: ['dp'] - a string (case insensitive) controlling the output detected photon data fields
%                          1 d  output detector ID (1)
%                          2 s  output partial scat. even counts (#media)
%                          4 p  output partial path-lengths (#media)
%                          8 m  output momentum transfer (#media)
%                         16 x  output exit position (3)
%                         32 v  output exit direction (3)
%                         64 w  output initial weight (1)
%                      combine multiple items by using a string, or add selected numbers together
%                      by default, mcx only saves detector ID (d) and partial-path data (p)
%      cfg.issaveexit: [0]-save the position (x,y,z) and (vx,vy,vz) for a detected photon
%                      same as adding 'xv' to cfg.savedetflag. Example: <demo_lambertian_exit_angle.m>
%      cfg.ismomentum: 1 to save photon momentum transfer,[0] not to save.
%                      save as adding 'M' to cfg.savedetflag string
%      cfg.issaveref:  [0]-save diffuse reflectance/transmittance in the non-zero voxels
%                      next to a boundary voxel. The reflectance data are stored as 
%                      negative values; must pad zeros next to boundaries
%                      Example: see the demo script at the bottom
%      cfg.outputtype: 'flux' - fluence-rate, (default value)
%                      'fluence' - fluence integrated over each time gate, 
%                      'energy' - energy deposit per voxel
%                      'jacobian' or 'wl' - mua Jacobian (replay mode), 
%                      'nscat' or 'wp' - weighted scattering counts for computing Jacobian for mus (replay mode)
%                      for type jacobian/wl/wp, example: <demo_mcxlab_replay.m>
%                      and  <demo_replay_timedomain.m>
%      cfg.session:    a string for output file names (only used when no return variables)
%
%== Debug ==
%      cfg.debuglevel:  debug flag string (case insensitive), one or a combination of ['R','M','P'], no space
%                    'R':  debug RNG, output fluence.data is filled with 0-1 random numbers
%                    'M':  return photon trajectory data as the 5th output
%                    'P':  show progress bar
%      cfg.maxjumpdebug: [10000000|int] when trajectory is requested in the output, 
%                     use this parameter to set the maximum position stored. By default,
%                     only the first 1e6 positions are stored.
%
%      fields with * are required; options in [] are the default values
%
% Output:
%      fluence: a struct array, with a length equals to that of cfg.
%            For each element of fluence, 
%            fluence(i).data is a 4D array with
%                 dimensions specified by [size(vol) total-time-gates]. 
%                 The content of the array is the normalized fluence at 
%                 each voxel of each time-gate.
%            fluence(i).dref is a 4D array with the same dimension as fluence(i).data
%                 if cfg.issaveref is set to 1, containing only non-zero values in the 
%                 layer of voxels immediately next to the non-zero voxels in cfg.vol,
%                 storing the normalized total diffuse reflectance (summation of the weights 
%                 of all escaped photon to the background regardless of their direction);
%                 it is an empty array [] when if cfg.issaveref is 0.
%            fluence(i).stat is a structure storing additional information, including
%                 runtime: total simulation run-time in millisecond
%                 nphoton: total simulated photon number
%                 energytot: total initial weight/energy of all launched photons
%                 energyabs: total absorbed weight/energy of all photons
%                 normalizer: normalization factor
%                 unitinmm: same as cfg.unitinmm, voxel edge-length in mm
%
%      detphoton: (optional) a struct array, with a length equals to that of cfg.
%            Starting from v2018, the detphoton contains the below subfields:
%              detphoton.detid: the ID(>0) of the detector that captures the photon
%              detphoton.nscat: cummulative scattering event counts in each medium
%              detphoton.ppath: cummulative path lengths in each medium (partial pathlength)
%                   one need to multiply cfg.unitinmm with ppath to convert it to mm.
%              detphoton.mom: cummulative cos_theta for momentum transfer in each medium  
%              detphoton.p or .v: exit position and direction, when cfg.issaveexit=1
%              detphoton.w0: photon initial weight at launch time
%              detphoton.prop: optical properties, a copy of cfg.prop
%              detphoton.data: a concatenated and transposed array in the order of
%                    [detid nscat ppath mom p v w0]'
%              "data" is the is the only subfield in all MCXLAB before 2018
%      vol: (optional) a struct array, each element is a preprocessed volume
%            corresponding to each instance of cfg. Each volume is a 3D int32 array.
%      seeds: (optional), if give, mcxlab returns the seeds, in the form of
%            a byte array (uint8) for each detected photon. The column number
%            of seed equals that of detphoton.
%      trajectory: (optional), if given, mcxlab returns the trajectory data for
%            each simulated photon. The output has 6 rows, the meanings are 
%               id:  1:    index of the photon packet
%               pos: 2-4:  x/y/z/ of each trajectory position
%                    5:    current photon packet weight
%                    6:    reserved
%            By default, mcxlab only records the first 1e7 positions along all
%            simulated photons; change cfg.maxjumpdebug to define a different limit.
%
%
% Example:
%      % first query if you have supported GPU(s)
%      info=mcxlab('gpuinfo')
%
%      % define the simulation using a struct
%      cfg.nphoton=1e7;
%      cfg.vol=uint8(ones(60,60,60));
%      cfg.vol(20:40,20:40,10:30)=2;    % add an inclusion
%      cfg.prop=[0 0 1 1;0.005 1 0 1.37; 0.2 10 0.9 1.37]; % [mua,mus,g,n]
%      cfg.issrcfrom0=1;
%      cfg.srcpos=[30 30 1];
%      cfg.srcdir=[0 0 1];
%      cfg.detpos=[30 20 1 1;30 40 1 1;20 30 1 1;40 30 1 1];
%      cfg.vol(:,:,1)=0;   % pad a layer of 0s to get diffuse reflectance
%      cfg.issaveref=1;
%      cfg.gpuid=1;
%      cfg.autopilot=1;
%      cfg.tstart=0;
%      cfg.tend=5e-9;
%      cfg.tstep=5e-10;
%      % calculate the fluence distribution with the given config
%      [fluence,detpt,vol,seeds,traj]=mcxlab(cfg);
%
%      % integrate time-axis (4th dimension) to get CW solutions
%      cwfluence=sum(fluence.data,4);  % fluence rate
%      cwdref=sum(fluence.dref,4);     % diffuse reflectance
%      % plot configuration and results
%      subplot(231);
%      mcxpreview(cfg);title('domain preview');
%      subplot(232);
%      imagesc(squeeze(log(cwfluence(:,30,:))));title('fluence at y=30');
%      subplot(233);
%      hist(detpt.ppath(:,1),50); title('partial path tissue#1');
%      subplot(234);
%      plot(squeeze(fluence.data(30,30,30,:)),'-o');title('TPSF at [30,30,30]');
%      subplot(235);
%      newtraj=mcxplotphotons(traj);title('photon trajectories')
%      subplot(236);
%      imagesc(squeeze(log(cwdref(:,:,1))));title('diffuse refle. at z=1');
%
% This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

try
    defaultocl=evalin('base','USE_MCXCL');
catch
    defaultocl=0;
end

useopencl=defaultocl;

if(nargin==2 && ischar(varargin{2}))
    if(strcmp(varargin{2},'preview'))
        [varargout{1:nargout}]=mcxpreview(varargin{1});
	    return;
    elseif(strcmp(varargin{2},'opencl'))
        useopencl=1;
    end
end

if(isstruct(varargin{1}))
    for i=1:length(varargin{1})
        castlist={'srcpattern','srcpos','detpos','prop','workload','srcdir'};
        for j=1:length(castlist)
            if(isfield(varargin{1}(i),castlist{j}))
                varargin{1}(i).(castlist{j})=double(varargin{1}(i).(castlist{j}));
            end
        end
        if (isfield(varargin{1}(i),'vol') && ndims(varargin{1}(i).vol)==4)
            if((isa(varargin{1}(i).vol,'single') || isa(varargin{1}(i).vol,'double')) && isfield(varargin{1}(i),'unitinmm'))
                varargin{1}(i).vol=varargin{1}(i).vol*varargin{1}(i).unitinmm;
            end
        end
        if (~isfield(varargin{1}(i),'tstart'))
            varargin{1}(i).tstart=0;
        end
        if (~isfield(varargin{1}(i),'tend'))
            error('you must define cfg.tend for the maximum time-of-flight of a photon in seconds');
        end
        if (~isfield(varargin{1}(i),'tstep'))
            varargin{1}(i).tstep=varargin{1}(i).tend;
        end
        if (~isfield(varargin{1}(i),'srcpos'))
            error('you must define cfg.srcpos to defin the x/y/z position of the source in voxel unit');
        end
        if(isfield(varargin{1}(i),'detphotons') && isstruct(varargin{1}(i).detphotons))
            if(isfield(varargin{1}(i).detphotons,'data'))
                varargin{1}(i).detphotons=varargin{1}(i).detphotons.data;
            else
                fulldetdata={'detid','nscat','ppath','mom','p','v','w0'};
                detfields=ismember(fulldetdata,fieldnames(varargin{1}(i).detphotons));
                detdata=[];
                for j=1:length(detfields)
                    if(detfields(j))
                        val=typecast(varargin{1}(i).detphotons.(fulldetdata{j})(:),'single');
                        detdata=[detdata reshape(val,size(varargin{1}(i).detphotons.(fulldetdata{j})))];
                    end
                end
                varargin{1}(i).detphotons=detdata';
                varargin{1}(i).savedetflag='dspmxvw';
                varargin{1}(i).savedetflag(detfields==0)=[];
            end
        end
    end
end

if(useopencl==0)
    [varargout{1:max(1,nargout)}]=mcx(varargin{1});
else
    [varargout{1:max(1,nargout)}]=mcxcl(varargin{1});
end

if(nargin==0)
    return;
end

cfg=varargin{1};

if(~ischar(cfg))
    for i=1:length(varargout{1})
        if(isfield(cfg(i),'srcnum') && cfg(i).srcnum>1)
            dim=size(varargout{1}(i).data);
            varargout{1}(i).data=reshape(varargout{1}(i).data,[cfg(i).srcnum, dim(1)/cfg(i).srcnum dim(2:end)]);
            varargout{1}(i).data=permute(varargout{1}(i).data,[2:(length(dim)+1) 1]);
            if(isfield(varargout{1}(i),'dref') && ~isempty(varargout{1}(i).dref))
                varargout{1}(i).dref=reshape(varargout{1}(i).dref,[cfg(i).srcnum, dim(1)/cfg(i).srcnum dim(2:end)]);
                varargout{1}(i).dref=permute(varargout{1}(i).dref,[2:(length(dim)+1) 1]);
            end
        end
    end
end

if(nargout>=2)

    for i=1:length(varargout{2})
        if((~isfield(cfg(i),'savedetflag')) || ((isfield(cfg(i),'savedetflag')) && isempty(cfg(i).savedetflag)))
            cfg(i).savedetflag='DP';
        end
        if(isfield(cfg(i),'issaveexit') && cfg(i).issaveexit)
            cfg(i).savedetflag=[cfg(i).savedetflag,'XV'];
        end
        if(isfield(cfg(i),'ismomentum') && cfg(i).ismomentum)
            cfg(i).savedetflag=[cfg(i).savedetflag,'M'];
        end
        if(ndims(cfg(i).vol)==4)
            cfg(i).savedetflag='';
            if((isa(cfg(i).vol,'single') || isa(cfg(i).vol,'double')) && isfield(cfg(i),'unitinmm'))
                cfg(i).vol=cfg(i).vol*cfg(i).unitinmm;
            end
        end
        if((~isfield(cfg(i),'issaveexit') || cfg(i).issaveexit~=2))
            medianum=size(cfg(i).prop,1)-1;
            detp=varargout{2}(i).data;
            if(isempty(detp))
                continue;
            end
            flags={cfg(i).savedetflag};
            if(isfield(cfg(i),'issaveref'))
                flags{end+1}=cfg(i).issaveref;
            end
            if(isfield(cfg(i),'srcnum'))
                flags{end+1}=cfg(i).srcnum;
            end
            newdetp=mcxdetphoton(detp,medianum,flags{:});
            newdetp.prop=cfg(i).prop;
            if(isfield(cfg(i),'unitinmm'))
                newdetp.unitinmm=cfg(i).unitinmm;
            end
            newdetp.data=detp;      % enable this line for compatibility
            newdetpstruct(i)=newdetp;
        else
            newdetpstruct(i)=varargout{2}(i);
        end
    end
    if(exist('newdetpstruct','var'))
        varargout{2}=newdetpstruct;
    end

    if(nargout>=5)
        for i=1:length(varargout{5})
            data=varargout{5}.data;
            if(isempty(data))
               continue;
            end
            traj.pos=data(2:4,:).';
            traj.id=typecast(data(1,:),'uint32').';
            [traj.id,idx]=sort(traj.id);
            traj.pos=traj.pos(idx,:);
            traj.data=[single(traj.id)' ; data(2:end,idx)];
            newtraj(i)=traj;
        end
        if(exist('newtraj','var'))
            varargout{5}=newtraj;
        end
    end
end
