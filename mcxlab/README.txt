= MCXLAB: MCX for MATLAB and GNU Octave =

Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
License: GNU General Public License version 3 (GPLv3)
Version: this package is part of Monte Carlo eXtreme (MCX) v2016.4

<toc>


== # Introduction ==

MCXLAB is the native MEX version of MCX for Matlab and GNU Octave. It compiles
the entire MCX code into a MEX function which can be called directly inside
Matlab or Octave. The input and output files in MCX are replaced by convenient
in-memory struct variables in MCXLAB, thus, making it much easier to use 
and interact. Matlab/Octave also provides convenient plotting and data
analysis functions. With MCXLAB, your analysis can be streamlined and speed-
up without involving disk files.

Because MCXLAB contains the exact computational codes for the GPU calculations
as in the MCX binaries, MCXLAB is expected to have identical performance when
running simulations. By default, we compile MCXLAB with the support of recording
detected photon partial path-lengths (i.e. the "make det" option). In addition,
we also provide "mcxlab_atom": an atomic version of mcxlab compiled similarly
as "make detbox" for MCX. It supports atomic operations using shared memory
enabled by setting "cfg.sradius" input parameter to a positive number.


== # Installation ==

To download MCXLAB, please visit [http://mcx.sourceforge.net/cgi-bin/index.cgi?Download#Download_the_Latest_Release this link]. 
If you choose to [http://mcx.sourceforge.net/cgi-bin/index.cgi?register/mcx register], 
you will have an option to be notified for any future updates.

The system requirements for MCXLAB are the same as MCX: you have to make
sure that you have a CUDA-capable graphics card with properly configured 
CUDA driver (you can run the standard MCX binary first to test if your 
system is capable to run MCXLAB). Of course, you need to have either Matlab
or Octave installed.

Once you set up the CUDA toolkit and NVIDIA driver, you can then add the 
"mcxlab" directory to your Matlab/Octave search path using the addpath command.
If you want to add this path permanently, please use the "pathtool" 
command, or edit your startup.m (~/.octaverc for Octave).

If everything works ok, typing "help mcxlab" in Matlab/Octave will print the
help information. If you see any error, particularly any missing libraries,
please make sure you have downloaded the matching version built for your
platform.


== # How to use MCXLAB in MATLAB/Octave ==

To learn the basic usage of MCXLAB, you can type

  help mcxlab

in Matlab/Octave to see the help information regarding how to use this 
function. The help information is listed below. You can find the input/output 
formats and examples. The input cfg structure has very similar field names as
the verbose command line options in MCX.

<pre> ====================================================================
       MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave
 --------------------------------------------------------------------
 Copyright (c) 2011-2016 Qianqian Fang <q.fang at neu.edu>
                       URL: http://mcx.space
 ====================================================================
 
  Format:
     [flux,detphoton,vol,seed]=mcxlab(cfg);
 
  Input:
     cfg: a struct, or struct array. Each element of cfg defines 
          the parameters associated with a simulation. 
 
     It may contain the following fields:
 
      *cfg.nphoton:    the total number of photons to be simulated (integer)
      *cfg.vol:        a 3D array specifying the media index in the domain
      *cfg.prop:       an N by 4 array, each row specifies [mua, mus, g, n] in order.
                       the first row corresponds to medium type 0 which is 
                       typically [0 0 1 1]. The second row is type 1, and so on.
      *cfg.tstart:     starting time of the simulation (in seconds)
      *cfg.tstep:      time-gate width of the simulation (in seconds)
      *cfg.tend:       ending time of the simulation (in second)
      *cfg.srcpos:     a 1 by 3 vector, the position of the source in grid unit
      *cfg.srcdir:     a 1 by 3 vector, specifying the incident vector
       cfg.sradius:    radius within which we use atomic operations (in grid) [0.0]
                       sradius=0 to disable atomic operations; if sradius=-1,
                       use cfg.crop0 and crop1 to define a cubic atomic zone; if
                       sradius=-2, perform atomic operations in the entire domain;
                       by default, srandius=-2 (atomic operations is used).
       cfg.nblocksize: how many CUDA thread blocks to be used [64]
       cfg.nthread:    the total CUDA thread number [2048]
       cfg.maxgate:    the num of time-gates per simulation
       cfg.session:    a string for output file names (used when no return variables)
       cfg.seed:       seed for the random number generator (integer) [0]
                       if set to a uint8 array, the binary data in each column is used 
                       to seed a photon (i.e. the "replay" mode)
       cfg.maxdetphoton:   maximum number of photons saved by the detectors [1000000]
       cfg.detpos:     an N by 4 array, each row specifying a detector: [x,y,z,radius]
       cfg.respin:     repeat simulation for the given time (integer) [1]
       cfg.gpuid:      which GPU to use (run 'mcx -L' to list all GPUs) [1]
                       if set to an integer, gpuid specifies the index (starts at 1)
                       of the GPU for the simulation; if set to a binary string made
                       of 1s and 0s, it enables multiple GPUs. For example, '1101'
                       allows to use the 1st, 2nd and 4th GPUs together.
       cfg.workload    an array denoting the relative loads of each selected GPU. 
                       for example, [50,20,30] allocates 50%, 20% and 30% photons to the
                       3 selected GPUs, respectively; [10,10] evenly divides the load 
                       between 2 active GPUs. A simple load balancing strategy is to 
                       use the GPU core counts as the weight.
       cfg.isreflect:  [1]-consider refractive index mismatch, 0-matched index
       cfg.isrefint:   1-ref. index mismatch at inner boundaries, [0]-matched index
       cfg.isnormalized:[1]-normalize the output flux to unitary source, 0-no reflection
       cfg.issrcfrom0: 1-first voxel is [0 0 0], [0]- first voxel is [1 1 1]
       cfg.isgpuinfo:  1-print GPU info, [0]-do not print
       cfg.autopilot:  1-automatically set threads and blocks, [0]-use nthread/nblocksize
       cfg.minenergy:  terminate photon when weight less than this level (float) [0.0]
       cfg.unitinmm:   defines the length unit for a grid edge length [1.0]
       cfg.shapes:     a JSON string for additional shapes in the grid
       cfg.reseedlimit:number of scattering events before reseeding RNG
       cfg.srctype:    source type, the parameters of the src are specified by cfg.srcparam{1,2}
                       'pencil' - default, pencil beam, no param needed
                       'isotropic' - isotropic source, no param needed
                       'cone' - uniform cone beam, srcparam1(1) is the half-angle in radian
                       'gaussian' - a collimated gaussian beam, srcparam1(1) specifies the waist radius (in voxels)
                       'planar' - a 3D quadrilateral uniform planar source, with three corners specified 
                                 by srcpos, srcpos+srcparam1(1:3) and srcpos+srcparam2(1:3)
                       'pattern' - a 3D quadrilateral pattern illumination, same as above, except
                                 srcparam1(4) and srcparam2(4) specify the pattern array x/y dimensions,
                                 and srcpattern is a pattern array, valued between [0-1]. 
                       'fourier' - spatial frequency domain source, similar to 'planar', except
                                 the integer parts of srcparam1(4) and srcparam2(4) represent
                                 the x/y frequencies; the fraction part of srcparam1(4) multiplies
                                 2*pi represents the phase shift (phi0); 1.0 minus the fraction part of
                                 srcparam2(4) is the modulation depth (M). Put in equations:
                                     S=0.5*[1+M*cos(2*pi*(fx*x+fy*y)+phi0)], (0<=x,y,M<=1)
                       'arcsine' - similar to isotropic, except the zenith angle is uniform
                                 distribution, rather than a sine distribution.
                       'disk' - a uniform disk source pointing along srcdir; the radius is 
                                set by srcparam1(1) (in grid unit)
                       'fourierx' - a general Fourier source, the parameters are 
                                srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phi0,M]
                                normalized vectors satisfy: srcdir cross v1=v2
                                the phase shift is phi0*2*pi
                       'fourierx2d' - a general 2D Fourier basis, parameters
                                srcparam1: [v1x,v1y,v1z,|v2|], srcparam2: [kx,ky,phix,phiy]
                                the phase shift is phi{x,y}*2*pi
                       'zgaussian' - an angular gaussian beam, srcparam1(0) specifies the variance in the zenith angle
                       'line' - a line source, emitting from the line segment between 
                                cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), radiating 
                                uniformly in the perpendicular direction
                       'slit' - a colimated slit beam emitting from the line segment between 
                                cfg.srcpos and cfg.srcpos+cfg.srcparam(1:3), with the initial  
                                dir specified by cfg.srcdir
       cfg.{srcparam1,srcparam2}: 1x4 vectors, see cfg.srctype for details
       cfg.srcpattern: see cfg.srctype for details
       cfg.voidtime:   for wide-field sources, [1]-start timer at launch, or 0-when entering 
                       the first non-zero voxel
       cfg.outputtype:  [X] - output flux, F - fluence, E - energy deposit
                        J - Jacobian (replay)
       cfg.faststep: when set to 1, this option enables the legacy 1mm fix-step photon
                     advancing strategy; although this method is fast, the results were
                     found inaccurate, and therefore is not recommended. Setting to 0
                     enables precise ray-tracing between voxels; this is the default.
 
       fields with * are required; options in [] are the default values
 
  Output:
       flux: a struct array, with a length equals to that of cfg.
             For each element of flux, flux(i).data is a 4D array with
             dimensions specified by [size(vol) total-time-gates]. 
             The content of the array is the normalized flux at 
             each voxel of each time-gate.
       detphoton: a struct array, with a length equals to that of cfg.
             For each element of detphoton, detphoton(i).data is a 2D array with
             dimensions [size(cfg.prop,1)+1 saved-photon-num]. The first row
             is the ID(>0) of the detector that captures the photon; the second row
 	     saves the number of scattering events of each exiting photon; the rest rows
 	     are the partial path lengths (in grid unit) traveling in medium 1 up 
             to the last. If you set cfg.unitinmm, you need to multiply the path-lengths
             to convert them to mm unit.
       vol: (optional) a struct array, each element is a preprocessed volume
             corresponding to each instance of cfg. Each volume is a 3D uint8 array.
       seeds: (optional), if give, mcxlab returns the seeds, in the form of
             a byte array (uint8) for each detected photon. The column number
             of seed equals that of detphoton.
 
 
  Example:
       cfg.nphoton=1e7;
       cfg.vol=uint8(ones(60,60,60));
       cfg.srcpos=[30 30 1];
       cfg.srcdir=[0 0 1];
       cfg.gpuid=1;
       cfg.autopilot=1;
       cfg.prop=[0 0 1 1;0.005 1 0 1.37];
       cfg.tstart=0;
       cfg.tend=5e-9;
       cfg.tstep=5e-10;
       % calculate the flux distribution with the given config
       flux=mcxlab(cfg);
 
       cfgs(1)=cfg;
       cfgs(2)=cfg;
       cfgs(1).isreflect=0;
       cfgs(2).isreflect=1;
       cfgs(2).issavedet=1;
       cfgs(2).detpos=[30 20 1 1;30 40 1 1;20 30 1 1;40 30 1 1];
       % calculate the flux and partial path lengths for the two configurations
       [fluxs,detps]=mcxlab(cfgs);
 
       imagesc(squeeze(log(fluxs(1).data(:,30,:,1)))-squeeze(log(fluxs(2).data(:,30,:,1))));
 
 
  This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
 
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

==== demo_sphere_cube_subpixel.m ====

In this example, we demonstrate how to use sub-pixel resolution 
to represent the problem domain. The domain is consisted of a 
6x6x6 cm box with a 2cm diameter sphere embedded at the center.

==== demo_4layer_head.m ====

In this example, we simulate a 4-layer brain model using MCXLAB.
We will investigate the differences between the solutions with and 
witout boundary reflections (both external and internal) and show
you how to display and analyze the resulting data.

==== demo_mcxlab_srctype.m ====
This demo script shows how to use 9 different types sources in your
simulations. These 9 source types include pencil beam, isotropic source,
Gaussian beam, uniform plannar source, uniform disk source, Fourier 
pattern illumuniation (spatial frequency domain sources), arcsine 
distribution beam, uniform cone beam, and an arbitrary light pattern 
(defined by a 2D image).

==== bench_reseedlimit.m ====
In this simulation, we verify the reduction of Monte Carlo stochastic
noise with respect to the increase of photon numbers. We also test the
effect of the reseedlimit parameter and its impact to the noise.


== # How to compile MCXLAB ==

To compile MCXLAB for Matlab, you need to cd mcx/src directory, and type 

 make mex

from a shell window. You need to make sure your Matlab is installed and 
the command <tt>mex</tt> is included in your PATH environment variable. Similarly, 
to compile MCXLAB for Octave, you type

 make oct

The command <tt>mkoctfile</tt> must be accessible from your command line
and it is provided in a package named "octave3.x-headers" in Ubuntu (3.x
can be 3.2 or 3.4 etc).

If your graphics card is a Fermi-class or newer, you can compile MCXLAB
with make fermimex or fermioct. The output mex file can determine the
level of atomic operations using the cfg.sradius settings.

== # Screenshots ==

Screenshot for using MCXLAB in Matlab:
  http://mcx.sourceforge.net/upload/matlab_mcxlab.png

Screenshot for using MCXLAB in GNU Octave:
  http://mcx.sourceforge.net/upload/octave_mcxlab.png


== # Reference ==

 [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation 
  of photon migration in 3D turbid media accelerated by graphics processing 
  units," Opt. Express 17, 20178-20190 (2009)

 [Fang2010] Fang Q, "Mesh-based Monte Carlo method using fast ray-tracing
   in Plucker coordinates," Biomed. Opt. Express 1, 165-175 (2010) 

