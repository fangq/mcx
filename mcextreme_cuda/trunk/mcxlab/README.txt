= MCXLAB: MCX for MATLAB and GNU Octave =

<toc>

== # Introduction ==

MCXLAB is the native MEX version of MCX for Matlab and GNU Octave. It compiles
the entire MCX code into a MEX function which can be called directly inside
Matlab or Octave. The input and output files in MCX are replaced by convenient
in-memory struct variables in MCXLAB, thus, making it much easier to use 
and interact. Matlab/Octave also provides convenient plotting and data
analysis functions. With MCXLAB, your analysis can be streamlined and speed
up without involving disk files.

Because MCXLAB contains the exact instructions for the GPU calculations
as in the MCX binaries, MCXLAB is expected to have identical performance when
running simulations. By default, we compile MCXLAB with the support of recording
detected photon partial path-lengths (i.e. the "make det" option). 

The system requirement for MCXLAB is the same as MCX, you have to make
sure that you have a CUDA-capable graphics card with properly configured 
CUDA driver (you can run the standard MCX binary first to test if your 
system is capable to run MCXLAB). Of course, you need to have either Matlab
or Octave installed.

== # How to use MCXLAB in MATLAB/Octave ==

You can type
  help mcxlab
or simply
  mcxlab
and enter in Matlab/Octave to see the help information regarding how to use this 
function. The help information is listed below, where you can find the input/output 
formats and examples. The input cfg structure has very similar field names as
the command line options in MCX.

<pre>====================================================================
      MCXLAB - Monte Carlo eXtreme (MCX) for MATLAB/GNU Octave
--------------------------------------------------------------------
Copyright (c) 2010,2011 Qianqian Fang <fangq at nmr.mgh.harvard.edu>
                      URL: http://mcx.sf.net
====================================================================

Format:
    [flux,detphoton]=mcxlab(cfg);

Input:
    cfg: a struct, or struct array. Each element of cfg defines 
         the parameters associated with a simulation. 

    It may contain the following fields:
     *cfg.nphoton:    the total number of photons to be simulated (integer)
     *cfg.vol:        a 3D array specifying the media index in the domain
     *cfg.prop:       an N by 4 array, each row specifies [mua, mus, n, g] in order.
                      the first row corresponds to medium type 0 which is 
                      typically [0 0 1 1]. The second row is type 1, and so on.
     *cfg.tstart:     starting time of the simulation (in seconds)
     *cfg.tstep:      time-gate width of the simulation (in seconds)
     *cfg.tend:       ending time of the simulation (in second)
     *cfg.srcpos:     a 1 by 3 vector, specifying the position of the source
     *cfg.srcdir:     a 1 by 3 vector, specifying the incident vector
      cfg.nblocksize: how many CUDA thread blocks to be used [64]
      cfg.nthread:    the total CUDA thread number [2048]
      cfg.session:    a string for output file names (used when no return variables)
      cfg.seed:       seed for the random number generator (integer) [0]
      cfg.maxdetphoton:   maximum number of photons saved by the detectors [1000000]
      cfg.detpos:     an N by 4 array, each row specifying a detector: [x,y,z,radius]
      cfg.detradius:  radius of the detector (in mm) [1.0]
      cfg.sradius:    radius within which we use atomic operations (in mm) [0.0]
      cfg.respin:     repeat simulation for the given time (integer) [1]
      cfg.gpuid:      which GPU to use (run 'mcx -L' to list all GPUs) [1]
      cfg.isreflect:  [1]-consider refractive index mismatch, 0-matched index
      cfg.isref3:     [1]-consider maximum 3 reflection interface; 0-only 2
      cfg.isnormalized:[1]-normalize the output flux to unitary source, 0-no reflection
      cfg.issavedet:  1-to save detected photon partial path length, [0]-do not save
      cfg.issave2pt:  [1]-to save flux distribution, 0-do not save
      cfg.isgpuinfo:  1-print GPU info, [0]-do not print
      cfg.autopilot:  1-automatically set threads and blocks, [0]-use nthread/nblocksize
      cfg.minenergy:  terminate photon when weight less than this level (float) [0.0]
      cfg.unitinmm:   defines the length unit for a grid edge length [1.0]

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
            is the ID(>0) of the detector that captures the photon; the second
	    row is the weight of the photon when it is detected; the rest rows
	    are the partial path lengths (in mm) traveling in medium 1 up to the last.

      if detphoton is ignored, the detected photon will be saved in an .mch file 
      if cfg.issavedeet=1; if no output is given, the flux will be saved to an 
      .mc2 file if cfg.issave2pt=1 (which is true by default).

Example:
      cfg.nphoton=1e7;
      cfg.vol=uint8(ones(60,60,60));
      cfg.srcpos=[30 30 1];
      cfg.srcdir=[0 0 1];
      cfg.gpuid=1;
      cfg.autopilot=1;
      cfg.prop=[0 0 1 1;0.005 1 1.37 0];
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
</pre>

== # How to compile ==

To compile MCXLAB for Matlab, you need to cd mcx/src directory, and type 
 make mex
from a shall window. You need to make sure your Matlab is installed and 
the command "mex" is included in your PATH environment variables. Similarly, 
to compile MCXLAB for Octave, you type
 make oct
The command <tt>mkoctfile</tt> must be accessible from your command line
and it is provided in a package named "octave3.x-headers" in Ubuntu (3.x
can be 3.2 or 3.4 etc).

