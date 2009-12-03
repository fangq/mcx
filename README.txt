---------------------------------------------------------------------
                   Monte Carlo eXtreme  (MCX)
                          CUDA Edition
---------------------------------------------------------------------

Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
License: GNU General Public License version 3 (GPLv3)
Version: 0.2 (Aurora)

---------------------------------------------------------------------

Table of Content:

I.  Introduction
II. Requirement and Installation
III.Running Simulations
IV. Interpret the Outputs
V.  Reference

---------------------------------------------------------------------

I.  Introduction

Monte Carlo eXtreme (MCX) is a simulation software for modeling photon
propagation in 3D turbid media. By taking advantage of the massively
parallel threads and extremely low memory latency, this program is able
to perform Monte Carlo (MC) simulations at a blazing speed on a 
low-cost graphics card, typically 100 to 300 times faster than a fully 
optimized CPU-based MC code.

The algorithm details of this software can be found in the Reference [1]. 
A short summary of the main features includes:

*. 3D arbitrary media
*. boundary reflection support
*. time-resolved photon migration
*. optimized random number generators
*. build-in fluence normalization
*. fine-tuned performance for high computational efficiency

The software can be used on Windows, Linux and Mac OS. Two variants 
will be provided, one for nVidia(TM) graphics hardware written in CUDA, 
and one for ATI(TM) written in Brook+ (the ATI version is currently
work-in-progress).



II. Requirement and Installation

For MCX-CUDA, the requirements for using this software include

*. a CUDA capable nVidia graphics card
*. pre-installed CUDA driver [1] and nVidia graphics driver

If your hardware does not support CUDA, the installation of CUDA toolkit 
will fail. A list of CUDA capable cards can be found at [2].
Generally speaking, GeForce 8XXX series or newer are required.
For simulations with large domain, sufficient video memory is
also required to perform the simulation. The minimally required 
graphics memory is roughly Nx*Ny*Nz*(4+1)/1024/1024 MB where 
Nx/Ny/Nz is the dimension of the problem domain, 4 for a floating-point 
number to save the output fluence, 1 for input media index. If your 
simulation involves multiple time-gates, you need to specify a maximum 
time-gate number (Ng) for one single run (if your total time-gates 
is greater than Ng, it will be split into multiple groups), and the 
total required memory need to be multiplied by Ng.

The single-precision computation appears to be sufficient for this 
application and is well supported by the stock graphics hardware. 
So, MCX does not require double-precision support in your hardware.

To install MCX, you simply download the binary corresponding to your 
computer architecture (32 or 64bit) and platform, extract the package and
run the executable under the <mcx root>/bin directory. For Linux
and Mac OS users, you need to add the following lines to your
shell initialization scripts. First, use "echo $SHELL" command to 
identify your shell type. For csh/tcsh, add the following lines 
to your ~/.cshrc file:

  if ("`uname -p`" =~ "*_64" ) then
	  setenv LD_LIBRARY_PATH "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  else
	  setenv LD_LIBRARY_PATH "/usr/local/cuda/lib:$LD_LIBRARY_PATH"
  endif
  setenv PATH "/usr/local/cuda/bin:$PATH"

and for bash/sh users, add 

  if [[ "`uname -p`" =~ .*_64 ]]; then
	  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  else
	  export LD_LIBRARY_PATH="/usr/local/cuda/lib:$LD_LIBRARY_PATH"
  fi
  export PATH="/usr/local/cuda/bin:$PATH"

to your ~/.bash_profile.

Make sure the path "/usr/local/cuda/lib*" exists on your system.
If your CUDA library is not installed under this directory, 
please replace it to the actual path where you can find libcudart.*


III.Running Simulations

To run a simulation, the minimally required input is an input file,
and a volume (a binary file with each byte representing a medium 
index). If you do not know the format to supply command line 
options to MCX, simply type the name of the executable without 
any parameters, it will print a list of supported parameters, 
such as the following:

 usage: ./mcx <param1> <param2> ...
 where possible parameters include (the first item in [] is the default value)
 -i            (--interactive) interactive mode
 -f config     (--input)       read config from a file
 -t [1024|int] (--thread)      total thread number
 -T [128|int]  (--blocksize)   thread number per block
 -m [0|int]    (--move)        total photon moves
 -n [0|int]    (--photon)      total photon number (not supported yet, use -m only)
 -r [1|int]    (--repeat)      number of repetitions
 -a [1|0]      (--array)       1 for C array, 0 for Matlab array
 -g [1|int]    (--gategroup)   number of time gates per run
 -b [1|0]      (--reflect)     1 to reflect the photons at the boundary, 0 to exit
 -B [0|1]      (--reflect3)    1 to consider maximum 3 reflections, 0 consider only 2
 -e [0.|float] (--minenergy)   minimum energy level to propagate a photon
 -R [0.|float] (--skipradius)  minimum distance to source to start accumulation
 -U [1|0]      (--normalize)   1 to normalize the fluence to unitary, 0 to save raw fluence
 -d [1|0]      (--savedet)     1 to save photon info at detectors, 0 not to save
 -S [1|0]      (--save2pt)     1 to save the fluence field, 0 do not save
 -s sessionid  (--session)     a string to identify this specific simulation (and output files)
 -p [0|int]    (--printlen)    number of threads to print (debug)
 -h            (--help)        print this message
 -l            (--log)         print messages to a log file instead
 -L            (--listgpu)     print GPU information only
 -I            (--printgpu)    print GPU information and run program
 example:
       ./mcx -t 1024 -T 256 -m 1000000 -f input.inp -s test -r 2 -a 0 -g 10 -U 0


Currently, MCX supports an extended input file format for tMCimg,
A typical MCX input file looks like

1000000              # total photon (not used), use -m to specify photon moves
29012392             # RNG seed, negative to generate
30.0 30.0 1.0        # source position (mm)
0 0 1                # initial directional vector
0.e+00 1.e-09 1.e-10 # time-gates(s): start, end, step
semi60x60x60.bin     # volume ('unsigned char' format)
1 60 1 60            # x: voxel size, dim, start/end indices
1 60 1 60            # y: voxel size, dim, start/end indices 
1 60 1 60            # z: voxel size, dim, start/end indices
1                    # num of media
1.010101 0.01 0.005 1.37  # scat(1/mm), g, mua (1/mm), n
4       1            # detector number and radius (mm)
30.0    20.0    1.0  # detector 1 position (mm)
30.0    40.0    1.0  # ...
20.0    30.0    1.0
40.0    30.0    1.0

Notice that the scattering coefficient mus=musp/(1-g).
For the volume file, semi60x60x60.bin in the above example,
if you save the data using matlab/fortran, the byte order
is column-major[3], i.e. data traversing column direction 
first. You have to use "-a 0" in MCX's command line to 
specify this order. If you save your data with fwrite() in C, 
the order is row-major, and you should use "-a 1" or omit.

The time-gate settings are specified by three numbers,
the start, end and step (in seconds). In the above case,
the simulation specify a window of [0 1] ns, with 0.1 ns
resolution. That means the total time gates is 10. Based 
on the memory requirement in Section II, if one wants to
simulate all 10 time gates in a single call, you have to 
make sure the video memory is sufficient, in this case,
the needed memory is 60*60*60*10*5=10M, which is quite small.
You can use "-g 10" option to simulate all 10 time-gates
in one run. If you do not specify -g option, MCX will 
assume you simulate 1 time gate per run, and run the 
simulation 10 times for all the time-gates. If you specify
a time-gate number more than needed, for example, "-g 20",
MCX will stop when the 10 needed time-gates are complete.


IV. Interpret the Outputs

MCX's outputs include two parts, the fluence volume 
file and messages printed on the screen.

4.1 Output files

An mc2 file is a binary file to save the fluence distribution
of the problem domain. By default, this fluence is a normalized
solution (oppose to the raw probability), therefore, one can
compare this directly to the analytical solutions (i.e. Green's 
function). The storing order in the mc2 files is the same as
the input: if the input is row-major, the output is row-major,
and so on. The dimension of the file is [Nx Ny Nz Ng] where Ng
is the total number of time gates.

One can load a mc2 output file into Matlab or Octave using the
loadmc2 function in <mcx root>/utils. If one wants to get a 
continuous-wave solution, he should run simulation with sufficiently
long time window, and sum the fluence along the time dimension, 
for example

   mcx=loadmc2('output.mc2',[60 60 60 10],'float');
   cw_mcx=sum(mcx,4);

Note that for time-resolved simulations, the corresponding solution
in the results approximates the fluence at the center point
of each time window. For example, if the simulation time window setting
is [t0,t0+dt,t0+2dt,t0+3dt...,t1], the time points for the 
snapshots stored in the solution file is located at 
[t0+dt/2, t0+3*dt/2, t0+5*dt/2, ... ,t1-dt/2]


4.2 On-screen messages

The timing information is printed on the screen (stdout). The 
clock starts (T0) right before copying the initialization data 
from CPU to GPU. For each simulation, the elapse time from T0
is printed (in ms). If there is a memory transaction from GPU to CPU,
the accumulated elapse time is also printed. Depending on domain 
size, typically the data transfer takes about 50 ms per run.

By default, MCX calculates the unitary solution for fluence; 
the computed normalization factor, see Reference [1], will be 
printed on the screen. At the end of the simulation, the data
will be saved to a file; this may take a long time depending on 
the domain size.

At the end of the screen output, one can find a list of photon
information, something like:

   0[A-0.320966  0.693810 -0.644674]C9586 J   15 W 0.943725(P25.869 30.965  1.877)T 3.864e-11 L 1.265 3035790080
   1[A-0.121211 -0.151523  0.980989]C9682 J  184 W 0.408108(P13.841 33.778 25.937)T 5.979e-10 L-9999.000 172048448
   ......
   simulated 9996602 photons with 1024 threads and 795590 moves per threads (repeat x1)
   exit energy:  8.34534150e+06 + absorbed energy:  1.65226412e+06 = total:   9.99760600e+06

This output reflects the final states for each simulation thread (for each
thread, there is only one active photon). The fields can be interpreted as follow

0: thread id
[A-0.320966  0.693810 -0.644674]: direction vector
C9586: completed photons for this thread
J   15: number of jumps (scattering events)
W 0.943725: current photon packet weight
(P25.869 30.965  1.877): current photon position
T 3.864e-11: accumulative propagation time
L 1.265: remaining scattering length for the current jump
3035790080: the random number state

The above thread info is for debugging purposes. The total 
number of threads to be printed can be specified using the
"-p" option. 



V. Reference

[1] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

If you used MCX in your research, the authors of this software would like
you to cite the above paper in your related publications.

Links: 

[1] http://www.nvidia.com/object/cuda_get.html
[2] http://www.nvidia.com/object/cuda_learn_products.html
[3] http://en.wikipedia.org/wiki/Row-major_order
