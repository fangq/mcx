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
IV. Using MCX Studio GUI
V.  Interpreting the Outputs
VI. Reference

---------------------------------------------------------------------

I.  Introduction

Monte Carlo eXtreme (MCX) is simulation software for modeling photon
propagation in 3D turbid media. By taking advantage of the multi-core
processors and extremely low memory latency of a low-cost 
graphics card, this program is able to perform Monte Carlo (MC) 
simulations at a blazing speed, typically over 300 times faster than 
fully optimized CPU-based MC code.

The algorithm details of this software can be found in the Reference [1]. 
A short summary of the main features includes:

*. 3D arbitrary media
*. boundary reflection support
*. time-resolved photon migration
*. optimized random number generators
*. build-in fluence normalization
*. cross-platform graphics user interface

The software can be used on Windows, Linux and Mac OS. Two variants 
will be provided, one for nVidia(TM) graphics hardware written in CUDA, 
and one for ATI(TM) written in Brook+ (the ATI version is currently a
work-in-progress).



II. Requirement and Installation

For MCX-CUDA, the requirements for using this software are

*. a CUDA capable nVidia graphics card
*. pre-installed CUDA driver [1] and nVidia graphics driver

If your hardware does not support CUDA, the installation of the CUDA 
toolkit will fail. A list of CUDA capable cards can be found at [2].
Generally speaking, GeForce 8XXX series or newer are required.
For simulations on large volumes, sufficient video memory is
also required to perform the simulation. The minimum amount of 
graphics memory required for a MC simulation is Nx*Ny*Nz*Ng
bytes for the input tissue data plus Nx*Ny*Nz*Ng*4 bytes for 
the output fluence data - where Nx,Ny,Nz are the dimensions of the 
tissue volume, Ng is the number of concurrent time gates, 4 is 
the size of a single-precision floating-point number.
MCX does not require double-precision support in your hardware.

To install MCX, you simply download the binary executable corresponding to your 
computer architecture (32 or 64bit) and platform, extract the package 
and run the executable under the <mcx root>/bin directory. For Linux
and MacOS users, you need to add the following lines to your
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

If the path "/usr/local/cuda/lib*" does not exist on your system or
the CUDA library is not installed under this directory, then 
substitute the actual path under which libcudart.* exists.


III.Running Simulations

To run a simulation, the minimum input is a configuration (text) file,
and a volume (a binary file with each byte representing a medium 
index). Typing the name of the executable without any parameters, 
will print the help information and a list of supported parameters, 
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
       ./mcx -t 1024 -T 256 -m 1000000 -f input.inp -s test -r 2 -a 0 -g 10

 the above command will launch 1024 GPU threads (-t) with every 256 threads
 a block (-T); for each thread, it will simulate 1e6 photon moves (-m) and
 repeat twice (-r); the media/source configuration will be read from 
 input.inp (-f) and the output will be labeled with the session id "test"; 
 input media index array is in column-major format (-a); the simulation will
 run 10 concurrent time gates (-g).

Currently, MCX supports a modified version of the input file format used 
for tMCimg. (The difference is that MCX allows comments)
A typical MCX input file looks like this:

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
1.010101 0.01 0.005 1.37  # scat. mus (1/mm), g, mua (1/mm), n
4       1            # detector number and radius (mm)
30.0    20.0    1.0  # detector 1 position (mm)
30.0    40.0    1.0  # ...
20.0    30.0    1.0
40.0    30.0    1.0

Note that the scattering coefficient mus=musp/(1-g).

The volume file (semi60x60x60.bin in the above example),
can be read in two ways by MCX: row-major[3] or column-major
depending on the value of the user parameter "-a". If the volume file
was saved using matlab or fortran, the byte order is column-major,
and you should use "-a 0". If it was saved using the fwrite() in C, 
the order is row-major, and you can either use "-a 1" or leave it 
out of the command line.

The time gate parameter is specified by three numbers:
start time, end time and time step size (in seconds). In 
the above example, the configuration specifies a total time 
window of [0 1] ns, with a 0.1 ns resolution. That means the 
total number of time gates is 10. 

MCX provides the user with an advanced option, -g, to increase 
performance by specifying how many time gates to simulate 
concurrently. If GPU memory is limited then the user may want to limit 
that number to less than the total number specified in the input file - and 
by default it runs one gate at a time in a single simulation. But if there's 
enough memory based on the memory requirement in Section II, you can 
simulate all 10 time gates (from the above example) concurrently by using 
"-g 10" in which case you have to make sure the video card has at least  
60*60*60*10*5=10MB of free memory.   If you do not include the -g, 
MCX will assume you want to simulate just 1 time gate at a time.. 
If you specify a time-gate number greater than the total number in the 
input file, (e.g, "-g 20") MCX will stop when the 10 time-gates are 
completed.


IV. Using MCX Studio GUI

MCX Studio is a graphics user interface (GUI) for MCX. It gives users
a straightforward way to set the command line options and simulation
parameters. It also allows users to create different simulation tasks 
and organize them into a project and save for later use.
MCX Studio can be run on many platforms such as Windows,
GNU Linux and Mac OS.

To use MCX Studio, it is suggested to put the mcxstudio binary
in the same directory as the mcx command; alternatively, you can
also add the path to mcx command to your PATH environment variable.

When launching MCX Studio, it will automatically check if mcx
binary is in the search path, if so, the "GPU" button in the 
toolbar will be enabled. It is suggested to click on this button
once, and see if you can see a list of GPUs and their parameters 
printed in the output field at the bottom part of the window. 
If you are able to see this information, your system is ready
to run MCX simulations. If you get error messages or not able
to see any usable GPU, please check the following:

* are you running MCX Studio/MCX on a computer with a supported card?
* have you installed the CUDA/nVidia drivers correctly?
* did you put mcx in the same folder as mcxstudio or add its path to PATH?

If your system is properly configured, you can now add new simulations 
by clicking the "New" button. MCX Studio will ask you to give a session
id string for this task. Then you should be able to adjust the parameters
based on your needs. Once you finish the adjustment, you should click the 
"Verify" button to see if there are obvious mistakes. If everything is
fine, the "Run" button will be activated. Click on it once will start your
simulation. If you want to abort the current simulation, you can click
the "Stop" button.

You can create multiple tasks with MCX Studio by hitting the "New"
button multiple times. The information for all of the sessions can
be saved as a project file (with .mcxp extension) by clicking the
"Save" button. You can load a previously saved project file back
to MCX Studio by clicking the "Load" button.


V. Interpreting Output

MCX output consists of two parts, the fluence volume 
file and messages printed on the screen.

5.1 Output files

An mc2 file contains the fluence distribution from the simulation in 
the given medium. By default, this fluence is a normalized solution 
(as opposed to the raw probability) therefore, one can compare this directly 
to the analytical solutions (i.e. Green's function). The order of storage in the 
mc2 files is the same as the input file: i.e., if the input is row-major, the 
output is row-major, and so on. The dimensions of the file are Nx, Ny, Nz, and Ng
where Ng is the total number of time gates.

One can load a mc2 output file into Matlab or Octave using the
loadmc2 function in <mcx root>/utils. 

To get a continuous-wave solution, run a simulation with a sufficiently 
long time window, and sum the fluence along the time dimension, for 
example

   mcx=loadmc2('output.mc2',[60 60 60 10],'float');
   cw_mcx=sum(mcx,4);

Note that for time-resolved simulations, the corresponding solution
in the results approximates the fluence at the center point
of each time window. For example, if the simulation time window 
setting is [t0,t0+dt,t0+2dt,t0+3dt...,t1], the time points for the 
snapshots stored in the solution file is located at 
[t0+dt/2, t0+3*dt/2, t0+5*dt/2, ... ,t1-dt/2]


5.2 Console Print messages

Timing information is printed on the screen (stdout). The 
clock starts (at time T0) right before the initialization data is copied 
from CPU to GPU. For each simulation, the elapsed time from T0
is printed (in ms). Also the accumulated elapsed time is printed for 
all memory transaction from GPU to CPU. Depending on the domain 
size, typically the data transfer takes about 50 ms per run.

By default, MCX calculates the unitary-source solution for fluence; 
the normalization factor (see Reference [1]) used to produce this 
solution is printed on the screen at the end of the simulation, and 
the fluence distribution saved to a file.

At the end of the screen output, statistical  information about the 
photons is displayed. Here's sample output:

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



VI. Reference

[1] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

If you used MCX in your research, the authors of this software would like
you to cite the above paper in your related publications.

Links: 

[1] http://www.nvidia.com/object/cuda_get.html
[2] http://www.nvidia.com/object/cuda_learn_products.html
[3] http://en.wikipedia.org/wiki/Row-major_order
