---------------------------------------------------------------------
                   Monte Carlo eXtreme  (MCX)
                          CUDA Edition
---------------------------------------------------------------------

Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
License: GNU General Public License version 3 (GPLv3)
Version: 0.7.9 (Charm Quarks - beta)

---------------------------------------------------------------------

Table of Content:

I.    Introduction
II.   Requirement and Installation
III.  Running Simulations
IV.   Using JSON-formatted input files
V.    Using JSON-formatted shape description files
VI.   Using MCXLAB in MATLAB and Octave
VII.  Using MCX Studio GUI
VIII. Interpreting the Outputs
IX.   Reference

---------------------------------------------------------------------

I.  Introduction

Monte Carlo eXtreme (MCX) is a fast photon transport simulation 
software for 3D heterogeneous turbid media. By taking advantage of 
the massively parallel threads and extremely low memory latency in a 
modern graphics processing unit (GPU), this program is able to perform Monte 
Carlo (MC) simulations at a blazing speed, typically hundreds to
a thousand times faster than a fully optimized CPU-based MC 
implementation.

The algorithm of this software is detailed in the Reference [1]. 
A short summary of the main features includes:

*. 3D heterogeneous media represented by voxelated array
*. boundary reflection support
*. time-resolved photon transport simulation
*. saving photon partial path lengths at the detectors
*. optimized random number generators
*. build-in flux/fluence normalization to output Green's functions
*. user adjustable voxel resolution
*. improved accuracy near the source with atomic operations
*. cross-platform graphics user interface
*. native Matlab/Octave support for high usability
*. flexible JSON interface for future extensions

This software can be used on Windows, Linux and Mac OS. 
MCX is written in CUDA and can be used with NVIDIA hardware
with the native NVIDIA drivers, or used with GPU ocelot open-source
libraries for CPUs and AMD GPUs. An OpenCL implementation of
MCX, i.e. MCX-CL, will be announced soon and can support 
NVIDIA/AMD/Intel hardware out-of-box.


---------------------------------------------------------------------------
II. Requirement and Installation

For MCX-CUDA, the requirements for using this software are

*. a CUDA capable NVIDIA graphics card
*. pre-installed CUDA driver [1] and NVIDIA graphics driver

If your hardware does not support CUDA, the installation of the CUDA 
toolkit will fail. A list of CUDA capable cards can be found at [2].
Generally speaking, GeForce 8XXX series or newer are required.
Using the latest NVIDIA card is expected to generate the best
speed (GTX 4xx is twice faster than 2xx, which is twice faster than 
8800/9800). For simulations with large volumes, sufficient video memory 
is also required to perform the simulation. The minimum amount of 
graphics memory required for a MC simulation is Nx*Ny*Nz*Ng
bytes for the input tissue data plus Nx*Ny*Nz*Ng*4 bytes for 
the output flux/fluence data - where Nx,Ny,Nz are the dimensions of the 
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
and a volume file (a binary file with each byte representing a medium 
index). Typing the name of the executable without any parameters, 
will print the help information and a list of supported parameters, 
such as the following:

<pre>
usage: mcx <param1> <param2> ...
where possible parameters include (the first item in [] is the default value)
 -i 	       (--interactive) interactive mode
 -s sessionid  (--session)     a string to label all output file names
 -f config     (--input)       read config from a file
 -n [0|int]    (--photon)      total photon number (exponential form accepted)
 -t [2048|int] (--thread)      total thread number
 -T [64|int]   (--blocksize)   thread number per block
 -A [0|int]    (--autopilot)   auto thread config:1 dedicated GPU;2 non-dedica.
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto
 -r [1|int]    (--repeat)      number of repetitions
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array
 -z [0|1]      (--srcfrom0)    1 volume coord. origin [0 0 0]; 0 use [1 1 1]
 -g [1|int]    (--gategroup)   number of time gates per run
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit
 -B [0|1]      (--reflectin)   1 to reflect photons at int. boundary; 0 do not
 -e [0.|float] (--minenergy)   minimum energy level to terminate a photon
 -R [0.|float] (--skipradius)  cached zone radius from source to use atomics
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw
 -d [1|0]      (--savedet)     1 to save photon info at detectors; 0 not save
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save
 -H [1000000]  (--maxdetphoton)max number of detected photons
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save
 -E [0|int]    (--seed)        set random-number-generator seed
 -h            (--help)        print this message
 -l            (--log)         print messages to a log file instead
 -L            (--listgpu)     print GPU information only
 -I            (--printgpu)    print GPU information and run program
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid
 -v            (--version)     print MCX revision number
example:
       mcx -A -n 1e7 -f input.inp -G 1 
or
       mcx -t 2048 -T 64 -n 1e7 -f input.inp -s test -r 2 -g 10 -U 0 -b 1 -G 1
or
       mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
</pre>

the above command will launch 2048 GPU threads (-t) with every 64 threads
a block (-T); a total of 1e7 photons will be simulated by the first GPU (-G 1) 
with two equally divided runs (-r); the media/source configuration will be 
read from input.inp (-f) and the output will be labeled with the session 
id "test" (-s); input media index array is in column-major format (-a); the 
simulation will run 10 concurrent time gates (-g). Photons passing through
the defined detector positions will be saved for later rescaling (-d).

Currently, MCX supports a modified version of the input file format used 
by tMCimg. (The difference is that MCX allows comments in the input file)
A typical MCX input file looks like this:

1000000              # total photon, use -n to overwrite in the command line
29012392             # RNG seed, negative to generate
30.0 30.0 0.0 1      # source position (in grid unit), the last num sets srcfrom0 (-z)
0 0 1                # initial directional vector
0.e+00 1.e-09 1.e-10 # time-gates(s): start, end, step
semi60x60x60.bin     # volume ('unsigned char' format)
1 60 1 60            # x voxel size in mm (isotropic only), dim, start/end indices
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices 
1 60 1 60            # y voxel size, must be same as x, dim, start/end indices
1                    # num of media
1.010101 0.01 0.005 1.37  # scat. mus (1/mm), g, mua (1/mm), n
4       1.0          # detector number and default radius (in grid unit)
30.0  20.0  0.0  2.0 # detector 1 position (real numbers in grid unit) and radius if different
30.0  40.0  0.0      # ..., if radius is ignored, MCX will use the default radius
20.0  30.0  0.0      #
40.0  30.0  0.0      # 

Note that the scattering coefficient mus=musp/(1-g).

The volume file (semi60x60x60.bin in the above example),
can be read in two ways by MCX: row-major[3] or column-major
depending on the value of the user parameter "-a". If the volume file
was saved using matlab or fortran, the byte order is column-major,
and you should use "-a 0" or leave it out of the command line. 
If it was saved using the fwrite() in C, the order is row-major, 
and you can either use "-a 1".

The time gate parameter is specified by three numbers:
start time, end time and time step size (in seconds). In 
the above example, the configuration specifies a total time 
window of [0 1] ns, with a 0.1 ns resolution. That means the 
total number of time gates is 10. 

MCX provides an advanced option, -g, to run simulations when 
the GPU memory is limited. It specifies how many time gates to simulate 
concurrently. Users may want to limit that number to less than 
the total number specified in the input file - and by default 
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
IV. Using JSON-formatted input files

Starting from version 0.7.9, MCX accepts a JSON-formatted input file in
addition to the conventional tMCimg-like input format. JSON 
(JavaScript Object Notation) is a portable, human-readable and 
"fat-free" text format to represent complex and hierarchical data.
Using the JSON format makes a input file self-explanatory, extensible
and easy-to-interface with other applications (like MATLAB).

A sample JSON input file can be found under the examples/quicktest
folder. The same file, qtest.json, is also shown below:

 {
    "Help": {
      "[en]": {
        "Domain::VolumeFile": "file full path to the volume file, file mst be in the uchar binary format",
        "Domain::Dim": "dimension of the data array stored in the volume file",
        "Domain::OriginType": "similar to --srcfrom0, 1 if the origin is [0 0 0], 0 if it is [1.0,1.0,1.0]",
        "Domain::Step": "do not change this, should be always be 1",
        "Domain::CacheBoxP0": "for cachebox mcx with -R negative_num, this specifies a 3D index for 
                               a corner of the cache region, in grid unit",
        "Domain::CacheBoxP1": "the other corner, the starting value of the indices is 1",
        "Domain::Media": "the first medium is always assigned to voxels with a value of 0 or outside of
                         the volume, the second row is for medium type 1, and so on. mua and mus must 
                         be in 1/mm unit",
        "Session::Photons": "if -n is not specified in the command line, this defines the total photon number",
        "Session::ID": "if -s is not specified in the command line, this defines the output file name stub",
        "Forward::T0": "the start time of the simulation, in seconds",
        "Forward::T1": "the end time of the simulation, in seconds",
        "Forward::Dt": "the width of each time window, in seconds",
        "Optode::Source::Pos": "the grid position of the source, can be non-integers, in grid unit",
        "Optode::Detector::Pos": "the grid position of a detector, can be non-integers, in grid unit"
      }
    },
    "Domain": {
	"VolumeFile": "semi60x60x60.bin",
        "Dim":    [60,60,60],
        "OriginType": 1,
        "Step":   [1.0,1.0,1.0],
        "CacheBoxP0": [24,24,1],
        "CacheBoxP1": [34,34,10],
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
	    "Dir": [0.0, 0.0, 1.0]
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
recommended. In the alternative format, you can use 
 "rootobj_name.field_name": value 
to represent any parameter directly in the root level. For example

 {
    "Domain.VolumeFile": "semi60x60x60.bin",
    "Session.Photons": 10000000,
    ...
 }

You can even mix the alternative format with the standard format. 
If any input parameter has values in both formats in a single input 
file, the standard-formatted value has higher priority.

To invoke the JSON-formatted input file in your simulations, you 
can use the "-f" command line option with MCX, just like using an 
.inp file. For example:

  mcx -A -n 20 -f onecube.json -s onecubejson

The input file must have a ".json" suffix in order for MCX to 
recognize. If the input information is set in both command line,
and input file, the command line value has higher priority
(this is the same for .inp input files). For example, when 
using "-n 20", the value set in "Session"/"Photons" is overwritten 
to 20; when using "-s onecubejson", the "Session"/"ID" value is modified.
If your JSON input file is invalid, MCX will quit and point out
where the format is incorrect.

---------------------------------------------------------------------------
V. Using JSON-formatted shape description files

Starting from v0.7.9, MCX can also use a shape 
description file in the place of the volume file.
Using a shape-description file can save you from making
a binary .bin volume. A shape file uses more descriptive 
syntax and can be easily understood and shared with others.

Samples on how to use the shape files are included under
the example/shapetest folder. 

The sample shape file, shapes.json, is shown below:

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
put the file name in Line#6 of yoru .inp file, or set as the
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
VI. Using MCXLAB in MATLAB and Octave

MCXLAB is the native MEX version of MCX for Matlab and GNU Octave. It compiles
the entire MCX code into a MEX function which can be called directly inside
Matlab or Octave. The input and output files in MCX are replaced by convenient
in-memory struct variables in MCXLAB, thus, making it much easier to use
and interact. Matlab/Octave also provides convenient plotting and data
analysis functions. With MCXLAB, your analysis can be streamlined and speed-
up without involving disk files.

Please read the mcxlab/README.txt file for more details on how to
install and use MCXLAB.


---------------------------------------------------------------------------
VII. Using MCX Studio GUI

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
* have you installed the CUDA/NVIDIA drivers correctly?
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


---------------------------------------------------------------------------
VIII. Interpreting the Output

MCX output consists of two parts, the flux volume 
file and messages printed on the screen.

8.1 Output files

An mc2 file contains the flux distribution from the simulation in 
the given medium. By default, this flux is a normalized solution 
(as opposed to the raw probability) therefore, one can compare this directly 
to the analytical solutions (i.e. Green's function). The order of storage in the 
mc2 files is the same as the input file: i.e., if the input is row-major, the 
output is row-major, and so on. The dimensions of the file are Nx, Ny, Nz, and Ng
where Ng is the total number of time gates.

By default, MCX produces the '''Green's function''' of the 
'''fluence rate''' (or '''flux''') for the given domain and 
source. Sometime it is also known as the time-domain "two-point" 
function. If you run MCX with the following command

  mcx -f input.inp -s output ....

the flux data will be saved in a file named "output.dat" under
the current folder. If you run MCX without "-s output", the
output file will be named as "input.inp.dat".

To understand this further, you need to know that a '''flux''' is
measured by number of particles passing through an infinitesimal 
spherical surface per <em>unit time</em> at <em>a given location</em>.
The unit of MCX output flux is "1/(mm<sup>2</sup>s)", if the flux is interpreted as the 
"particle flux" [6], or "J/(mm<sup>2</sup>s)", if it is interpreted as the 
"energy flux" [6].

The Green's function of the flux simply means that the flux is produced
by a '''unitary source'''. In simple terms, this represents the 
fraction of particles/energy that arrives a location per second 
under <em>the radiation of 1 unit (packet or J) of particle or energy 
at time t=0</em>. The Green's function is calculated by a process referred
to as the "normalization" in the MCX code and is detailed in the 
MCX paper [6] (MCX and MMC outputs share the same meanings).

Please be aware that the output flux is calculated at each time-window 
defined in the input file. For example, if you type 

 0.e+00 5.e-09 1e-10  # time-gates(s): start, end, step

in the 5th row in the input file, MCX will produce 50 flux
distributions, corresponding to the time-windows at [0 0.1] ns, 
[0.1 0.2]ns ... and [4.9,5.0] ns. To convert the flux distributions
to the fluence distributions for each time-window, you just need to
multiply each solution by the width of the window, 0.1 ns in this case. To convert the time-domain flux
to the continuous-wave (CW) fluence, you need to integrate the
flux in t=[0,inf]. Assuming the flux after 5 ns is negligible, then the CW
fluence is simply sum(flux_i*0.1 ns, i=1,50). You can read 
<tt>mcx/examples/validation/plotsimudata.m</tt>
and <tt>mcx/examples/sphbox/plotresults.m</tt> for examples 
to compare an MCX output with the analytical flux/fluence solutions.

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
http://mcx.sf.net/cgi-bin/index.cgi?MMC/Doc/FAQ#How_do_I_interpret_MMC_s_output_data


8.2 Console print messages

Timing information is printed on the screen (stdout). The 
clock starts (at time T0) right before the initialization data is copied 
from CPU to GPU. For each simulation, the elapsed time from T0
is printed (in ms). Also the accumulated elapsed time is printed for 
all memory transaction from GPU to CPU.

---------------------------------------------------------------------------
IX. Reference

[1] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

If you use MCX in your research, the author of this software would like
you to cite the above paper in your related publications.

Links: 

[1] http://developer.nvidia.com/cuda-downloads
[2] http://www.nvidia.com/object/cuda_gpus.html
[3] http://en.wikipedia.org/wiki/Row-major_order
[4] http://iso2mesh.sourceforge.net/cgi-bin/index.cgi?jsonlab
[5] http://science.jrank.org/pages/60024/particle-fluence.html
[6] http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-17-22-20178
