= README for the quicktest example =

As the name indicates, this example runs a short simulation
for the purpose of testing the command line options build
with MCX. 

To run this example, you need to first compile the mcx binary, or
download a pre-compiled mcx package, so that ../../bin/mcx exists.
Then you can choose to run the following shell scripts:

listgpu.sh
    A script to call mcx to print the GPU information 
    only. Run this script and find out how many GPUs you 
    have in your graphics card and the related parameters 
    (memory, cores etc) 

run_qtest.sh
    This will run MCX for a simple homogeneous medium as 
    the example used in the Fang2009 paper. It launches 
    1e7 photons and prints a progress bar; the fluence is 
    saved in qtest.jnii.

run_grid2x.sh/run_grid3x.sh
    These scripts do the same thing as run_qtest.sh, 
    except that they use JSON input files and a voxel size
    of 0.5 mm and 1/3 mm, respectively.

run_widefield.sh
    This script runs a simulation with a wide-field source
    defined in qtest_widefield.inp.


[Fang2009]   Qianqian Fang and David A. Boas, "Monte Carlo 
   Simulation of Photon Migration in 3D Turbid Media Accelerated 
   by Graphics Processing Units," Opt. Express, 
   vol. 17, issue 22, pp. 20178-20190 (2009)
