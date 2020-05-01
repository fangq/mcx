== Digimouse Atlas Photon Simulation ==

In this example, we demonstrate light transport simulation in a mouse
atlas template (Digimouse). There are 21 tissue types in the atlas. The volume
is made of 190 x 496 x 104 0.8 mm^3 isotropic voxels. See [Fang2012].

To run this example, please call

 ./run_atlas.sh

or

 ./run_atlas.sh -n 1e6 

to specify a different photon number


The JSON files (.json, .jnii) utlizes the JData specifiation (https://github.com/fangq/jdata) 
to include binary data with compression support. Please download JSONLab from

https://github.com/fangq/jsonlab

to open these files in MATLAB and GNU Octave, or PyJData from 

https://github.com/fangq/pyjdata

to open such in Python.


=== Reference ===

[Fang2012] Fang Q and Kaeli D, "Accelerating mesh-based Monte Carlo method 
 on modern CPU architectures," Biomed. Opt. Express, 3(12), 3223-3230, 2012

