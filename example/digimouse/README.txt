== Digimouse Atlas Photon Simulation ==

In this example, we demonstrate light transport simulation in a mouse
atlas template (Digimouse). There are 21 tissue types in the atlas. The volume
is made of 190 x 496 x 104 0.8 mm^3 isotropic voxels. See [Fang2012].

To run the simulation, you must first unzip the domain binary file using 
unlzma on Linux or 7-zip on Windows. For example, on Linux:

 unlzma digimouse_0.8mm.bin.lzma

and then run it using

 ./run_atlas.sh

or

 ./run_atlas.sh -n 1e6 

to specify a different photon number


=== Reference ===

[Fang2012] Fang Q and Kaeli D, "Accelerating mesh-based Monte Carlo method 
 on modern CPU architectures," Biomed. Opt. Express, 3(12), 3223-3230, 2012

