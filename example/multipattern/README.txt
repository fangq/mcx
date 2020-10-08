== Simultaneous forward simulations of multiple patterned sources (Photon Sharing) ==

This script demonstrates the "photon sharing" feature (Yao&Yan et al, 2019,
Photonics West) to simultaneously create forward solutions of multiple
patterned source.

In this example, we use 3 patterns and the data is stored in the multipattern_pattern.bin 
binary file stored as a 3x6x11 3D float (single-precision) array, where 3 is the pattern 
count, 6x11 is the dimension for each 2D pattern. To use this feature in the command line,
you must use a .json input file, like the one in this example, and set "SrcNum"  to an integer
greater than 1, and the "SrcType" to "pattern" or "pattern3d";

The output of the data in the .mc2 file is stored in a 5-D array: 3x60x60x60x1, where
3 is the pattern count, 60x60x60 is the 3D domain size, and 1 is the time-gate.

In other words, when running multiple patterns using "Photon Sharing", one must pack the
source pattern data so that the inner-most loop (fastest index) is the pattern; this will
produce a fluence output where the fastest index is also the number of patterns.

and then run it using

 ./run_multipattern.sh

or

 ./run_multipattern.sh -n 1e6 

to specify a different photon number

=== Reference ===
[Yan2020] Shijie Yan, Ruoyang Yao, Xavier Intes, and Qianqian Fang, "Accelerating 
Monte Carlo modeling of structured-light-based diffuse optical imaging via 
'photon sharing'," Opt. Lett. 45, 2842-2845 (2020)
URL: https://www.biorxiv.org/content/10.1101/2020.02.16.951590v2
