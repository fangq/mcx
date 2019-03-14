== Colin27 Brain Atlas Photon Simulations ==

In this example, we demonstrate light transport simulation in a full-head 
atlas template (Colin27). There are 7 tissue types:

0: background (air)
1: scalp
2: skull
3: CSF
4: gray matter
5: white matter
6: air cavities in the brain

To run the simulation, you must first unzip the domain binary file using 
unlzma on Linux or 7-zip on Windows. For example, on Linux:

 unlzma colin27_v3.bin.lzma

This demo is identical to the MCX simulation used for Fig.6 in
the original MCX paper [Fang2009].


 [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation
  of photon migration in 3D turbid media accelerated by graphics processing
  units," Opt. Express 17, 20178-20190 (2009)

