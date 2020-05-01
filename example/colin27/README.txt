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


This demo is identical to the MCX simulation used for Fig.6 in
the original MCX paper [Fang2009].

This example is a built-in dataset in MCX. Run "mcx --bench" to show
the full list of built-in examples.

The JSON files (.json, .jnii) utlizes the JData specifiation (https://github.com/fangq/jdata) 
to include binary data with compression support. Please download JSONLab from

https://github.com/fangq/jsonlab

to open these files in MATLAB and GNU Octave, or PyJData from 

https://github.com/fangq/pyjdata

to open such in Python.


 [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation
  of photon migration in 3D turbid media accelerated by graphics processing
  units," Opt. Express 17, 20178-20190 (2009)

