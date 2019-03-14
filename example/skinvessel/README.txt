== The skin-vessel benchmark from mcxyz by Dr. Jacques ==

In this example, we compare between MCX and mcxyz written by Dr. Steve Jacques.
The same benchmark can be found at https://omlc.org/software/mc/mcxyz/index.html

By default, this simulation outputs fluence rate. To change the output to energy 
deposition, please run the script using

 ./run_mcxyz_bench.sh -O E

To plot the mcx solutions and compare with those from mcxyz, please use the
matlab script demo_mcxyz_skinvessel.m as part of mcxlab, see

https://github.com/fangq/mcx/blob/master/mcxlab/examples/demo_mcxyz_skinvessel.m
