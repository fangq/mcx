== The skin-vessel benchmark from mcxyz by Dr. Jacques ==

In this example, we compare between MCX and mcxyz written by Dr. Steve Jacques.
The same benchmark can be found at https://omlc.org/software/mc/mcxyz/index.html

By default, this simulation outputs fluence rate. To change the output to energy 
deposition, please run the script using

 ./run_mcxyz_bench.sh -O E

To plot the mcx solutions and compare with those from mcxyz, please use the
matlab script demo_mcxyz_skinvessel.m as part of mcxlab, see

https://github.com/fangq/mcx/blob/master/mcxlab/examples/demo_mcxyz_skinvessel.m

This example can be found in Fig. 2d in the below paper 

[Fang2020] Qianqian Fang* and Shijie Yan, “Graphics processing unit-accelerated 
mesh-based Monte Carlo photon transport simulations,” J. of Biomedical 
Optics, 24(11), 115002 (2019).

URL: https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-24/issue-11/115002/Graphics-processing-unit-accelerated-mesh-based-Monte-Carlo-photon-transport/10.1117/1.JBO.24.11.115002.full?SSO=1
