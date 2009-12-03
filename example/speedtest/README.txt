= README for the speed benchmark example =

In this example, we benchmark the speed of MCX code 
with different combinations of compilation options, 
number of photons atomic/non-atomic memory access.

The detailed report can be found in the paper Fang2009 
Section 3.4 and Fig. 7.

To run this example, you need to first compile the mcx
binary, and then run "runspeedbench.sh > speed.log" 
command. This may take up to 10 minutes to run all the 
tests. To make sure nothing goes wrong, it is probably
a better idea to run runspeedbench.sh without the 
redirection first, and make sure everything compiles
and runs properly.

When you finish the simulation, run "genreport.sh speed.log"
to get a tabulated report for all conditions.
From this, you can reproduce the data used for
Fig. 7 in Fang2009.

For some simulations with more threads, you may 
experience the "kernel launch timed-out" error.
Please read the doc/FAQ.txt to find out why.


[Fang2009]   Qianqian Fang and David A. Boas, "Monte Carlo 
   Simulation of Photon Migration in 3D Turbid Media Accelerated 
   by Graphics Processing Units," Opt. Express, 
   vol. 17, issue 22, pp. 20178-20190 (2009)
