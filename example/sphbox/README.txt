= README for the sphbox example =

In this example, we validate MCX code with a heterogeneous
medium defined by a 10mm radius sphere embedded inside a 
60x60x60mm uniform grid. The background medium has mua=0.005/mm, 
musp=1/mm, anisotropy g=0.01 and refraction index n=1 (without 
reflection) or n=1.37 (with reflection); the sphere has mus=5/mm, 
g=0.9, mua=0.05/mm and n=1.37.

The detailed report can be found in the paper Fang2010 
Section 3.2 and Fig. 3.

To run this example, you need to first compile the mcx
binary. Once complete, you can run the simulation by

   mcx -A -n 3e7 -f spherebox.json -s sphbox -G 1

or

   runspherebox.sh

use the number following the -G option to specify which GPU you
want to use on your system. Once complete, you can use the 
plotresults.m script in matlab to plot the results. Please make
sure to edit the header of plotresults.m to add path to the matlab
folder in mesh-based Monte Carlo (MMC) package [1].



[Fang2010]  Qianqian Fang, "Mesh-based Monte Carlo method 
            using fast ray-tracing in Plücker coordinates," 
            Biomed. Opt. Express 1, 165-175 (2010)

[1] http://mcx.sourceforge.net/cgi-bin/index.cgi?MMC
