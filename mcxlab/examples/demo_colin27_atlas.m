% == Colin27 Brain Atlas Photon Simulations ==
% 
% In this example, we demonstrate light transport simulation in a full-head 
% atlas template (Colin27). There are 7 tissue types:
% 
% 0: background (air)
% 1: scalp
% 2: skull
% 3: CSF
% 4: gray matter
% 5: white matter
% 6: air cavities in the brain
% 
% To run the simulation, you must first unzip the domain binary file using 
% unlzma on Linux or 7-zip on Windows. For example, on Linux:
% 
%  unlzma colin27_v3.bin.lzma
% 
% This demo is identical to the MCX simulation used for Fig.6 in
% the original MCX paper [Fang2009].
% 
% 
%  [Fang2009] Qianqian Fang and David A. Boas, "Monte Carlo simulation
%   of photon migration in 3D turbid media accelerated by graphics processing
%   units," Opt. Express 17, 20178-20190 (2009)

clear cfg

load colin27_v3.mat
cfg.vol=colin27;

cfg.tstart=0;
cfg.tend=5e-09;
cfg.tstep=2e-10;

cfg.srcpos=[75 67.38 167.5];
cfg.srcdir=[0.1636 0.4569 -0.8743];
cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);

cfg.detpos=[   75.0000   77.1900  170.3000    3.0000
   75.0000   89.0000  171.6000    3.0000
   75.0000   97.6700  172.4000    3.0000
   75.0000  102.4000  172.0000    3.0000];

cfg.issrcfrom0=1;

cfg.prop=[         0         0    1.0000    1.0000 % background/air
    0.0190    7.8182    0.8900    1.3700 % scalp
    0.0190    7.8182    0.8900    1.3700 % skull
    0.0040    0.0090    0.8900    1.3700 % csf
    0.0200    9.0000    0.8900    1.3700 % gray matters
    0.0800   40.9000    0.8400    1.3700 % white matters
         0         0    1.0000    1.0000]; % air pockets

cfg.seed=29012392;
cfg.nphoton=10000000;
cfg.issaveexit=1;

[flue,detps]=mcxlab(cfg);
mcxplotvol(log10(flue.data));
figure;
plotmesh(detps.p,'r.')