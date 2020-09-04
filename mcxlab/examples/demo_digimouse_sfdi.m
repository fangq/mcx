%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we demonstrate light transport simulation in a digital
% mouse atlas - Digimouse
%
% This demo is similar to the MCX simulation used for Fig. 2 in
% [Fang2012], except this uses a voxelated model instead of a mesh.
% 
%
% [Fang2012] Qianqian Fang and David R. Kaeli, "Accelerating mesh-based
% Monte Carlo method on modern CPU architectures ," Biomed. Opt. Express
% 3(12), 3223-3230 (2012)  
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg;

load digimouse.mat
cfg.vol=digimouse;

cfg.prop=[1 0.0191 6.6 0.9 1.37
2 0.0136 8.6 0.9 1.37
3 0.0026 0.01 0.9 1.37
4 0.0186 11.1 0.9 1.37
5 0.0186 11.1 0.9 1.37
6 0.0186 11.1 0.9 1.37
7 0.0186 11.1 0.9 1.37
8 0.0186 11.1 0.9 1.37
9 0.0240 8.9  0.9 1.37
10 0.0026 0.01 0.9 1.37
11 0.0240 8.9  0.9 1.37
12 0.0240 8.9  0.9 1.37
13 0.0240 8.9  0.9 1.37
14 0.0240 8.9  0.9 1.37
15 0.0240 8.9  0.9 1.37
16 0.072  5.6 0.9 1.37
17 0.072  5.6 0.9 1.37
18 0.072  5.6 0.9 1.37
19 0.050  5.4 0.9 1.37
20 0.024  8.9 0.9 1.37
21 0.076 10.9 0.9 1.37];

cfg.prop(:,1)=[];
cfg.prop(2:end+1,:)=cfg.prop;
cfg.prop(1,:)=[0 0 1 1];

cfg.srctype='fourier';
cfg.srcpos=[50.0 200.0 100.0];
cfg.srcparam1=[100.0 0.0 0.0 2];
cfg.srcparam2=[0 100.0 0.0 0];
cfg.srcdir=[0 0 -1];
cfg.issrcfrom0=1;

cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.nphoton=1e8;
cfg.autopilot=1;
cfg.gpuid=1;
cfg.unitinmm=0.4*2;
cfg.debuglevel='P';

flux=mcxlab(cfg);
fcw=flux.data;
mcxplotvol(log10(double(fcw)));

%mcx2json(cfg,'digimouse.json');