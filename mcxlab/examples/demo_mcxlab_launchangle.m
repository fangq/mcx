%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to customize phase function using cfg.invcdf
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% only clear cfg to avoid accidentally clearing other useful data
clear cfg
cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 0.0 1 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.isreflect=0;

cfg.srctype='cone';   % source type must be a non-pencil beam, the launch angle of the selected source will be replaced by those computed by angleinvcdf

% define angleinvcdf in launch angle using cfg.angleinvcdf
cfg.angleinvcdf=linspace(1/4, 1/3);  % launch angle is uniformly distributed between [pi/4 and pi/3]
flux=mcxlab(cfg);

mcxplotvol(log10(abs(flux.data)))