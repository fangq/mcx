%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this demo script, we verify the following two bugs are fixed
%
%  https://github.com/fangq/mcx/issues/103  (to handle focal point)
%  https://github.com/fangq/mcx/issues/104  (to handle mirror bc)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg
cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srctype='pattern';
cfg.srcpos=[-40,-40, 0];
cfg.srcparam1=[80 0 0 100];
cfg.srcparam2=[0 80 0 100];
cfg.srcdir=[0 0 1 30];
cfg.issrcfrom0=1;
cfg.srcpattern=zeros(100,100);
cfg.srcpattern(51:end,51:end)=1;
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 0.1 0 1];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-09;
cfg.bc='mm____';

flux=mcxlab(cfg);

mcxplotvol(log10(double(flux.data)))
