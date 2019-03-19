%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to simulate an infinite slab using cyclic
% boundary condition
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg;
cfg.nphoton=1e8;
cfg.issrcfrom0=1;
cfg.vol=uint8(ones(60,60,20));
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.seed=99999;

% a uniform planar source outside the volume
cfg.srctype='planar';
cfg.srcpos=[0 0 0];
cfg.srcparam1=[60 0 0 0];
cfg.srcparam2=[0 60 0 0];
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.bc='ccrccr';
flux=mcxlab(cfg);

fcw=flux.data*cfg.tstep;
imagesc(log10(abs(squeeze(fcw(:,30,:)))))
axis equal; colorbar
title('a uniform planar source incident along an infinite slab');
