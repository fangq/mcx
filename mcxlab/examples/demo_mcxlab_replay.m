%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.issrcfrom0=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;
% calculate the flux distribution with the given config
cfg.detpos=[15 30 0 2];
%cfg.savedetflag='dsp';
[flux, detp, vol, seeds]=mcxlab(cfg);

newcfg=cfg;
newcfg.seed=seeds.data;
newcfg.outputtype='jacobian';
newcfg.detphotons=detp.data;
[flux2, detp2, vol2, seeds2]=mcxlab(newcfg);
jac=sum(flux2.data,4);
imagesc(log10(abs(squeeze(jac(:,30,:)))))
