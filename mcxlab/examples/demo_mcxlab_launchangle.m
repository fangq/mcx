%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to customize photon launch angle distrbution
% using cfg.angleinvcdf
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% only clear cfg to avoid accidentally clearing other useful data
clear cfg

cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 15];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0 0 1 1];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.isreflect=0;

% define angleinvcdf in launch angle using cfg.angleinvcdf
cfg.angleinvcdf=linspace(1/6, 1/5, 5);  % launch angle is uniformly distributed between [pi/6 and pi/5] with interpolation (odd-number length)
flux=mcxlab(cfg);

mcxplotvol(log10(abs(flux.data)))
hold on
plot(cfg.angleinvcdf)


%% test Lambertian/cosine distribution
cfg.angleinvcdf=asin(0:0.01:1)/pi;  % Lambertian distribution pdf: PDF(theta)=cos(theta)/pi, CDF=sin(theta)/pi, invcdf=asin(0:1)/pi, with interpolation (cfg.srcdir(4) is not specified)
flux=mcxlab(cfg);
mcxplotvol(log10(abs(flux.data)))


%% discrete angles with interpolation
cfg.angleinvcdf=[0 0 0 0 1/6 1/6 1/4];  % interpolatation is used between discrete angular values
flux=mcxlab(cfg);
mcxplotvol(log10(abs(flux.data)))


%% discrete angles without interpolation (by setting focal-length cfg.srcdir(4) to 1)
cfg.angleinvcdf=[0 0 0 0 1/6 1/6 1/4];
cfg.srcdir(4)=1;   % disable interpolation, use the discrete angles in angleinvcdf elements only
flux=mcxlab(cfg);
mcxplotvol(log10(abs(flux.data)))

%% can be applied to area source - convolve with the area
cfg.srctype='disk';
cfg.srcparam1=[10,0,0,0];
cfg.srcdir(4)=0;
flux=mcxlab(cfg);
mcxplotvol(log10(abs(flux.data)))
