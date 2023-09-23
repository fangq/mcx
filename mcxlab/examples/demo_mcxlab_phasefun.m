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
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

% use built-in Henyey-Greenstein phase function
flux=mcxlab(cfg);

% define Henyey-Greenstein phase function using cfg.invcdf
invhg =@(u,g) (1 + g*g - ((1-g*g)./(1-g+2*g*u)).^2)./(2*g);
cfg.invcdf=invhg(0.01:0.01:1-0.01, 0.8);
flux=mcxlab(cfg);

% now use Rayleigh scattering phase function P(cos(theta))=P(u)=3/4*(1+u.^2)
invrayleigh=@(u) (4*u + ((4*u - 2).^2 + 1).^(1/2) - 2).^(1/3) - 1./(4*u + ((4*u - 2).^2 + 1).^(1/2) - 2).^(1/3);
cfg.invcdf=invrayleigh(0.01:0.01:1-0.01);

% calculate the flux distribution using Rayleigh scattering phase function
flux=mcxlab(cfg);