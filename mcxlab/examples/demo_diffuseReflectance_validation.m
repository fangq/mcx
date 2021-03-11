%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we validate surface diffuse reflectance extracted from MC
% detected photon profiles against analytical solutions computed from
% diffusion equations (semi-infinite domain).
%
% [Yao2018] Ruoyang Yao, Xavier Intes, and Qianqian Fang, "Direct approach 
% to compute Jacobians for diffuse optical tomography using perturbation Monte 
% Carlo-based photon “replay”," Biomed. Opt. Express 9, 4588-4603 (2018)
%
% [Kienle1997] Alwin Kienle and Michael S. Patterson, "Improved solutions 
% of the steady-state and the time-resolved diffusion equations for reflectance 
% from a semi-infinite turbid medium," J. Opt. Soc. Am. A 14, 246-254 (1997)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

%% Monte Carlo
cfg.nphoton=1e8; % photon number
cfg.vol=uint8(ones(60,60,60)); % size in mm (length,width,height)

cfg.prop=[0.00  0.0  1.0 1.0;  % first row: ambient medium by default
          0.01  10.0 0.9 1.4]; % media optical properties
      
cfg.srcpos=[30 30 0]; % (x,y,z) co-ordinates of light source
cfg.srcdir=[0 0 1];   % Direction - positive z-direction
cfg.issrcfrom0=1;

cfg.gpuid=1;    
cfg.autopilot=1;

% time window: for time-solved results
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

% save positions & directions of escaping photons
cfg.issaveexit=1;

% set up detectors: non-overlapping fiber detectors along x = 30
detpY=31:1:55; % y changes from 31 to 55
r=0.25; % radius 0.25mm
for i=1:size(detpY,2)
    cfg.detpos(i,:) = [30,detpY(i),0,r];
end

% visualize domain settings
figure(1);mcxpreview(cfg);

% launch simulations
[~,detp]=mcxlab(cfg);

% compute diffuse reflectance at detectors
drefMC = cwDrefMC(detp, cfg);

%% Diffusion
% flux at detectors
fluxDiffusion = cwFluxDiffusion(cfg.prop(2,1), cfg.prop(2,2) * (1 - cfg.prop(2,3)), ...
    cfg.prop(2,4), cfg.srcpos, cfg.detpos(:, 1:3));

% fluence at detectors
fluenceDiffusion = cwdiffusion(cfg.prop(2,1), cfg.prop(2,2) * (1 - cfg.prop(2,3)), ...
    findReff(cfg.prop(2,4)), cfg.srcpos, cfg.detpos(:, 1:3));

% surface diffuse reflectance
drefDiffusion = 0.118 * fluenceDiffusion + 0.306 * fluxDiffusion; % Eq. 8 of Kienle1997

%% compare diffuse reflectance
figure(2);
h=semilogy(1:length(detpY), drefDiffusion, 'r--', 1:length(detpY), drefMC, 'b+');
legend("Diffusion", "MC");
title("log10(Diffuse reflectance) [1/mm^{2}]");
xlabel("Source-Detector seperation [mm]");