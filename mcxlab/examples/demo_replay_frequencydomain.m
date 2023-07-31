%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqian Fang
%
% In this example, we show how to compute the
% Jacobians for log-amplitude and phase shift in the
% frequency domain (FD)/radio frequency (RF) mode.
%
% NOTE: Requires (nightly-build) version from 15 May 2023 or later!
% Add the path to the MCXLAB-files.
%
% Ref.: Hirvi et al. (2023). Effects of atlas-based anatomy on modelled
% light transport in the neonatal head. Phys. Med. Biol.
% https://doi.org/10.1088/1361-6560/acd48c
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FORM TARGET MODEL
% Here we consider a simple layered slab model with brain-like optical parameters.
layered_model = ones(90,60,60); 
layered_model(:,:,4) = 2; 
layered_model(:,:,5:8) = 3; 
layered_model(:,:,9:end) = 4;
opt_params = [0,0,1,1; 0.0150,16,0.9,1.4; 0.004,1.6,0.9,1.4; 0.048,5,0.9,1.4; 0.037,10,0.9,1.4];
f_mod = 100e6; % 100 MHz RF modulation frequency

% MCXLAB SETTINGS
clear cfg
cfg.nphoton=1e9;
cfg.vol=uint8(layered_model);
cfg.prop=opt_params;
cfg.tstart=0;
cfg.tend=1/f_mod;
cfg.tstep=cfg.tend;
cfg.isnormalized=0;
cfg.unitinmm=1.0;
cfg.isspecular=0; % photons have initial weight 1
cfg.issrcfrom0=0; % first voxel is [1 1 1] in MATLAB 
cfg.srcpos=[30,30,1];
cfg.srcdir=[0,0,1];
cfg.detpos=[45,30,1,2; 60,30,1,2]; % two detectors, one source
cfg.maxdetphoton=5e7; 
cfg.gpuid=1; % cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.savedetflag='dp';

% FORWARD SIMULATION
% Calculate the forward simulation distribution with the given config
% for one source and both detectors.
[flux, detp, vol, seeds]=mcxlab(cfg);

% RF REPLAY JACOBIANS
[rfjac_lnA, rfjac_phase]=mcxrfreplay(cfg,f_mod,detp,seeds,1:2); % function from utils

% Separate source-detector -pairwise Jacobians as 3D matrices.
rfjac_lnA_det1=rfjac_lnA(:,:,:,1); % det 1
rfjac_lnA_det2=rfjac_lnA(:,:,:,2); % det 2
rfjac_phase_det1=rfjac_phase(:,:,:,1); % det 1
rfjac_phase_det2=rfjac_phase(:,:,:,2); % det 2

% Visualize Jacobian slices for both data types and source-detector pairs.
set(figure,'Position',[0 0 900 400]);tiledlayout(2,2,'Padding','compact','TileSpacing','compact');set(gcf,'Color','w');
nexttile;imagesc(((squeeze(rfjac_lnA_det1(:,30,1:20)))));view(-90,-90);axis image;title('d(lnA) det 1 [mm]');set(gca,'FontSize',14);colorbar
nexttile;imagesc(((squeeze(rfjac_lnA_det2(:,30,1:20)))));view(-90,-90);axis image;title('d(lnA) det 2 [mm]');set(gca,'FontSize',14);colorbar
nexttile;imagesc(((squeeze(rfjac_phase_det1(:,30,1:20)))));view(-90,-90);axis image;title('d(phase) det 1 [rad x mm]');set(gca,'FontSize',14);colorbar
nexttile;imagesc(((squeeze(rfjac_phase_det2(:,30,1:20)))));view(-90,-90);axis image;title('d(phase) det 2 [rad x mm]');set(gca,'FontSize',14);colorbar

