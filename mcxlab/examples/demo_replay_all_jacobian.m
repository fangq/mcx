%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqian Fang
%
% In this example, we show how to compute any or all of the available
% Jacobians for the continuous wave (CW), time (TD) and frequency domain (FD)
% measurements with respect to voxel-wise absorption and scattering coefficients.
%
% NOTE: Requires (nightly-build) version from May 2025 or later!
% Remember to add path to /mcx/mcxlab and /mcx/utils.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%
% Modified by Pauliina Hirvi 4/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORM TARGET MODEL

% Here we consider a simple layered slab model.
layered_model = ones(60, 60, 60);
layered_model(:, :, 4) = 2;
layered_model(:, :, 5:8) = 3;
layered_model(:, :, 9:end) = 4;

% Brain-like optical parameters.
% mua = absorption coeff, mus = scattering coeff, g = anisotropy coeff,
% n = index of refraction.
opt_params = [0, 0, 1, 1; 0.0150, 14, 0.9, 1.4; 0.004, 1.6, 0.9, 1.4;...
    0.048, 5, 0.9, 1.4; 0.037, 10, 0.9, 1.4]; % [mua,mus,g,n]
mus_matrix = ones(size(layered_model)); % matrix with mus value for each non-zero voxel  
mus_matrix(find(layered_model(:)>0)) = opt_params(1+layered_model(find(layered_model(:)>0)),2);  
mus_matrix = mus_matrix + double(mus_matrix==0); % avoid division with zero

% Select modulation frequency for FD.
f_mod = 100e6; % 100 MHz RF modulation frequency

%% MCXLAB FORWARD SIMULATION 

% Settings for forward simulation.
clear cfg;
cfg.nphoton = 5e9; % large for scattering Jacobians
cfg.vol = uint8(layered_model);
cfg.prop = opt_params;
cfg.tstart = 0;
cfg.tend = 1/f_mod;
cfg.tstep = cfg.tend; % cfg.tend for one time window
cfg.isnormalized = 0; % for Jacobians, avoid division with weight
cfg.unitinmm = 1.0;
cfg.isspecular = 0; % photons have initial weight 1
cfg.issrcfrom0 = 0; % first voxel is [1 1 1] in MATLAB
cfg.voidtime = 0; % only record travel time within medium
cfg.srcpos = [30, 30, 1];
cfg.srcdir = [0, 0, 1];
cfg.detpos = [45, 30, 1, 2]; % one detector
cfg.maxdetphoton = cfg.nphoton/10;
cfg.gpuid = 1; % cfg.gpuid='11'; % use two GPUs together
cfg.autopilot = 1;
cfg.issave2pt = 0; % faster forward simulation by not saving flux matrix
cfg.savedetflag = 'dp';

% Calculate the forward simulation distribution with the given config
% for one source and one detector.
[~, detp, ~, seeds] = mcxlab(cfg);

% Compute total detected intensity as sum of detected weights.
detw = mcxdetweight(detp, cfg.prop, cfg.unitinmm); % array with detected photon weights
I = sum(detw);

% Compute weighted mean time-of-flight (TOF).
dett = mcxdettime(detp, cfg.prop, cfg.unitinmm); % array with photon time-of-flights
meanTOF = sum(detw.*dett)/sum(detw);

%% REPLAY MODE

% Common settings for replay mode.
cfg_jacobian = cfg;
cfg_jacobian.seed = seeds.data;
cfg_jacobian.detphotons = detp.data;
cfg_jacobian.issave2pt = 1; % enable saving first output matrix from this on
cfg_jacobian.isnormalized = 0; % avoid double division with weight

%% REPLAY JACOBIAN: CW/TD (LOG-)INTENSITY VERSUS ABSORPTION 

% Absorption Jacobian for CW/TD (log-)intensity.
% See also: demo_mcxlab_replay.m
cfg_dIdmua = cfg_jacobian;
cfg_dIdmua.outputtype = 'jacobian';
dIdmua = mcxlab(cfg_dIdmua);
dIdmua = (-cfg_dIdmua.unitinmm).*dIdmua.data; % correct sign and units to [mm]
dlogIdmua = dIdmua./I; % convert to log-intensity Jacobian, i.e., weighted mean path for each voxel

%% REPLAY JACOBIAN: CW/TD LOG-INTENSITY VERSUS SCATTERING

% Scattering Jacobian for CW/TD (log-)intensity.
cfg_dIdmus = cfg_jacobian;
cfg_dIdmus.outputtype = 'nscat'; % 'wp' or 'nscat' 
meanP = mcxlab(cfg_dIdmus); % sum of weighted scattering counts in each voxel
meanP = meanP.data/I;
dlogIdmus = meanP./mus_matrix + dlogIdmua;

%% REPLAY JACOBIAN: TD MEAN TIME-OF-FLIGHT VERSUS ABSORPTION 

% Absorption Jacobian for TD weighted mean time-of-flight (TOF).
cfg_dTOFdmua = cfg_jacobian;
cfg_dTOFdmua.outputtype = 'wltof';
meanLTOF = mcxlab(cfg_dTOFdmua);
meanLTOF = cfg_dTOFdmua.unitinmm.*(meanLTOF.data)./I; % correct units to [mm]
dTOFdmua = -meanLTOF - meanTOF.*dlogIdmua; 

%% REPLAY JACOBIAN: TD MEAN TIME-OF-FLIGHT VERSUS SCATTERING

% Scattering Jacobian for TD weighted mean time-of-flight (TOF).
cfg_dTOFdmus = cfg_jacobian;
cfg_dTOFdmus.outputtype = 'wptof';
meanPTOF = mcxlab(cfg_dTOFdmus); 
meanPTOF = meanPTOF.data./I;
dTOFdmus = (meanPTOF - meanTOF.*meanP)./mus_matrix + dTOFdmua;

%% REPLAY JACOBIAN: FD LOG-AMPLITUDE VERSUS ABSORPTION 

% Absorption Jacobians for FD log-amplitude and phase shift in radians.
% See also:
[dlogAdmua, dphasedmua] = mcxrfreplay(cfg, f_mod, detp, seeds, 1); % function from utils

%% JACOBIAN: FD LOG-AMPLITUDE VERSUS SCATTERING

% Scattering Jacobians for FD log-amplitude and phase shift in radians.
[dlogAdmus, dphasedmus] = mcxrfmusreplay(cfg, f_mod, detp, seeds, 1); % function from utils

%% VISUALIZE ALL JACOBIANS

% Visualize slices of each Jacobian type.
set(figure, 'Position', [0 0 1200 1200]);set(gcf, 'Color', 'w');
tiledlayout(4, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
% CW/TD dlogIdmua:
nexttile;
imagesc((squeeze(dlogIdmua(:, 31, 1:20))));
view(-90, -90); colorbar;
title('CW/TD d(lnI)/dmua [mm]');
axis image off; set(gca, 'FontSize', 14); 
% FD dlogAdmua:
nexttile;
imagesc((squeeze(dlogAdmua(:, 31, 1:20))));
view(-90, -90); colorbar;
title('FD d(lnA)/dmua [mm]');
axis image off; set(gca, 'FontSize', 14); 
% CW/TD dlogIdmus:
nexttile;
imagesc((squeeze(dlogIdmus(:, 31, 1:20))));
view(-90, -90); colorbar;
title('CW/TD d(lnI)/dmus [mm]');
axis image off; set(gca, 'FontSize', 14); 
% FD dlogAdmus:
nexttile;
imagesc((squeeze(dlogAdmus(:, 31, 1:20))));
view(-90, -90); colorbar;
title('FD d(lnA)/dmus [mm]');
axis image off; set(gca, 'FontSize', 14); 
% TD dTOFdmua:
nexttile;
imagesc((squeeze((2*pi*f_mod).*dTOFdmua(:, 31, 1:20))));
view(-90, -90); colorbar;
title('TD (2 x pi x f ) x d(TOF)/dmua [mm]');
axis image off; set(gca, 'FontSize', 14); 
% FD dphasedmua:
nexttile;
imagesc((squeeze(dphasedmua(:, 31, 1:20))));
view(-90, -90); colorbar;
title('FD d(Phase)/dmua [mm]');
axis image off; set(gca, 'FontSize', 14); 
% TD dTOFdmus:
nexttile;
imagesc((squeeze((2*pi*f_mod).*dTOFdmus(:, 31, 1:20))));
view(-90, -90); colorbar;
title('TD (2 x pi x f) x d(TOF)/dmus [mm]');
axis image off; set(gca, 'FontSize', 14); 
% FD dphasedmus:
nexttile;
imagesc((squeeze(dphasedmus(:, 31, 1:20))));
view(-90, -90); colorbar;
title('FD d(Phase)/dmus [mm]');
axis image off; set(gca, 'FontSize', 14); 
