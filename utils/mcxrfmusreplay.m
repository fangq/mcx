function [rfmusjac_lnA, rfmusjac_phase] = mcxrfmusreplay(cfg, f_mod, detp, seeds, detnums)
%
% [rfmusjac_lnA, rfmusjac_phase] = mcxrfmusreplay(cfg, f_mod, detp, seeds, detnums)
%
% Compute the frequency domain (FD) log-amplitude and phase shift Jacobians
% with respect to voxel-wise scattering coefficients using the radio
% frequency mode (RF) replay algorithm.
%
% Author: Pauliina Hirvi (pauliina.hirvi <at> aalto.fi)
%         Qianqian Fang (q.fang <at> neu.edu)
%
% Ref.: Hirvi et al. (2025): https://www.overleaf.com/read/qgtqcdyvqfrw#485e8c
% Hirvi et al. (2023): https://doi.org/10.1088/1361-6560/acd48c
%
% Input:
%     cfg: struct used in the main forward simulation
%     f_mod: RF modulation frequency in Hz
%     detp: the 2nd output from mcxlab, must be a struct
%     seeds: the 4th output from mcxlab
%     detnums: row array with the indices of the detectors to replay and obtain Jacobians
%
% Output:
%     rfmusjac_lnA: a 4D array with dimensions specified by [size(vol) numb-of-detectors];
%                each 3D array contains the Jacobians for log-amplitude measurements
%     rfmusjac_phase: a 4D array with dimensions specified by [size(vol) numb-of-detectors];
%                each 3D array contains the Jacobians for phase shift measurements
%
% License: GPLv3, see http://mcx.space/ for details
%

% Control.
if (nargin < 5)
    error('you must provide all 5 required input parameters');
end
if (~isfield(cfg, 'unitinmm'))
    cfg.unitinmm = 1;
end

% Initialize the 4D arrays for collecting the Jacobians. The 4th dimension
% corresponds to the detector index.
rfmusjac_lnA = zeros([size(cfg.vol), length(detnums)]);
rfmusjac_phase = zeros([size(cfg.vol), length(detnums)]);
% Return if no photons detected.
if (isempty(detp) || isempty(seeds))
    fprintf('MCX WARNING: No detected photons for replay.\n')
    return;
end

% Form matrix with mus for each nonzero voxel, and 1 in 0 type or mus=0.
nonzero_ind = find(cfg.vol(:)>0);
nonzero_prop_row = double(cfg.vol(nonzero_ind)) + ones(size(nonzero_ind));
mus_matrix = ones(size(cfg.vol));
mus_matrix(nonzero_ind) = cfg.prop(nonzero_prop_row,2);
mus_matrix = mus_matrix + double(mus_matrix==0); % avoid division by zero if mus=0

% General replay settings.
clear cfg_jac;
cfg_jac = cfg;
cfg_jac.seed = seeds.data;
cfg_jac.detphotons = detp.data;
cfg_jac.omega = 2 * pi * f_mod; % RF modulation angular frequency
cfg_jac.isnormalized = 0; % !
cfg_jac.issave2pt = 1;

% Collect Jacobians one detector index at a time.
for d = detnums
    if ~ismember(d,detp.detid)
        fprintf('MCX WARNING: No detected photons for detector %d.\n', d);
        continue;
    end

    % REPLAY SIMULATION 1
    cfg_jac.replaydet = d;
    cfg_jac.outputtype = 'rf'; % FD absorption Jacobians 
    rfjac_d = mcxlab(cfg_jac);
    rfjac_d = sum(rfjac_d.data, 4); % sum over time instances
    if cfg_jac.isnormalized == 0
        rfjac_d = (cfg_jac.unitinmm) .* rfjac_d; % correct units to [mm]
    end
    % Jacobians for X and Y wrt mua:
    rfjac_X = rfjac_d(:, :, :, :, :, 1); % (-1*)cos-weighted paths
    rfjac_Y = rfjac_d(:, :, :, :, :, 2); % (-1*)sine-weighted paths
    clear rfjac_d

    % REPLAY SIMULATION 2
    cfg_jac.outputtype = 'rfmus'; % FD scattering Jacobians 
    [rfmusjac_d, detp_d, vol_d, seeds_d] = mcxlab(cfg_jac);
    rfmusjac_d = sum(rfmusjac_d.data, 4); % sum over time instances
    % Jacobians for X and Y wrt mus:
    rfmusjac_X = rfmusjac_d(:, :, :, :, :, 1); % cos-weighted nscatt
    rfmusjac_X = rfmusjac_X./mus_matrix + rfjac_X; clear rfjac_X
    rfmusjac_Y = rfmusjac_d(:, :, :, :, :, 2); % sine-weighted nscatt
    rfmusjac_Y = rfmusjac_Y./mus_matrix + rfjac_Y; clear rfjac_Y rfmusjac_d

    % FD MEASUREMENT ESTIMATES
    detw = mcxdetweight(detp_d, cfg_jac.prop, cfg_jac.unitinmm); % array with detected photon weights
    dett = mcxdettime(detp_d, cfg_jac.prop, cfg_jac.unitinmm); % array with photon time-of-flights    
    X = dot(detw, cos((cfg_jac.omega) .* dett));
    Y = dot(detw, sin((cfg_jac.omega) .* dett));
    A = sqrt(X^2 + Y^2); % amplitude [a.u.]
    phase = atan2(Y, X) + (double(atan2(Y, X) < 0)) * 2 * pi; % phase shift in [0,2*pi] [rad]

    % FINAL SCATTERING JACOBIANS
    rfmusjac_lnA(:, :, :, d) = (1 / (A^2)) .* (X .* rfmusjac_X + Y .* rfmusjac_Y);
    rfmusjac_phase(:, :, :, d) = (1 / (A^2)) .* (X .* rfmusjac_Y - Y .* rfmusjac_X);
end
