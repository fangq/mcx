
function [rfjac_lnA, rfjac_phase]=mcxrfreplay(cfg,f_mod,detp,seeds,detnums)
%
% [rfjac_lnA, rfjac_phase]=mcxrfreplay(cfg,f_mod,detp,seeds,detnums)
% 
% Compute the frequency domain (FD) log-amplitude and phase shift Jacobians
% with respect to voxel-wise absorption coefficients using the radio
% frequency (RF) replay algorithm.
%
% Author: Pauliina Hirvi (pauliina.hirvi <at> aalto.fi)
%         Qianqian Fang (q.fang <at> neu.edu)
%
% Ref.: Hirvi et al. (2023). Effects of atlas-based anatomy on modelled
% light transport in the neonatal head. Phys. Med. Biol.
% https://doi.org/10.1088/1361-6560/acd48c
%
% Input:
%     cfg: struct used in the main forward simulation
%     f_mod: RF modulation frequency
%     detp: the 2nd output from mcxlab, must be a struct
%     seeds: the 4th output from mcxlab
%     detnums: row array with the indices of the detectors to replay and obtain Jacobians
% 
% Output:
%     rfjac_lnA: a 4D array with dimensions specified by [size(vol) numb-of-detectors];
%                each 3D array contains the Jacobians for log-amplitude measurements
%     rfjac_phase: a 4D array with dimensions specified by [size(vol) numb-of-detectors];
%                  each 3D array contains the Jacobians for phase shift measurements
%
% License: GPLv3, see http://mcx.space/ for details
%

% Initialize the 4D arrays for collecting the Jacobians. The 4th dimension
% corresponds to the detector index.
rfjac_lnA=zeros([size(cfg.vol),length(detnums)]);
rfjac_phase=zeros([size(cfg.vol),length(detnums)]);

% Collect Jacobians one detector index at a time.
for d = detnums
    % MCXLAB REPLAY SETTINGS
    clear cfg_jac
    cfg_jac=cfg;
    cfg_jac.seed=seeds.data; 
    cfg_jac.detphotons=detp.data; 
    cfg_jac.replaydet=d; 
    cfg_jac.outputtype='rf';
    cfg_jac.omega=2*pi*f_mod; % 100 MHz RF modulation
    cfg_jac.isnormalized=0; % !

    % REPLAY SIMULATION
    [rfjac_d, detp_d, vol_d, seeds_d]=mcxlab(cfg_jac);
    detw=mcxdetweight(detp_d,cfg_jac.prop,cfg_jac.unitinmm); % array with detected photon weights
    dett=mcxdettime(detp_d,cfg_jac.prop,cfg_jac.unitinmm); % array with photon time-of-flights

    % FD MEASUREMENT ESTIMATES
    X=dot(detw,cospi((2*f_mod).*dett));
    Y=dot(detw,sinpi((2*f_mod).*dett));
    A=sqrt(X^2 + Y^2); % amplitude [a.u.]
    phase=atan2(Y,X) + (double(atan2(Y,X)<0))*2*pi; % phase shift in [0,2*pi] [rad]
    if A==0
        fprintf('MCX WARNING: No detected photons for detector %d.\n',d)
        continue;
    end

    % FD JACOBIANS
    % Compute the Jacobians with the rf replay feature.
    rfjac_d=sum(rfjac_d.data,4); % sum over time instances
    if cfg_jac.isnormalized==0
        rfjac_d=(cfg_jac.unitinmm).*rfjac_d; % correct units to [mm]
    end
    % Jacobians for X and Y wrt mua:
    rfjac_X=rfjac_d(:,:,:,:,:,1);
    rfjac_Y=rfjac_d(:,:,:,:,:,2);
    % Jacobians for log-amplitude and phase shift wrt mua:
    rfjac_lnA(:,:,:,d)=(1/(A^2)).*(X.*rfjac_X + Y.*rfjac_Y);
    rfjac_phase(:,:,:,d)=(1/(A^2)).*(X.*rfjac_Y - Y.*rfjac_X);
end

