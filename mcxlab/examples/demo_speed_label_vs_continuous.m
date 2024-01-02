%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show why using label-based simulation is preferred
% over per-voxel medium formats whenever possible.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% only clear cfg to avoid accidentally clearing other useful data
clear cfg cfgs;
cfg.nphoton = 1e7;
cfg.vol = uint8(ones(60, 60, 60));
cfg.vol(:, :, 4:10) = 2;
cfg.srcpos = [30 30 1];
cfg.srcdir = [0 0 1];
cfg.detpos = [30 33 1 2];
cfg.gpuid = 1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot = 1;
cfg.prop = [0 0 1 1; 0.05 1 0 1.37; 0.1 1 0 1.37];
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;

%% simulate with lable-based medium
tic;
flux = mcxlab(cfg);
toc;

%% simulate with mua_float format
cfg2 = cfg;
cfg2.vol = reshape(single(cfg.prop(cfg.vol + 1, 1)), [1 size(cfg.vol)]);

tic;
flux2 = mcxlab(cfg2);
toc;

%% simulate with muamus_float format
cfg3 = cfg;
cfg3.vol = reshape(single(cfg.prop(cfg.vol + 1, 1:2))', [2 size(cfg.vol)]);

tic;
flux3 = mcxlab(cfg3);
toc;

%% simulate with asgn_float format
cfg4 = cfg;
cfg4.vol = reshape(single(cfg.prop(cfg.vol + 1, 1:4))', [4 size(cfg.vol)]);

tic;
flux4 = mcxlab(cfg4);
toc;
