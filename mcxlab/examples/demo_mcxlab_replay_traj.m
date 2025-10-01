%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to handle photon trajectories and their
% index mapping to the detected photon data in a replay simulation.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs;
cfg.nphoton = 1e6;
cfg.vol = uint8(ones(60, 60, 60));
cfg.srcpos = [30 30 0];
cfg.srcdir = [0 0 1];
cfg.gpuid = 1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot = 1;
cfg.issrcfrom0 = 1;
cfg.prop = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-10;
cfg.issaveexit = 1;
% calculate the flux distribution with the given config
cfg.detpos = [15 30 0 2];

%% base line simulation
[flux, detp, vol, seeds] = mcxlab(cfg);

%% setup replay simulation data

newcfg = cfg;
newcfg.seed = seeds.data;
newcfg.outputtype = 'jacobian';
newcfg.detphotons = detp.data;
newcfg.maxjumpdebug = 29371108;

%% run replay
[flux2, detp2, vol2, seeds2, traj2] = mcxlab(newcfg);

jac = sum(flux2.data, 4);

figure;
imagesc(log10(abs(squeeze(jac(:, :, 1)'))) / 10);
hold on;

%% mcxplotphotons reorders trajectory positions in the sequential order
newtraj = mcxplotphotons(traj2);
hold on;
plot3(cfg.srcpos(1), cfg.srcpos(2), cfg.srcpos(3), 'ro', 'MarkerSize', 10);
plot3(cfg.detpos(1), cfg.detpos(2), cfg.detpos(3), 'yo', 'MarkerSize', 10);

title('photon trajectories');

% idx is the index of the last trajectory position in the reordered newtraj
idx = find(diff(newtraj.id));
% idx=[idx; length(newtraj.id)];

%% visually plotting the detected photon exit position detp2.p with the last position stored in the trajectory
figure;
subplot(121);
pos = newtraj.pos(idx, :);
plot3(pos(:, 1), pos(:, 2), pos(:, 3), 'o');
hold on;
plotmesh(detp2.p, 'r.');
legend('using traj.pos', 'using detp.p');
title('last position of detected photons');

%% compare and map the detected photon data with trajectory data using ismember
[tf, loc] = ismember(detp2.p, newtraj.pos([idx; end], :), 'rows');

% loc is the index in idx so that trajectory data can match detected photon
detp2.p(1:2, :);
newtraj.pos(idx(loc(1:2)), :);

subplot(122);
pos = newtraj.pos(idx + 1, :);
plot3(pos(:, 1), pos(:, 2), pos(:, 3), 'o');
title('launch position of each photon');
hold on;
