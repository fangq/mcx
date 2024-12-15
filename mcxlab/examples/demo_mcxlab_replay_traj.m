%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
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
[flux, detp, vol, seeds] = mcxlab(cfg);

newcfg = cfg;
newcfg.seed = seeds.data;
newcfg.outputtype = 'jacobian';
newcfg.detphotons = detp.data;
newcfg.maxjumpdebug = 29371108;
[flux2, detp2, vol2, seeds2, traj2] = mcxlab(newcfg);

jac = sum(flux2.data, 4);
imagesc(log10(abs(squeeze(jac(:, 30, :)))));

figure;
newtraj = mcxplotphotons(traj2);
title('photon trajectories');
idx = find(diff(newtraj.id));
% idx=[idx; length(newtraj.id)];

figure;
subplot(121);
pos = newtraj.pos(idx, :);
plot3(pos(:, 1), pos(:, 2), pos(:, 3), 'o');
hold on;
plotmesh(detp2.p, 'r.');

[tf, loc] = ismember(detp2.p, newtraj.pos([idx; end], :), 'rows');
detp2.p(1:2, :);
newtraj.pos(idx(loc(1:2)), :);

title('last position of detected photons');
subplot(122);
pos = newtraj.pos(idx + 1, :);
plot3(pos(:, 1), pos(:, 2), pos(:, 3), 'o');
title('launch position of each photon');
hold on;
