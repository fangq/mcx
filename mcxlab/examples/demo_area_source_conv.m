% ==========================================================================
% Comparing directly simulated widefield sources vs aperature-convolved
% pencil beam solutions
%
% Author: Qianqian Fang <q.fang at neu.edu>
% ==========================================================================

%% first, simulate a pencil beam solution

% only clear cfg to avoid accidentally clearing other useful data
clear cfg cfg2;
cfg.nphoton = 1e8;
cfg.vol = uint8(ones(60, 60, 60));
cfg.srcpos = [29.5 29.5 1];
cfg.srcdir = [0 0 1];
cfg.prop = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = cfg.tend;
% cfg.gpuid=1;
% cfg.autopilot=1;

flux = mcxlab(cfg);

%% next, perform a 2D convolution with a 3x7 kernel in the x/y plane

src = ones(3, 7);
fluxconv = zeros(size(flux.data));

for i = 1:size(flux.data, 3)
    fluxconv(:, :, i) = conv2(flux.data(:, :, i), src, 'same');
end
% normalize the convolved solution using total area
fluxconv = fluxconv ./ sum(src(:));

%% then directly simulate such an area source using `planar` source type

cfg2 = cfg;
cfg2.srctype = 'planar';
cfg2.srcpos = [28, 26, 0]; % notice this is shifted by half grid in x/y to align the two
cfg2.srcparam1 = [3, 0, 0, 0];
cfg2.srcparam2 = [0, 7, 0, 0];
fluxplanar = mcxlab(cfg2);

% %% additionally, we can simulate a pencilarray source
%
% cfg3=cfg2;
% cfg3.srctype='pencilarray';
% cfg3.srcparam1(4)=3;
% cfg3.srcparam2(4)=7;
% fluxpencilarray=mcxlab(cfg3);

%% comparing the two solutions

contournum = 30;

figure;
subplot(131);
contourf(log10(fluxplanar.data(:, :, 5)), contournum);
hold on;
contour(log10(fluxconv(:, :, 5)), contournum, 'r--');
% contour(log10(fluxpencilarray.data(:,:,5)), contournum, 'c--');
title('x-y plane (z=5)');
axis equal;
legend({'planar source', 'convolved pencil beam'});

subplot(132);
contourf(log10(squeeze(fluxplanar.data(:, 30, :))), contournum);
hold on;
contour(log10(squeeze(fluxconv(:, 30, :))), contournum, 'r--');
title('x-z plane (y=30)');
axis equal;
legend({'planar source', 'convolved pencil beam'});

subplot(133);
contourf(log10(squeeze(fluxplanar.data(30, :, :))), contournum);
hold on;
contour(log10(squeeze(fluxconv(30, :, :))), contournum, 'r--');
title('y-z plane (x=30)');
axis equal;
legend({'planar source', 'convolved pencil beam'});
