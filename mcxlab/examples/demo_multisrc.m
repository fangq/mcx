%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to simultaneously simulate multiple sources
% of the same type.
%
% Specifically, we create a complex source made of an array of 9 planar
% sources - where 8 of the sources form a circular array by rotating one
% of the sources around the center of the bottom-face center; the 9th
% source is manually placed at the center of the array.
%
% One can individually control the settings for each of the sources,
% including the position (srcpos), launch direction (srcdir), and
% additional parameters (srcparam1/srcparam2).

% By default, each source entry launches an equal fraction of photons,
% determined by cfg.nphoton/size(cfg.srcpos(1,:)). However, one can set the
% 4-th element of the srcpos() for each sourc to create different weights
% for each source. Here, we set the central source 4x of the intensity
% compared to other circular-array sources (weight of 1). This allows mcx
% to allocate 4/(8+4) = 1/3 of the total photons (cfg.nphoton) to the central
% source while each other source launches 1/12 of the total photon packets.
%
% Worth to mention that we also set the 4th element of cfg.srcdir, i.e. the
% focal length, to create a convergent beam for each planar source.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:https://mcx.space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% only clear cfg to avoid accidentally clearing other useful data
clear cfg cfgs;

cfg.nphoton = 1e7;
cfg.vol = uint8(ones(60, 60, 40));
cfg.srctype = 'planar';

Rs = 20;
delta = pi / 4;

ang = 0:delta:(2 * pi - delta);
rotmat2d = [cos(delta), -sin(delta); sin(delta), cos(delta)];
offset2d = [30 30];

% first create an array of planar sources by rotating around origin

% the 4th element of srcpos sets the initial weight
cfg.srcpos = repmat([Rs, 0, 0, 1], length(ang), 1);

% the 4th element of srcdir sets the focal length (positive: convergent)
cfg.srcdir = repmat([-0.5, 0, sqrt(3) / 2, 20], length(ang), 1);
cfg.srcparam1 = repmat([-8, 3, 0, 0], length(ang), 1);
cfg.srcparam2 = repmat([3, 3, 0, 0], length(ang), 1);

for i = 2:length(ang)
    cfg.srcpos(i, 1:2) = (rotmat2d * cfg.srcpos(i - 1, 1:2)')';
    cfg.srcdir(i, 1:2) = (rotmat2d * cfg.srcdir(i - 1, 1:2)')';
    cfg.srcparam1(i, 1:2) = (rotmat2d * cfg.srcparam1(i - 1, 1:2)')';
    cfg.srcparam2(i, 1:2) = (rotmat2d * cfg.srcparam2(i - 1, 1:2)')';
end

% translate the circular array to the desired offset
cfg.srcpos(:, 1) = cfg.srcpos(:, 1) + offset2d(1);
cfg.srcpos(:, 2) = cfg.srcpos(:, 2) + offset2d(2);

% manually add the 9th source to the center

cfg.srcpos(end + 1, :) = [offset2d(1) - Rs / 4, offset2d(2) - Rs / 4, 0, 4]; % initial weight is 4
cfg.srcdir(end + 1, :) = [0, 0 1 0];
cfg.srcparam1(end + 1, :) = [Rs / 2, 0, 0, 0];
cfg.srcparam2(end + 1, :) = [0, Rs / 2, 0, 0];

cfg.gpuid = 1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot = 1;
cfg.prop = [0 0 1 1; 0.005 0.2 0 1.37];
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;

flux = mcxlab(cfg);

mcxplotvol(log10(flux.data));

%% setting cfg.srcid to a positive number 1,2,3,.. specifies which src to simulate
cfg.srcid = 8;

flux = mcxlab(cfg);

mcxplotvol(log10(flux.data));

%% setting cfg.srcid to -1, solution of each src is stored separately

cfg.srcid = -1;

flux = mcxlab(cfg);

% use up-down button to shift between different sources, there are 9 of
% them

mcxplotvol(log10(squeeze(flux.data)));
