%% demo_mcxlab_rfforward.m - Frequency-domain forward MC with MCX
%
% Eq.9: w <- w * exp[-(mu_a + i*omega*n/c0) * ds]
% Leino, Pulkkinen, and Tarvainen, OSA Continuum 2(3), 957-965 (2019)
%
% Key setting: cfg.omega = 2*pi*freq  (no special outputtype needed for forward mode)
% When seed is NOT from file and omega > 0, MCX automatically runs RF forward mode.
% The output is a complex-valued fluence volume.

clear;
clc;

%% Common setup
cfg.nphoton = 1e7;
cfg.vol = uint8(ones(60, 60, 60));
cfg.srcpos = [30 30 1];
cfg.srcdir = [0 0 1];
cfg.issrcfrom0 = 1;
cfg.gpuid = 1;
cfg.autopilot = 1;
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.prop = [0 0 1 1; 0.01 10 0.9 1.37];

%% CW simulation
cfg_cw = cfg;
cfg_cw.outputtype = 'flux';
fprintf('Running CW ...\n');
fluence_cw = mcxlab(cfg_cw);

%% RF forward at 100 MHz
cfg_rf = cfg;
cfg_rf.omega = 2 * pi * 100e6;
fprintf('Running RF forward at %.0f MHz ...\n', cfg_rf.omega / (2 * pi) / 1e6);
fluence_rf = mcxlab(cfg_rf);

%% The output .data is complex
phi = fluence_rf.data;
fprintf('Complex output: %d, size: %s\n', ~isreal(phi), mat2str(size(phi)));

amp = abs(phi);
phs = angle(phi) * 180 / pi;

%% Plot
z = 30;
figure('Position', [100 100 1200 400]);

subplot(1, 3, 1);
imagesc(log10(max(fluence_cw.data(:, :, z), 1e-12)));
title('CW log10(fluence)');
axis image;
colorbar;

subplot(1, 3, 2);
imagesc(log10(max(amp(:, :, z), 1e-12)));
title('FD log10(amplitude)');
axis image;
colorbar;

subplot(1, 3, 3);
imagesc(phs(:, :, z));
title('FD phase (deg)');
axis image;
colorbar;

sgtitle(sprintf('MCX RF Forward  f=%g MHz  mua=%.3f/mm', ...
                cfg_rf.omega / (2 * pi) / 1e6, cfg.prop(2, 1)));

%% Depth profiles
figure;
zz = 1:60;
yyaxis left;
semilogy(zz, squeeze(fluence_cw.data(30, 30, :)), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(zz, squeeze(amp(30, 30, :)), 'r--', 'LineWidth', 1.5);
ylabel('Fluence / Amplitude');
yyaxis right;
plot(zz, squeeze(phs(30, 30, :)), 'g-', 'LineWidth', 1.5);
ylabel('Phase (deg)');
xlabel('Depth (voxels)');
title('Depth profile x=30 y=30');
legend('CW', 'FD Amp', 'Phase');
