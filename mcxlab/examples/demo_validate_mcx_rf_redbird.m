%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency-Domain comparison: Redbird (FEM diffusion) vs MCX (RF forward MC)
%
% Redbird solves the frequency-domain diffusion equation:
%   -div( D * grad(phi) ) + (mua + i*omega/(c/n)) * phi = q
% MCX RF forward uses complex photon weights per Eq.9:
%   w <- w * exp[ -(mua + i*omega*n/c0) * ds ]
%
% Both produce complex-valued fluence. We compare amplitude and phase.
%
% Based on demo_redbird_vs_mcx.m (CW version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg xcfg;

freq = 200e6;              % modulation frequency in Hz
omega = 2 * pi * freq;     % angular frequency in rad/s

mua  = 0.005;              % absorption coeff [1/mm]
musp = 1.0;                % reduced scattering coeff [1/mm]
g    = 0;                  % anisotropy (isotropic for diffusion comparison)
nmed = 1.37;               % refractive index

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird FEM setup (frequency domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[cfg.node, cfg.elem] = meshgrid6(0:2:60, 0:2:60, 0:2:30);
cfg.elem(:, 1:4) = meshreorient(cfg.node(:, 1:3), cfg.elem(:, 1:4));
cfg.face = volface(cfg.elem);

nn = size(cfg.node, 1);
cfg.seg = ones(size(cfg.elem, 1), 1);

cfg.srcpos = [29.5 29.5 0];
cfg.srcdir = [0 0 1];

cfg.prop = [0 0 1 1; mua musp g nmed];
cfg.omega = omega;             %% <-- KEY: non-zero omega for FD

cfg.detpos = [40 30 0];
cfg.detdir = [0 0 -1];

cfg = rbmeshprep(cfg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird: Build and solve FD system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Building Redbird FD system (f = %.0f MHz) ...\n', freq / 1e6);
[Amat, deldotdel] = rbfemlhs(cfg);
[rhs, loc, bary] = rbfemrhs(cfg);

tic;
fprintf('Solving Redbird FD ...\n');
phi_rb = rbfemsolve(Amat, rhs);
toc;

fprintf('Redbird output is complex: %d\n', ~isreal(phi_rb));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird CW for reference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cfg0 = cfg;
cfg0.omega = 0;
cfg0 = rbmeshprep(cfg0);
[Amat0] = rbfemlhs(cfg0);
[rhs0] = rbfemrhs(cfg0);
phi_rb0 = rbfemsolve(Amat0, rhs0);
phi_rb0(phi_rb0 < 0) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Extract Redbird detector readings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

detval_rb = rbfemgetdet(phi_rb, cfg, rhs);
fprintf('Redbird detector: amplitude = %.6e, phase = %.2f deg\n', ...
        abs(detval_rb), angle(detval_rb) * 180 / pi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   MCX RF Forward setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (~exist('mcxlab', 'file'))
    error('mcxlab is not found in path. Please add mcxlab to your MATLAB path.');
end

xcfg.nphoton = 1e8;
xcfg.vol = uint8(ones(60, 60, 30));
xcfg.srcdir = [0 0 1 0];
xcfg.autopilot = 1;
xcfg.prop = [0 0 1 1; mua musp g nmed];
xcfg.tstart = 0;
xcfg.tend = 5e-9;
xcfg.tstep = 5e-9;
xcfg.seed = 99999;
xcfg.issrcfrom0 = 1;
xcfg.isreflect = 1;
xcfg.srctype = 'pencil';
xcfg.srcpos = [29.5 29.5 0];  % match Redbird source; issrcfrom0=1 so coords are physical mm

xcfg.omega = omega;            %% <-- KEY: modulation angular frequency (omega>0 triggers RF forward)

fprintf('Running MCX RF forward (f = %.0f MHz, %.0e photons) ...\n', freq / 1e6, xcfg.nphoton);
tic;
flux_rf = mcxlab(xcfg);
toc;

% RF forward outputs complex flux rate (same unit as CW 'flux')
% Multiply by tstep to get fluence, same workflow as CW
phi_mcx = flux_rf.data * xcfg.tstep;
fprintf('MCX output is complex: %d, size: %s\n', ~isreal(phi_mcx), mat2str(size(phi_mcx)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   MCX CW for reference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xcfg_cw = xcfg;
xcfg_cw.outputtype = 'flux';
xcfg_cw = rmfield(xcfg_cw, 'omega');

fprintf('Running MCX CW (%.0e photons) ...\n', xcfg_cw.nphoton);
tic;
flux_cw = mcxlab(xcfg_cw);
toc;

phi_cw = flux_cw.data * xcfg_cw.tstep;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Extract cross-sections at y=29.5 (voxel index 30)
%%   MCX vol is [Nx, Ny, Nz] = [60, 60, 30]
%%   issrcfrom0=1: domain is [0,60]x[0,60]x[0,30], voxel centers at 0.5,1.5,...
%%   Redbird mesh: node coords in [0,60]x[0,60]x[0,30]  (same physical domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MCX: extract y-voxel index 30 -> center at y=29.5, result is [Nx, Nz] = [60, 30]
% With issrcfrom0=1, voxel centers are at 0.5, 1.5, ..., 59.5 (x,y) and 0.5, ..., 29.5 (z)

mcx_amp_yz   = squeeze(abs(phi_mcx(:, 30, :)));       % [60 x 30]
mcx_phase_yz = squeeze(angle(phi_mcx(:, 30, :))) * 180 / pi;
cw_yz        = squeeze(real(phi_cw(:, 30, :)));        % [60 x 30]

% For imagesc: we want x on horizontal axis, z on vertical axis (z increasing downward)
% imagesc(x_range, z_range, data) where data is [Nz x Nx]
x_mcx = 0.5:59.5;
z_mcx = 0.5:29.5;

% Redbird: cut at y=29.5 (closest to MCX y=30 slice center)
[cutpos_rb, cutval_amp] = qmeshcut(cfg.elem, cfg.node, abs(phi_rb(:, 1)), 'y=29.5');
[~, cutval_phase]       = qmeshcut(cfg.elem, cfg.node, angle(phi_rb(:, 1)) * 180 / pi, 'y=29.5');
[~, cutval_cw]          = qmeshcut(cfg.elem, cfg.node, real(phi_rb0(:, 1)), 'y=29.5');

% Grid the Redbird cut data onto a regular grid matching MCX
[xi, zi] = meshgrid(x_mcx, z_mcx);  % [30 x 60]
rb_amp_grid   = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_amp, xi, zi);
rb_phase_grid = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_phase, xi, zi);
rb_cw_grid    = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_cw, xi, zi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 1: Side-by-side 2D maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'FD Comparison: Maps', 'Position', [50 100 1400 800]);

% Shared color limits for amplitude
amp_clim = [-6 0];
phs_clim = [-60 5];

% Row 1: Log-amplitude
subplot(2, 3, 1);
imagesc(x_mcx, z_mcx, log10(cw_yz' + eps));
axis equal tight;
colorbar;
caxis(amp_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title('MCX CW log_{10}(fluence)');

subplot(2, 3, 2);
imagesc(x_mcx, z_mcx, log10(mcx_amp_yz' + eps));
axis equal tight;
colorbar;
caxis(amp_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('MCX RF Amp log_{10} (f=%g MHz)', freq / 1e6));

subplot(2, 3, 3);
imagesc(x_mcx, z_mcx, log10(rb_amp_grid + eps));
axis equal tight;
colorbar;
caxis(amp_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Redbird FD Amp log_{10} (f=%g MHz)', freq / 1e6));

% Row 2: Phase
subplot(2, 3, 4);
imagesc(x_mcx, z_mcx, zeros(length(z_mcx), length(x_mcx)));
axis equal tight;
colorbar;
caxis(phs_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title('MCX CW phase (0 deg)');

subplot(2, 3, 5);
imagesc(x_mcx, z_mcx, mcx_phase_yz');
axis equal tight;
colorbar;
caxis(phs_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('MCX RF Phase (deg, f=%g MHz)', freq / 1e6));

subplot(2, 3, 6);
imagesc(x_mcx, z_mcx, rb_phase_grid);
axis equal tight;
colorbar;
caxis(phs_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Redbird FD Phase (deg, f=%g MHz)', freq / 1e6));

sgtitle_compat(gcf, sprintf('Frequency-Domain: Redbird (FEM) vs MCX RF Forward (MC)  |  f = %g MHz, \\mu_a = %.3f/mm, \\mu_s'' = %.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 2: Contour overlay comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Contour Comparison', 'Position', [50 50 1200 500]);

% Amplitude contours
subplot(1, 2, 1);
clines_amp = -7:0.5:0;
contour(x_mcx, z_mcx, log10(rb_amp_grid + eps), clines_amp, 'r-', 'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, log10(mcx_amp_yz' + eps), clines_amp, 'b--', 'LineWidth', 2);
legend('Redbird FD', 'MCX RF Forward', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('log_{10}(Amplitude)  f = %g MHz', freq / 1e6));
axis equal tight;

% Phase contours
subplot(1, 2, 2);
clines_phase = -60:5:0;
contour(x_mcx, z_mcx, rb_phase_grid, clines_phase, 'r-', 'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, mcx_phase_yz', clines_phase, 'b--', 'LineWidth', 2);
legend('Redbird FD', 'MCX RF Forward', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Phase (deg)  f = %g MHz', freq / 1e6));
axis equal tight;

sgtitle_compat(gcf, 'Contour overlay: Redbird (red solid) vs MCX RF (blue dashed)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 3: Depth profiles at (x=30, y=30)
%%   MCX: column x=30 (MATLAB index 30), y=30
%%   Redbird: interpolate at nodes near x=30, y=30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Depth Profiles', 'Position', [100 100 1000 700]);

% MCX depth profiles (z = 0.5 to 29.5 for issrcfrom0=0)
z_line = (1:30) - 0.5;
mcx_amp_line   = squeeze(abs(phi_mcx(30, 30, :)));
mcx_phase_line = squeeze(angle(phi_mcx(30, 30, :))) * 180 / pi;
cw_mcx_line    = squeeze(real(phi_cw(30, 30, :)));

% Redbird: extract depth profile at x=29.5 from already-interpolated 2D grids
% rb_*_grid is [Nz x Nx] = [30 x 60] on (z_mcx, x_mcx); x_mcx(30) = 29.5
z_rb = z_mcx;
rb_amp_line   = rb_amp_grid(:, 30)';
rb_phase_line = rb_phase_grid(:, 30)';
rb_cw_line    = rb_cw_grid(:, 30)';

% Plot
subplot(2, 2, 1);
semilogy(z_line, cw_mcx_line, 'b-', 'LineWidth', 1.5);
hold on;
semilogy(z_rb, rb_cw_line, 'r--', 'LineWidth', 1.5);
legend('MCX CW', 'Redbird CW', 'Location', 'northeast');
ylabel('Fluence');
xlabel('Depth z (mm)');
title('CW Fluence');
grid on;

subplot(2, 2, 2);
semilogy(z_line, mcx_amp_line, 'b-', 'LineWidth', 1.5);
hold on;
semilogy(z_rb, rb_amp_line, 'r--', 'LineWidth', 1.5);
legend('MCX RF', 'Redbird FD', 'Location', 'northeast');
ylabel('Amplitude');
xlabel('Depth z (mm)');
title(sprintf('FD Amplitude (f=%g MHz)', freq / 1e6));
grid on;

subplot(2, 2, 3);
plot(z_line, mcx_phase_line, 'b-', 'LineWidth', 1.5);
hold on;
plot(z_rb, rb_phase_line, 'r--', 'LineWidth', 1.5);
legend('MCX RF', 'Redbird FD', 'Location', 'southwest');
ylabel('Phase (deg)');
xlabel('Depth z (mm)');
title(sprintf('FD Phase (f=%g MHz)', freq / 1e6));
grid on;

subplot(2, 2, 4);
mcx_ratio = mcx_amp_line ./ max(cw_mcx_line, eps);
rb_ratio  = rb_amp_line(:) ./ max(rb_cw_line(:), eps);
plot(z_line, mcx_ratio, 'b-', 'LineWidth', 1.5);
hold on;
plot(z_rb, rb_ratio, 'r--', 'LineWidth', 1.5);
legend('MCX', 'Redbird', 'Location', 'southwest');
ylabel('AC/DC ratio');
xlabel('Depth z (mm)');
title('Amplitude demodulation ratio');
grid on;
ylim([0 1.1]);

sgtitle_compat(gcf, sprintf('Depth profiles at (x=30, y=30)  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm, n=%.2f', ...
                            freq / 1e6, mua, musp, nmed));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Print summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========== Summary ==========\n');
fprintf('Frequency:     %.0f MHz\n', freq / 1e6);
fprintf('mua = %.4f/mm, musp = %.1f/mm, n = %.2f\n', mua, musp, nmed);
fprintf('\nAt detector (x=40, y=30, z=0):\n');
fprintf('  Redbird: amp = %.4e, phase = %.2f deg\n', abs(detval_rb), angle(detval_rb) * 180 / pi);

mcx_det = phi_mcx(40, 30, 1);
fprintf('  MCX:     amp = %.4e, phase = %.2f deg\n', abs(mcx_det), angle(mcx_det) * 180 / pi);
fprintf('  Amp ratio (MCX/Redbird):   %.4f\n', abs(mcx_det) / abs(detval_rb));
fprintf('  Phase diff (MCX-Redbird):  %.2f deg\n', ...
        angle(mcx_det) * 180 / pi - angle(detval_rb) * 180 / pi);

fprintf('\nNote: MCX (transport) and Redbird (diffusion) agree well in\n');
fprintf('the diffusive regime (far from source/boundary), with deviation\n');
fprintf('near the source where the diffusion approximation breaks down.\n');
fprintf('================================\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sgtitle_compat(figh, str)
    %% Octave/MATLAB-compatible substitute for sgtitle
    if exist('sgtitle') > 0
        sgtitle(str);
    else
        oldax = gca;
        axes('Parent', figh, 'Position', [0 0.96 1 0.04], 'Visible', 'off');
        text(0.5, 0.5, str, 'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'tex');
        axes(oldax);
    end
end
