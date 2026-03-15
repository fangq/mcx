%% demo_mcx_adjoint_banana.m
%
% Demonstrates adjoint Jacobian (banana-shaped sensitivity profiles) with
%   outputtype='adjoint'  -- Jacobian J(r) = phi_src(r) * phi_det(r)
%   srcid=-2              -- raw forward fluence for merged src+det set
%
% Both CW (omega=0) and RF frequency-domain (omega>0) cases are shown.
%
% Geometry (issrcfrom0=1, all positions in mm, voxel size 1 mm):
%   S1 at x=20, S2 at x=80  (pencil beam, pointing +z into slab)
%   D1 at x=38, D2 at x=62  (disk r=3 mm, same orientation)
%   Cross-section to plot: y = Ny/2, showing the x-z plane.

clear cfg cfg_adj cfg_rf cfg_fwd;

%% ---- Common geometry ------------------------------------------------
Nx = 100;
Ny = 60;
Nz = 50;
ym  = round(Ny / 2);          % mid-plane y position (0-based), use ym+1 for MATLAB indexing

cfg.nphoton    = 3e7;
cfg.vol        = uint8(ones(Nx, Ny, Nz));
cfg.issrcfrom0 = 1;            % positions are 0-based
cfg.gpuid      = 1;
cfg.autopilot  = 1;
cfg.tstart     = 0;
cfg.tend       = 5e-9;
cfg.tstep      = 5e-9;
cfg.unitinmm   = 1;
% [mua  mus   g    n]  (medium 0 = background, 1 = tissue)
cfg.prop       = [0 0 1 1; 0.005 1.0 0.9 1.37];

% Primary source S1 -- start at z=0.5 to be clearly inside voxel 0
cfg.srctype = 'pencil';
% Two pencil sources as rows: S1 (row 1) and S2 (row 2)
cfg.srcpos  = [20  ym  0.5
               80  ym  0.5];
cfg.srcdir  = [0   0   1   0
               0   0   1   0];

% Detectors: [x  y  z  radius_mm]  and matching directions
cfg.detpos = [38  ym  0.5  3
              62  ym  0.5  3];
cfg.detdir = [0  0  1  0
              0  0  1  0];

%% ---- Part 1: CW Adjoint Jacobian ------------------------------------
cfg_adj            = cfg;
cfg_adj.outputtype = 'adjoint';

fprintf('Running CW adjoint (Ns=2, Nd=2 -> 4 Jacobians) ...\n');
f_cw = mcxlab(cfg_adj);

% Output: real single [Nx, Ny, Nz, Ns*Nd]
%   (:,:,:,1) = J(S1,D1)   (:,:,:,2) = J(S1,D2)
%   (:,:,:,3) = J(S2,D1)   (:,:,:,4) = J(S2,D2)
J_cw = double(f_cw.data);
fprintf('  CW adjoint size: %s   range: [%.2e, %.2e]\n', ...
        mat2str(size(J_cw)), min(J_cw(:)), max(J_cw(:)));

%% ---- Part 2: RF Adjoint Jacobian (omega>0 triggers RF forward) ------
cfg_rf            = cfg;
cfg_rf.outputtype = 'adjoint';
cfg_rf.omega      = 2 * pi * 100e6;   % 100 MHz

fprintf('Running RF adjoint at 100 MHz ...\n');
f_rf = mcxlab(cfg_rf);

% Output: complex single [Nx, Ny, Nz, Ns*Nd]
J_rf = double(f_rf.data);
fprintf('  RF adjoint size: %s   complex: %d   |range|: [%.2e, %.2e]\n', ...
        mat2str(size(J_rf)), ~isreal(J_rf), min(abs(J_rf(:))), max(abs(J_rf(:))));

%% ---- Part 3: srcid=-2 merged forward fluences -----------------------
cfg_fwd       = cfg;
cfg_fwd.srcid = -2;

fprintf('Running srcid=-2 forward (Ns+Nd=4 merged sources) ...\n');
f_fwd = mcxlab(cfg_fwd);

% Output: [Nx, Ny, Nz, maxgate, 1, Ns+Nd]  -- reshape to 4-D
phi = double(reshape(f_fwd.data, Nx, Ny, Nz, []));
fprintf('  srcid=-2 fluence size (after reshape): %s   range: [%.2e, %.2e]\n', ...
        mat2str(size(phi)), min(phi(:)), max(phi(:)));
% phi(:,:,:,1) = S1 fluence    phi(:,:,:,2) = S2 fluence
% phi(:,:,:,3) = D1 fluence    phi(:,:,:,4) = D2 fluence

%% ---- Pair labels and source/detector x positions --------------------
pair_labels = {'J(S_1,D_1)', 'J(S_1,D_2)', 'J(S_2,D_1)', 'J(S_2,D_2)'};
sx = [20 80];   % source x positions (0-based)
dx = [38 62];   % detector x positions (0-based)
clim_log = [-10 0];

%% ---- Figure 1: all four CW banana Jacobians -------------------------
figure('Name', 'CW Adjoint Jacobian -- Banana Profiles', ...
       'Position', [50 50 1200 900]);
colormap hot;
for k = 1:4
    s = floor((k - 1) / 2) + 1;   % 1 or 2
    d = mod(k - 1, 2)   + 1;
    subplot(2, 2, k);
    plot_xz_slice(J_cw(:, :, :, k), ym + 1, pair_labels{k}, clim_log, sx(s), dx(d));
end
sgtitle('CW Adjoint Jacobians  (log_{10})   \color{red}\triangledown S   \color{blue}\squareD');

%% ---- Figure 2: CW vs RF for pair S1-D1 -----------------------------
figure('Name', 'CW vs RF Adjoint -- S1-D1', 'Position', [50 50 1300 380]);
colormap hot;

subplot(1, 3, 1);
plot_xz_slice(J_cw(:, :, :, 1), ym + 1, 'CW  J(S_1,D_1)', clim_log, sx(1), dx(1));

subplot(1, 3, 2);
plot_xz_slice(abs(J_rf(:, :, :, 1)), ym + 1, '|RF J(S_1,D_1)|  100 MHz', clim_log, sx(1), dx(1));

subplot(1, 3, 3);
ph = squeeze(angle(J_rf(:, ym + 1, :, 1)))' * 180 / pi;
imagesc(0:Nx - 1, 0:Nz - 1, ph);
hold on;
plot(sx(1), 0, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
plot(dx(1), 0, 'bs', 'MarkerSize', 9, 'MarkerFaceColor', 'b');
xlabel('x (mm)');
ylabel('z (mm)');
title('Phase RF J(S_1,D_1)  (deg)');
colorbar;
colormap(gca, hsv);
set(gca, 'YDir', 'normal');

sgtitle('CW vs RF (100 MHz) Adjoint Jacobian -- Pair S_1-D_1');

%% ---- Figure 3: individual fluences from srcid=-2 --------------------
figure('Name', 'srcid=-2: Fluence volumes (merged src+det)', ...
       'Position', [50 500 1200 500]);
colormap hot;
vol_labels = {'S_1 fluence', 'S_2 fluence', 'D_1 adj-src fluence', 'D_2 adj-src fluence'};
marker_x   = {sx(1), sx(2), dx(1), dx(2)};
marker_sym = {'r^',  'r^',  'bs',  'bs'};
for k = 1:4
    subplot(2, 2, k);
    slice = squeeze(phi(:, ym + 1, :, k))';
    imagesc(0:Nx - 1, 0:Nz - 1, log10(max(slice, 10^clim_log(1))));
    hold on;
    plot(marker_x{k}, 0, marker_sym{k}, 'MarkerSize', 9, 'MarkerFaceColor', marker_sym{k}(1));
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(vol_labels{k});
    colorbar;
    set(gca, 'YDir', 'normal');
end
sgtitle('srcid=-2: Forward fluence for each merged source/detector');

%% ---- Figure 4: consistency check  J_CW ≈ phi_S1 * phi_D1 -----------
figure('Name', 'Consistency: J vs phi_S * phi_D', 'Position', [50 50 1200 380]);
colormap hot;

Jprod = phi(:, :, :, 1) .* phi(:, :, :, 3);   % S1 × D1 direct product

subplot(1, 2, 1);
plot_xz_slice(J_cw(:, :, :, 1), ym + 1, 'J(S_1,D_1)  adjoint mode', clim_log, sx(1), dx(1));

subplot(1, 2, 2);
plot_xz_slice(Jprod, ym + 1, '\phi_{S1}\cdot\phi_{D1}  srcid=-2 product', clim_log, sx(1), dx(1));

sgtitle('Adjoint consistency: J(S_1,D_1) vs \phi_{S1}\cdot\phi_{D1}');
fprintf('\nDone.\n');

%% ---- Local helper: plot log10 of xz slice with source/det markers ---
function plot_xz_slice(vol3d, ys, ttl, clim_log, src_x, det_x)
    Nx_  = size(vol3d, 1);
    Nz_  = size(vol3d, 3);
    sl   = squeeze(vol3d(:, ys, :))';   % [Nz, Nx]
    imagesc(0:Nx_ - 1, 0:Nz_ - 1, log10(max(double(abs(sl)), 10^clim_log(1))));
    % clim(clim_log);
    hold on;
    for k = 1:numel(src_x)
        plot(src_x(k), 0, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
    end
    for k = 1:numel(det_x)
        plot(det_x(k), 0, 'bs', 'MarkerSize', 9, 'MarkerFaceColor', 'b');
    end
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(ttl);
    colorbar;
    set(gca, 'YDir', 'normal');
end
