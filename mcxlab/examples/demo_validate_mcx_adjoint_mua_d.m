%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dual-Jacobian comparison: Redbird (FEM diffusion) vs MCX (RF adjoint_mua_d)
%
% Demonstrates outputtype='adjoint_mua_d', which computes BOTH J_mua and J_D
% in a single MCX session (one set of forward/adjoint photon propagations).
%
%   J_mua(r) = -phi_src(r) * phi_det(r) * dV      [mua sensitivity]
%   J_D(r)   = -grad(phi_src) . grad(phi_det) * dV [diffusion-coeff sensitivity]
%
% Output: flux.data is complex [Nx, Ny, Nz, Ns*Nd, 2]
%   flux.data(:,:,:,:,1)  ->  J_mua  (same as outputtype='adjoint')
%   flux.data(:,:,:,:,2)  ->  J_D    (same as outputtype='adjoint_dcoeff')
%
% Both are validated against Redbird FEM diffusion.  Agreement is expected in
% the diffusive regime (away from source/boundary).
%
% Geometry: slab domain [0,60]x[0,60]x[0,30] mm, 1 source, 1 detector.
% See also: demo_validate_mcx_adjoint_jacobian.m
%           demo_validate_mcx_adjoint_jacobian_dcoeff.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg xcfg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Shared optical parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq  = 200e6;              % modulation frequency [Hz]
omega = 2 * pi * freq;      % angular frequency [rad/s]

mua  = 0.005;               % absorption coeff [1/mm]
musp = 1.0;                 % reduced scattering [1/mm]
g    = 0;                   % anisotropy (isotropic, for diffusion comparison)
nmed = 1.37;                % refractive index

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird FEM setup  (domain [0,60]x[0,60]x[0,30] mm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[cfg.node, cfg.elem] = meshgrid6(0:1:60, 0:1:60, 0:1:30);
cfg.elem(:, 1:4) = meshreorient(cfg.node(:, 1:3), cfg.elem(:, 1:4));
cfg.face = volface(cfg.elem);

cfg.seg = ones(size(cfg.elem, 1), 1);

cfg.srcpos = [20  29.5  0];     % source at top surface [mm]
cfg.srcdir = [0     0     1];   % pointing into medium (+z)
cfg.detpos = [40    30    0];   % detector at top surface [mm]
cfg.detdir = [0     0     1];   % pointing into medium (+z)

cfg.prop  = [0 0 1 1; mua musp g nmed];
cfg.omega = omega;              % non-zero omega: frequency-domain FEM

cfg = rbmeshprep(cfg);
sd  = rbsdmap(cfg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird: one FD solve -> extract both J_mua and J_D
%%
%%  rbrunforward solves from BOTH source and detector positions.
%%  phi_rb is [nnodes x 2]: col 1 = phi_src, col 2 = phi_det.
%%
%%  J_mua_rb = rbfemmatrix(...)    -> [1 x nnodes] complex  (mua sensitivity)
%%  J_D_rb   = rbjac(...) 3rd out  -> [1 x nnodes] complex  (D-coeff sensitivity)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Building Redbird FD system (f = %.0f MHz)...\n', freq / 1e6);
tic;
[detphi_rb, phi_rb] = rbrunforward(cfg, 'solverflag', {'qmr', 1e-7, 1000});
toc;

% mua Jacobian from Redbird (same formula as adjoint_mua MCX output)
J_mua_rb = rbfemmatrix(cfg, sd, phi_rb);          % [1 x nnodes] complex

% D-coeff Jacobian from Redbird (same formula as adjoint_dcoeff MCX output)
[~, ~, J_D_rb] = rbjac(sd, phi_rb, cfg.deldotdel, cfg.elem, cfg.evol);  % [1 x nnodes] complex

fprintf('Redbird: detphi = %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)\n', ...
        real(detphi_rb(1)), imag(detphi_rb(1)), ...
        abs(detphi_rb(1)), angle(detphi_rb(1)) * 180 / pi);
fprintf('J_mua_rb size: %s,  complex: %d\n', mat2str(size(J_mua_rb)), ~isreal(J_mua_rb));
fprintf('J_D_rb   size: %s,  complex: %d\n', mat2str(size(J_D_rb)),   ~isreal(J_D_rb));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   MCX: single session with outputtype='adjoint_mua_d'
%%
%%  One photon run computes both J_mua and J_D simultaneously.
%%  Output flux.data is complex [Nx, Ny, Nz, Ns*Nd, 2]:
%%    flux.data(:,:,:,:,1)  ->  J_mua   (point product of fluences)
%%    flux.data(:,:,:,:,2)  ->  J_D     (dot product of fluence gradients)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('mcxlab', 'file')
    error('mcxlab not found. Add mcxlab to your MATLAB path.');
end

xcfg.nphoton    = 1e8;
xcfg.vol        = uint8(ones(60, 60, 30));
xcfg.prop       = [0 0 1 1; mua musp g nmed];
xcfg.tstart     = 0;
xcfg.tend       = 5e-9;
xcfg.tstep      = 5e-9;
xcfg.autopilot  = 1;
xcfg.issrcfrom0 = 1;    % physical-mm coords; domain [0,60]x[0,60]x[0,30]
xcfg.isreflect  = 1;
xcfg.srctype    = 'pencil';
xcfg.srcpos     = [20  29.5  0];    % matches Redbird srcpos
xcfg.srcdir     = [0     0     1  0]; % +z direction, focallength=0
xcfg.detpos     = [40    30    0  1]; % matches Redbird detpos, radius=1 mm
xcfg.detdir     = [0     0     1  0]; % +z (reversed photons enter medium)
xcfg.omega      = omega;              % RF modulation frequency
xcfg.outputtype = 'adjoint_mua_d';   % single session: output J_mua + J_D together

fprintf('Running MCX RF adjoint_mua_d (f = %.0f MHz, %.0e photons)...\n', freq / 1e6, xcfg.nphoton);
tic;
flux = mcxlab(xcfg);
toc;

% flux.data is complex [60, 60, 30, 1, 2]
fprintf('flux.data size: %s,  complex: %d\n', mat2str(size(flux.data)), ~isreal(flux.data));

J_mua_mcx = squeeze(flux.data(:, :, :, :, 1));  % [60, 60, 30] complex  -- J_mua
J_D_mcx   = squeeze(flux.data(:, :, :, :, 2));  % [60, 60, 30] complex  -- J_D

fprintf('J_mua_mcx sum(real) = %.4e\n', sum(real(J_mua_mcx(:))));
fprintf('J_D_mcx   sum(real) = %.4e\n', sum(real(J_D_mcx(:))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Interpolate Redbird results onto MCX regular grid
%%   xz cross-section at y = 29.5 mm
%%   MCX voxel index 30 (1-based) has center at y = 29.5 (issrcfrom0=1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ym_idx = 30;              % MCX 1-based voxel index (center at y=29.5 mm)
x_mcx  = (0.5:1:59.5);   % voxel-center x coords [mm]
z_mcx  = (0.5:1:29.5);   % voxel-center z coords [mm]
[xi, zi] = meshgrid(x_mcx, z_mcx);

% MCX xz slices [Nz x Nx] (non-conjugate transpose)
J_mua_mcx_grid = squeeze(J_mua_mcx(:, ym_idx, :)).';  % [30 x 60]
J_D_mcx_grid   = squeeze(J_D_mcx(:,   ym_idx, :)).';  % [30 x 60]

% Redbird: cut at y=29.5 and interpolate Re/Im parts separately
J_mua_rb_nodes = J_mua_rb(1, :).';   % [nnodes x 1] complex
J_D_rb_nodes   = J_D_rb(1,   :).';   % [nnodes x 1] complex

[cutpos_rb, cutval_mua_re] = qmeshcut(cfg.elem, cfg.node, real(J_mua_rb_nodes), 'y=29.5');
[~,         cutval_mua_im] = qmeshcut(cfg.elem, cfg.node, imag(J_mua_rb_nodes), 'y=29.5');
[~,         cutval_D_re]   = qmeshcut(cfg.elem, cfg.node, real(J_D_rb_nodes),   'y=29.5');
[~,         cutval_D_im]   = qmeshcut(cfg.elem, cfg.node, imag(J_D_rb_nodes),   'y=29.5');

J_mua_rb_grid = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_mua_re, xi, zi) + ...
                1i * griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_mua_im, xi, zi);
J_D_rb_grid   = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_D_re, xi, zi) + ...
                1i * griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_D_im, xi, zi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 1: Amplitude maps -- J_mua (top row) and J_D (bottom row)
%%   Each row: MCX | Redbird | difference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Dual-Jacobian amplitude: J_mua and J_D', 'Position', [50 80 1400 750]);

titles_mua = {sprintf('MCX |J_{mua}| log_{10}  (f=%g MHz)', freq / 1e6), ...
              sprintf('Redbird |J_{mua}| log_{10}  (f=%g MHz)', freq / 1e6), ...
              'log_{10}|J_{mua}| diff (MCX - Redbird)'};
titles_D   = {sprintf('MCX |J_D| log_{10}  (f=%g MHz)', freq / 1e6), ...
              sprintf('Redbird |J_D| log_{10}  (f=%g MHz)', freq / 1e6), ...
              'log_{10}|J_D| diff (MCX - Redbird)'};

data_mua = {log10(abs(J_mua_mcx_grid) + eps), log10(abs(J_mua_rb_grid) + eps), ...
            log10(abs(J_mua_mcx_grid) + eps) - log10(abs(J_mua_rb_grid) + eps)};
data_D   = {log10(abs(J_D_mcx_grid) + eps), log10(abs(J_D_rb_grid) + eps), ...
            log10(abs(J_D_mcx_grid) + eps) - log10(abs(J_D_rb_grid) + eps)};

for k = 1:3
    subplot(2, 3, k);
    imagesc(x_mcx, z_mcx, data_mua{k});
    axis equal tight;
    colorbar;
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(titles_mua{k});

    subplot(2, 3, k + 3);
    imagesc(x_mcx, z_mcx, data_D{k});
    axis equal tight;
    colorbar;
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(titles_D{k});
end

sgtitle_compat(gcf, sprintf('Dual-Jacobian amplitude  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 2: Phase maps -- J_mua (top) and J_D (bottom)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Dual-Jacobian phase: J_mua and J_D', 'Position', [50 80 1400 750]);

titles_mua_ph = {sprintf('MCX angle(J_{mua}) (deg, f=%g MHz)', freq / 1e6), ...
                 sprintf('Redbird angle(J_{mua}) (deg, f=%g MHz)', freq / 1e6), ...
                 'Phase diff J_{mua}: MCX - Redbird (deg)'};
titles_D_ph   = {sprintf('MCX angle(J_D) (deg, f=%g MHz)', freq / 1e6), ...
                 sprintf('Redbird angle(J_D) (deg, f=%g MHz)', freq / 1e6), ...
                 'Phase diff J_D: MCX - Redbird (deg)'};

data_mua_ph = {angle(J_mua_mcx_grid) * 180 / pi, angle(J_mua_rb_grid) * 180 / pi, ...
               angle(J_mua_mcx_grid) * 180 / pi - angle(J_mua_rb_grid) * 180 / pi};
data_D_ph   = {angle(J_D_mcx_grid) * 180 / pi, angle(J_D_rb_grid) * 180 / pi, ...
               angle(J_D_mcx_grid) * 180 / pi - angle(J_D_rb_grid) * 180 / pi};

for k = 1:3
    subplot(2, 3, k);
    imagesc(x_mcx, z_mcx, data_mua_ph{k});
    axis equal tight;
    colorbar;
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(titles_mua_ph{k});

    subplot(2, 3, k + 3);
    imagesc(x_mcx, z_mcx, data_D_ph{k});
    axis equal tight;
    colorbar;
    xlabel('x (mm)');
    ylabel('z (mm)');
    title(titles_D_ph{k});
end

sgtitle_compat(gcf, sprintf('Dual-Jacobian phase  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 3: Contour overlay -- banana shapes
%%   Left: J_mua,  Right: J_D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Dual-Jacobian contour overlay', 'Position', [50 50 1200 500]);

clines_amp = [-12:0.5:-6 -5.5:0.25:-4];
clines_phs = -180:10:180;

subplot(2, 2, 1);
contour(x_mcx, z_mcx, log10(abs(J_mua_rb_grid) + eps), clines_amp, 'r-',  'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, log10(abs(J_mua_mcx_grid) + eps), clines_amp, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('log_{10}|J_{mua}| contour (f=%g MHz)', freq / 1e6));
axis equal tight;

subplot(2, 2, 2);
contour(x_mcx, z_mcx, log10(abs(J_D_rb_grid) + eps), clines_amp, 'r-',  'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, log10(abs(J_D_mcx_grid) + eps), clines_amp, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('log_{10}|J_D| contour (f=%g MHz)', freq / 1e6));
axis equal tight;

subplot(2, 2, 3);
contour(x_mcx, z_mcx, angle(J_mua_rb_grid) * 180 / pi, clines_phs, 'r-',  'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, angle(J_mua_mcx_grid) * 180 / pi, clines_phs, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title('angle(J_{mua}) contour (deg)');
axis equal tight;

subplot(2, 2, 4);
contour(x_mcx, z_mcx, angle(J_D_rb_grid) * 180 / pi, clines_phs, 'r-',  'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, angle(J_D_mcx_grid) * 180 / pi, clines_phs, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title('angle(J_D) contour (deg)');
axis equal tight;

sgtitle_compat(gcf, 'Contour overlay: Redbird (red solid) vs MCX adjoint_mua_d (blue dashed)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 4: Depth profiles at x = 29.5 mm (column 30)
%%   Top: J_mua,  Bottom: J_D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Dual-Jacobian depth profiles', 'Position', [100 100 1100 700]);

J_mua_mcx_line = J_mua_mcx_grid(:, 30);   % [Nz x 1] complex
J_mua_rb_line  = J_mua_rb_grid(:,  30);
J_D_mcx_line   = J_D_mcx_grid(:, 30);
J_D_rb_line    = J_D_rb_grid(:,  30);

subplot(2, 2, 1);
semilogy(z_mcx, abs(J_mua_mcx_line) / max(abs(J_mua_mcx_line)), 'b-',  'LineWidth', 1.5);
hold on;
semilogy(z_mcx, abs(J_mua_rb_line)  / max(abs(J_mua_rb_line)),  'r--', 'LineWidth', 1.5);
legend('MCX adjoint\_mua\_d', 'Redbird FEM', 'Location', 'northeast');
xlabel('Depth z (mm)');
ylabel('|J_{mua}| (normalised)');
title('J_{mua} amplitude profile at x=29.5 mm');
grid on;

subplot(2, 2, 2);
plot(z_mcx, angle(J_mua_mcx_line) * 180 / pi, 'b-',  'LineWidth', 1.5);
hold on;
plot(z_mcx, angle(J_mua_rb_line)  * 180 / pi, 'r--', 'LineWidth', 1.5);
legend('MCX adjoint\_mua\_d', 'Redbird FEM', 'Location', 'southwest');
xlabel('Depth z (mm)');
ylabel('Phase (deg)');
title('J_{mua} phase profile at x=29.5 mm');
grid on;

subplot(2, 2, 3);
semilogy(z_mcx, abs(J_D_mcx_line) / max(abs(J_D_mcx_line)), 'b-',  'LineWidth', 1.5);
hold on;
semilogy(z_mcx, abs(J_D_rb_line)  / max(abs(J_D_rb_line)),  'r--', 'LineWidth', 1.5);
legend('MCX adjoint\_mua\_d', 'Redbird FEM', 'Location', 'northeast');
xlabel('Depth z (mm)');
ylabel('|J_D| (normalised)');
title('J_D amplitude profile at x=29.5 mm');
grid on;

subplot(2, 2, 4);
plot(z_mcx, angle(J_D_mcx_line) * 180 / pi, 'b-',  'LineWidth', 1.5);
hold on;
plot(z_mcx, angle(J_D_rb_line)  * 180 / pi, 'r--', 'LineWidth', 1.5);
legend('MCX adjoint\_mua\_d', 'Redbird FEM', 'Location', 'southwest');
xlabel('Depth z (mm)');
ylabel('Phase (deg)');
title('J_D phase profile at x=29.5 mm');
grid on;

sgtitle_compat(gcf, sprintf('Depth profiles at x=29.5 mm  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Print summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========== Summary ==========\n');
fprintf('Frequency:  %.0f MHz\n', freq / 1e6);
fprintf('mua=%.4f/mm,  musp=%.1f/mm,  n=%.2f\n', mua, musp, nmed);
fprintf('\nRedbird detector value:  %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)\n', ...
        real(detphi_rb(1)), imag(detphi_rb(1)), abs(detphi_rb(1)), angle(detphi_rb(1)) * 180 / pi);
fprintf('\nJ_mua_rb  sum(real) = %.4e\n', sum(real(J_mua_rb(1, :))));
fprintf('J_mua_mcx sum(real) = %.4e\n', sum(real(J_mua_mcx(:))));
fprintf('\nJ_D_rb    sum(real) = %.4e\n', sum(real(J_D_rb(1, :))));
fprintf('J_D_mcx   sum(real) = %.4e\n', sum(real(J_D_mcx(:))));
fprintf('================================\n');
fprintf('Note: MCX (transport, RTE) and Redbird (diffusion equation)\n');
fprintf('agree well in the diffusive regime (far from source/boundary).\n');
fprintf('Both Jacobians were computed from a SINGLE MCX session using\n');
fprintf('outputtype=''adjoint_mua_d'' (flux.data size: [Nx,Ny,Nz,Ns*Nd,2]).\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Helper function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sgtitle_compat(figh, str)
    %% Octave/MATLAB-compatible sgtitle substitute
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
