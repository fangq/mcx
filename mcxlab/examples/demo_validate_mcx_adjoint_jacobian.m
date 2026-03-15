%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency-Domain Jacobian comparison: Redbird (FEM diffusion) vs MCX (RF adjoint MC)
%
% Both compute the adjoint Jacobian J(r) = phi_src(r) * phi_det(r) per voxel,
% where phi_src is the forward fluence from the source and phi_det is the
% (adjoint) forward fluence launched from the detector position.
%
% Redbird: J_rb(r) from rbfemmatrix, complex [1 x nnodes]
% MCX:     J_mcx(r) from outputtype='adjoint' + omega>0,
%          output is complex [Nx,Ny,Nz] = [60,60,30]
%
% Agreement is expected in the diffusive regime (away from source/boundary).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg xcfg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Shared optical parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq  = 200e6;              % modulation frequency [Hz]
omega = 2 * pi * freq;      % angular frequency [rad/s]

mua  = 0.005;               % absorption coeff [1/mm]
musp = 1.0;                 % reduced scattering [1/mm]
g    = 0;                   % anisotropy (isotropic)
nmed = 1.37;                % refractive index

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird FEM setup  (domain [0,60]x[0,60]x[0,30] mm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[cfg.node, cfg.elem] = meshgrid6(0:1:60, 0:1:60, 0:1:30);
cfg.elem(:, 1:4) = meshreorient(cfg.node(:, 1:3), cfg.elem(:, 1:4));
cfg.face = volface(cfg.elem);

cfg.seg = ones(size(cfg.elem, 1), 1);

cfg.srcpos = [20  29.5  0];   % source at top surface, physical mm
cfg.srcdir = [0     0     1];   % pointing into medium (+z)
cfg.detpos = [40    30    0];   % detector at top surface
cfg.detdir = [0     0     1];   % pointing into medium (+z)

cfg.prop  = [0 0 1 1; mua musp g nmed];
cfg.omega = omega;              % non-zero omega: frequency-domain

cfg = rbmeshprep(cfg);
sd  = rbsdmap(cfg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Redbird: solve FD forward + compute Jacobian
%%
%%  rbrunforward solves from BOTH source and detector positions.
%%  phi_rb is [nnodes x 2]: col 1 = phi from src, col 2 = phi from det.
%%  J_rb = rbfemmatrix(cfg, sd, phi_rb) returns [1 x nnodes], complex.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Building Redbird FD system (f = %.0f MHz)...\n', freq / 1e6);
tic;
[detphi_rb, phi_rb] = rbrunforward(cfg, 'solverflag', {'qmr', 1e-7, 1000});
toc;

% J_rb: node-based Jacobian, size [nsd_pairs x nnodes] = [1 x nnodes], complex
J_rb = rbfemmatrix(cfg, sd, phi_rb);

fprintf('Redbird: detphi = %.4e + %.4ei  (amp=%.4e, phase=%.2f deg)\n', ...
        real(detphi_rb(1)), imag(detphi_rb(1)), ...
        abs(detphi_rb(1)), angle(detphi_rb(1)) * 180 / pi);
fprintf('J_rb size: %s,  complex: %d\n', mat2str(size(J_rb)), ~isreal(J_rb));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   MCX RF adjoint setup  (same physical domain)
%%
%%  outputtype='adjoint' + omega>0:
%%    - all Ns+Nd sources (source + detector acting as reversed source)
%%      are simulated together (srcid=-1 internally)
%%    - adjoint kernel runs after normalization:
%%        J(r) = phi_src(r) * phi_det(r)   (complex product for RF)
%%    - output data is complex [Nx,Ny,Nz,Ns*Nd] = [60,60,30,1]
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
xcfg.outputtype = 'adjoint';          % compute J = phi_src * phi_det per voxel

fprintf('Running MCX RF adjoint (f = %.0f MHz, %.0e photons)...\n', freq / 1e6, xcfg.nphoton);
tic;
flux_adj = mcxlab(xcfg);
toc;

% flux_adj.data is complex [60, 60, 30, 1]:
%   real part = Re(J) = Re(phi_src)*Re(phi_det) - Im(phi_src)*Im(phi_det)
%   imag part = Im(J) = Re(phi_src)*Im(phi_det) + Im(phi_src)*Re(phi_det)
J_mcx = squeeze(flux_adj.data);   % [60, 60, 30] complex
fprintf('J_mcx size: %s,  complex: %d\n', mat2str(size(J_mcx)), ~isreal(J_mcx));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Extract xz cross-section at y = 29.5 mm
%%   MCX voxel index 30 (1-based) has center at y = 29.5 (issrcfrom0=1)
%%   Redbird qmeshcut slices at y = 29.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ym_idx   = 30;             % MCX voxel index (1-based), center at y = 29.5

% Voxel center coordinates
x_mcx = (0.5:1:59.5);     % [1 x 60]
z_mcx = (0.5:1:29.5);     % [1 x 30]

% MCX: extract xz slice, transpose to [Nz x Nx] for imagesc
J_mcx_xz = squeeze(J_mcx(:, ym_idx, :));   % [Nx=60, Nz=30] complex
J_mcx_grid = J_mcx_xz.';                   % [Nz=30, Nx=60] complex  (.' = non-conj transpose)

% Redbird: cut mesh at y = 29.5 -- extract real and imaginary parts separately
J_rb_nodes = J_rb(1, :).';    % [nnodes x 1] complex (single src-det pair)

[cutpos_rb, cutval_Jre] = qmeshcut(cfg.elem, cfg.node, real(J_rb_nodes), 'y=29.5');
[~,         cutval_Jim] = qmeshcut(cfg.elem, cfg.node, imag(J_rb_nodes), 'y=29.5');

% Interpolate onto MCX regular grid  (xi,zi are [Nz x Nx] = [30 x 60])
[xi, zi] = meshgrid(x_mcx, z_mcx);
J_rb_grid = griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_Jre, xi, zi) + ...
            1i * griddata(cutpos_rb(:, 1), cutpos_rb(:, 3), cutval_Jim, xi, zi);
% J_rb_grid: [Nz=30, Nx=60] complex

J_mcx_norm = J_mcx_grid;
J_rb_norm = J_rb_grid;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 1: Jacobian amplitude maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'RF Jacobian Amplitude', 'Position', [50 100 1400 430]);
amp_clim = [-4  0];

subplot(1, 3, 1);
imagesc(x_mcx, z_mcx, log10(abs(J_mcx_norm) + eps));
axis equal tight;
colorbar; % caxis(amp_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('MCX adj |J| log_{10} (f=%g MHz)', freq / 1e6));

subplot(1, 3, 2);
imagesc(x_mcx, z_mcx, log10(abs(J_rb_norm) + eps));
axis equal tight;
colorbar; % caxis(amp_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Redbird FEM |J| log_{10} (f=%g MHz)', freq / 1e6));

subplot(1, 3, 3);
imagesc(x_mcx, z_mcx, log10(abs(J_mcx_norm) + eps) - log10(abs(J_rb_norm) + eps));
axis equal tight;
colorbar;
xlabel('x (mm)');
ylabel('z (mm)');
title('log_{10}|J| difference (MCX - Redbird)');

sgtitle_compat(gcf, sprintf('RF adjoint |J| comparison  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 2: Jacobian phase maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'RF Jacobian Phase', 'Position', [50 100 1400 430]);
phs_clim = [-360  0];

subplot(1, 3, 1);
imagesc(x_mcx, z_mcx, angle(J_mcx_grid) * 180 / pi);
axis equal tight;
colorbar; % caxis(phs_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('MCX adj angle(J) (deg, f=%g MHz)', freq / 1e6));

subplot(1, 3, 2);
imagesc(x_mcx, z_mcx, angle(J_rb_grid) * 180 / pi);
axis equal tight;
colorbar; % caxis(phs_clim);
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Redbird FEM angle(J) (deg, f=%g MHz)', freq / 1e6));

subplot(1, 3, 3);
diff_phase = angle(J_mcx_grid) * 180 / pi - angle(J_rb_grid) * 180 / pi;
imagesc(x_mcx, z_mcx, diff_phase);
axis equal tight;
colorbar;
xlabel('x (mm)');
ylabel('z (mm)');
title('Phase difference MCX - Redbird (deg)');

sgtitle_compat(gcf, sprintf('RF adjoint phase(J) comparison  |  f=%g MHz', freq / 1e6));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 3: Contour overlay -- banana shape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Contour Overlay', 'Position', [50 50 1200 500]);

subplot(1, 2, 1);
clines_amp = [-12:0.5:-6 -5.5:0.25:-4];
contour(x_mcx, z_mcx, log10(abs(J_rb_norm) + eps), clines_amp, 'r-', 'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, log10(abs(J_mcx_norm) + eps), clines_amp, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('log_{10}|J| contour overlay (f=%g MHz)', freq / 1e6));
axis equal tight;

subplot(1, 2, 2);
clines_phs = 0:10:270;
contour(x_mcx, z_mcx, angle(J_rb_grid) * 180 / pi, clines_phs, 'r-', 'LineWidth', 2);
hold on;
contour(x_mcx, z_mcx, angle(J_mcx_grid) * 180 / pi, clines_phs, 'b--', 'LineWidth', 2);
legend('Redbird FEM', 'MCX adjoint', 'Location', 'northeast');
xlabel('x (mm)');
ylabel('z (mm)');
title('angle(J) contour overlay (deg)');
axis equal tight;

sgtitle_compat(gcf, 'Contour overlay: Redbird (red solid) vs MCX adjoint (blue dashed)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Figure 4: Depth profiles at x = 29.5 mm (source x)
%%   Grid column index 30 = x_mcx(30) = 29.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'Depth Profiles', 'Position', [100 100 900 500]);

J_mcx_line = J_mcx_grid(:, 30);   % [Nz x 1] complex
J_rb_line  = J_rb_grid(:,  30);   % [Nz x 1] complex

subplot(1, 2, 1);
semilogy(z_mcx, abs(J_mcx_line) / max(abs(J_mcx_line)), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(z_mcx, abs(J_rb_line)  / max(abs(J_rb_line)),  'r--', 'LineWidth', 1.5);
legend('MCX RF adjoint', 'Redbird FEM', 'Location', 'northeast');
xlabel('Depth z (mm)');
ylabel('|J');
title('Amplitude depth profile at x=29.5');
grid on;

subplot(1, 2, 2);
plot(z_mcx, angle(J_mcx_line) * 180 / pi, 'b-',  'LineWidth', 1.5);
hold on;
plot(z_mcx, angle(J_rb_line)  * 180 / pi, 'r--', 'LineWidth', 1.5);
legend('MCX RF adjoint', 'Redbird FEM', 'Location', 'southwest');
xlabel('Depth z (mm)');
ylabel('Phase (deg)');
title('Phase depth profile at x=29.5');
grid on;

sgtitle_compat(gcf, sprintf('Depth profiles at x=29.5 mm  |  f=%g MHz, \\mu_a=%.3f/mm, \\mu_s''=%.1f/mm', ...
                            freq / 1e6, mua, musp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Print summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========== Summary ==========\n');
fprintf('Frequency:  %.0f MHz\n', freq / 1e6);
fprintf('mua=%.4f/mm,  musp=%.1f/mm,  n=%.2f\n', mua, musp, nmed);
fprintf('\nRedbird detector value (src->det):  %.4e + %.4ei\n', ...
        real(detphi_rb(1)), imag(detphi_rb(1)));
fprintf('  |amp| = %.4e,  phase = %.2f deg\n', ...
        abs(detphi_rb(1)), angle(detphi_rb(1)) * 180 / pi);

fprintf('\nJ_rb  sum(real) = %.4e\n', sum(real(J_rb(1, :))));
fprintf('J_mcx sum(real) = %.4e\n', sum(real(J_mcx(:))));
fprintf('================================\n');
fprintf('Note: MCX (transport, RTE) and Redbird (diffusion equation)\n');
fprintf('agree well in the diffusive regime (far from source/boundary).\n');
fprintf('Deviations near the source are expected.\n');

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
