function test_mcxlab(varargin)
%
% test_mcxlab
% test_mcxlab('gpuid', 2)
%
% Regression tests of mcxlab, for MATLAB and GNU Octave; an error is raised
% if any test fails
%
% input:
%     optional name/value pairs added to the cfg of every test, e.g. 'gpuid'
%
% This file is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

extra = struct(varargin{:});
failed = {};

failed = [failed, test_pervoxel_unitinmm(extra)];

if (~isempty(failed))
    error('%d mcxlab test(s) failed:\n  %s', numel(failed), strjoin(failed, '\n  '));
end
fprintf(1, 'all mcxlab tests passed\n');

%% -------------------------------------------------------------------------
function failed = test_pervoxel_unitinmm(extra)
%
% the mua/mus (1/mm) of the per-voxel media formats are scaled by cfg.unitinmm
% exactly once, whether cfg.unitinmm is set before or after cfg.vol; each
% format describes the same homogeneous medium as a label volume, so the
% absorbed fractions must agree within the Monte Carlo noise

failed = {};
dim = [30 30 30];
mua = 0.01;
mus = 1;
prop = [0 0 1 1; mua mus 0.9 1.37];
unitinmm = 2;

% per-voxel formats of the same medium
formats = struct('name', {}, 'vol', {});
formats(end + 1) = struct('name', 'mua_float', 'vol', single(mua * ones([1 dim])));
formats(end + 1) = struct('name', 'muamus_float', 'vol', single(repmat([mua; mus], [1 dim])));
formats(end + 1) = struct('name', 'asgn_float', 'vol', single(repmat([mua; mus; 0.9; 1.37], [1 dim])));
% labelplus: [value, property index, label]; index 0 replaces the mua of label 1, 1 its mus
formats(end + 1) = struct('name', 'labelplus_mua', 'vol', single(repmat([mua; 0; 1], [1 dim])));
formats(end + 1) = struct('name', 'labelplus_mus', 'vol', single(repmat([mus; 1; 1], [1 dim])));

ref = run_absorbed(struct('vol', uint8(ones(dim)), 'prop', prop, 'unitinmm', unitinmm), extra);
for i = 1:numel(formats)
    vprop = prop;
    if (strcmp(formats(i).name, 'labelplus_mua'))
        vprop(2, 1) = 0.05;  % must be replaced by the per-voxel value
    elseif (strcmp(formats(i).name, 'labelplus_mus'))
        vprop(2, 2) = 5;
    end
    % cfg.vol before cfg.unitinmm
    after = run_absorbed(struct('vol', formats(i).vol, 'prop', vprop, 'unitinmm', unitinmm), extra);
    % cfg.unitinmm before cfg.vol
    before = run_absorbed(struct('unitinmm', unitinmm, 'vol', formats(i).vol, 'prop', vprop), extra);
    ok = abs(after - ref) < 0.01 && abs(before - ref) < 0.01;
    fprintf(1, '  %-13s absorbed: label %.4f, unitinmm last %.4f, unitinmm first %.4f ... %s\n', ...
            formats(i).name, ref, after, before, passfail(ok));
    if (~ok)
        failed{end + 1} = ['per-voxel unitinmm scaling: ' formats(i).name];
    end
end

%% -------------------------------------------------------------------------
function absorbed = run_absorbed(cfg, extra)
% run a pencil beam into the domain of cfg, return the absorbed fraction
base = struct('nphoton', 2e5, 'tstart', 0, 'tend', 5e-9, 'tstep', 5e-9, ...
              'srcpos', [15 15 0], 'srcdir', [0 0 1], 'issrcfrom0', 1, ...
              'seed', 12345);
if (ispc)
    base.flog = 'NUL';  % silence the log
else
    base.flog = '/dev/null';
end
names = [fieldnames(base); fieldnames(extra)];
vals = [struct2cell(base); struct2cell(extra)];
for i = 1:numel(names)
    cfg.(names{i}) = vals{i};
end
flux = mcxlab(cfg);
absorbed = flux.stat.energyabs / flux.stat.energytot;

%% -------------------------------------------------------------------------
function str = passfail(ok)
if (ok)
    str = 'ok';
else
    str = 'FAILED';
end
