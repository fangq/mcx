function test_mcxlab(varargin)
%
% test_mcxlab
% test_mcxlab('gpuid', 2)
%
% Unit tests of the mcxlab interface, for MATLAB and GNU Octave: the cfg
% parsing, the errors, the outputs and their sizes, and short absorbed-
% fraction end-to-end checks. Each simulation is small (a 20^3 domain,
% 1e4-1e5 photons), so the suite takes seconds; an error is raised if any
% test fails
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
tests = {@test_info, @test_errors, @test_outputs, @test_detected_photons, ...
         @test_no_fluence, @test_struct_array, @test_patterns, @test_rf, ...
         @test_replay, @test_energy_output, @test_flog_silent, ...
         @test_end_to_end, @test_pervoxel_unitinmm, @test_input_types, ...
         @test_launch_angles, @test_polarized};
failed = {};
for i = 1:numel(tests)
    name = func2str(tests{i});
    try
        tests{i}(extra);
        fprintf(1, '  %-24s ok\n', name);
    catch err
        fprintf(1, '  %-24s FAILED: %s\n', name, err.message);
        failed{end + 1} = [name ': ' err.message];
    end
end

if (~isempty(failed))
    error('%d of %d mcxlab tests failed:\n  %s', numel(failed), numel(tests), strjoin(failed, '\n  '));
end
fprintf(1, 'all %d mcxlab tests passed\n', numel(tests));

%% -------------------------------------------------------------------------
function varargout = runsim(varargin)
% the simulator under test; the mcxlabcl copy of this file calls mcxlabcl
[varargout{1:max(1, nargout)}] = mcxlab(varargin{:});

%% -------------------------------------------------------------------------
function cfg = mkcfg(extra, varargin)
% a 20^3 homogeneous cube with a pencil beam; name/value pairs override
cfg = struct('nphoton', 1e4, 'vol', uint8(ones(20, 20, 20)), 'tstart', 0, ...
             'tend', 5e-9, 'tstep', 5e-9, 'srcpos', [10 10 0], 'srcdir', [0 0 1], ...
             'prop', [0 0 1 1; 0.01 1 0.9 1.37], 'issrcfrom0', 1, 'seed', 12345);
if (ispc)
    cfg.flog = 'NUL';  % silence the log
else
    cfg.flog = '/dev/null';
end
opt = [fieldnames(extra), struct2cell(extra); reshape(varargin, 2, [])'];
for i = 1:size(opt, 1)
    cfg.(opt{i, 1}) = opt{i, 2};
end

function prop2 = twomedia()
prop2 = [0 0 1 1; 0.01 1 0.9 1.37; 0.05 5 0.8 1.37];

function a = absorbed(flux)
a = flux.stat.energyabs / flux.stat.energytot;

function assert_error(fun, msg)
try
    fun();
catch
    return
end
error('no error raised: %s', msg);

%% -------------------------------------------------------------------------
function test_info(extra)
ver = runsim('version');
assert(ischar(ver) && ver(1) == 'v', 'version must be a string starting with v');
info = runsim('gpuinfo');
assert(isstruct(info) && ~isempty(info), 'gpuinfo must return a struct array');
assert(all(isfield(info, {'name', 'id', 'devcount', 'globalmem', 'core'})), 'gpuinfo fields');

function test_errors(extra)
cfg = mkcfg(extra);
assert_error(@() runsim(rmfield(cfg, 'vol')), 'missing vol');
assert_error(@() runsim(rmfield(cfg, 'prop')), 'missing prop');
assert_error(@() runsim(mkcfg(extra, 'prop', [0 0 1; 0.01 1 0.9])), 'prop with 3 columns');
assert_error(@() runsim(mkcfg(extra, 'srctype', 'xyz')), 'unknown srctype');
assert_error(@() runsim(mkcfg(extra, 'outputtype', 'xyz')), 'unknown outputtype');

function test_outputs(extra)
% time gates, stat, the reflectance and the label volume with a shape drawn in
vol = uint8(ones(20, 20, 20));
vol(:, :, 1) = 0;  % the reflectance is recorded in the background voxels next to the domain
cfg = mkcfg(extra, 'vol', vol, 'tstep', 1e-9, 'issaveref', 1, 'prop', twomedia(), ...
            'shapes', '{"Shapes":[{"Sphere":{"O":[10,10,10],"R":4,"Tag":2}}]}');
[flux, detp, vol] = runsim(cfg);
assert(isequal(size(flux.data), [20 20 20 5]), 'fluence size [Nx Ny Nz Ngate]');
assert(all(isfield(flux.stat, {'runtime', 'nphoton', 'energytot', 'energyabs', 'normalizer', 'unitinmm', 'workload'})), 'stat fields');
assert(isequal(size(flux.dref), size(flux.data)) && sum(flux.dref(:)) > 0, 'dref');
assert(isequal(size(vol.data), [20 20 20]), 'label volume size');
assert(vol.data(11, 11, 11) == 2 && vol.data(5, 5, 5) == 1, 'the sphere must be drawn into the label volume');
% the beam enters at z=0, so the fluence decays along the 3rd dimension
assert(flux.data(11, 11, 3, 1) > flux.data(11, 11, 15, 1), 'fluence orientation');

function test_detected_photons(extra)
cfg = mkcfg(extra, 'detpos', [10 8 0 2], 'savedetflag', 'dspmxvw', 'nphoton', 2e4);
[flux, detp, vol, seeds, traj] = runsim(cfg);
ndet = numel(detp.detid);
assert(ndet > 0, 'no photon detected');
assert(all(detp.detid == 1), 'detector id');
assert(isequal(size(detp.ppath), [ndet 1]) && all(detp.ppath > 0), 'ppath: one column per medium');
assert(isequal(size(detp.nscat), [ndet 1]) && isequal(size(detp.mom), [ndet 1]), 'nscat/mom');
assert(isequal(size(detp.p), [ndet 3]) && isequal(size(detp.v), [ndet 3]), 'exit position and direction');
assert(isequal(detp.prop, cfg.prop), 'detp.prop');
assert(isa(seeds.data, 'uint8') && size(seeds.data, 2) == ndet, 'one seed column per detected photon');
assert(all(isfield(traj, {'id', 'pos', 'data'})) && size(traj.pos, 2) == 3, 'trajectory fields');
assert(issorted(double(traj.id)), 'trajectories sorted by photon id');
% the default savedetflag 'dp': the detector id and the partial path
[flux, detp] = runsim(mkcfg(extra, 'detpos', [10 8 0 2]));
assert(size(detp.data, 1) == 2, 'default savedetflag');

function test_no_fluence(extra)
[flux, detp] = runsim(mkcfg(extra, 'detpos', [10 8 0 2], 'issave2pt', 0));
assert(isempty(flux.data) || ~any(flux.data(:)), 'issave2pt=0 must skip the fluence');
assert(~isempty(detp.detid), 'issave2pt=0 must still detect photons');

function test_struct_array(extra)
% cfgs(1).isreflect is empty, which keeps the default (1); it was read as 0
cfgs = mkcfg(extra, 'prop', [0 0 1 1; 0.05 2 0.8 1.37], 'nphoton', 2e4);
cfgs(2) = cfgs(1);
cfgs(2).isreflect = 0;
flux = runsim(cfgs);
assert(numel(flux) == 2, 'one output per cfg element');
% Fresnel reflection keeps more light inside the domain
assert(absorbed(flux(1)) > absorbed(flux(2)) + 0.05, 'isreflect of each element');

function test_patterns(extra)
% several patterns go to the last dimension, without mixing up the voxels
pat = ones(2, 4, 4);
pat(2, :, 1:2) = 0;  % the 2nd pattern is dark for low y
cfg = mkcfg(extra, 'nphoton', 5e4, 'srctype', 'pattern', 'srcpos', [5 5 -0.01], ...
            'srcparam1', [8 0 0 4], 'srcparam2', [0 8 0 4], 'srcpattern', pat, 'srcnum', 2);
flux = runsim(cfg);
data = reshape(flux.data, 20, 20, 20, []);  % [Nx Ny Nz 1 2], shown as [Nx Ny Nz 2] for one gate
assert(numel(flux.data) == 20^3 * 2, 'patterns in the last dimension');
f1 = data(:, :, :, 1);
f2 = data(:, :, :, 2);
lo = sum(sum(sum(f2(:, 1:10, :)))) / sum(sum(sum(f1(:, 1:10, :))));
hi = sum(sum(sum(f2(:, 11:end, :)))) / sum(sum(sum(f1(:, 11:end, :))));
assert(lo < hi, 'the dark half of pattern 2 must be at low y');
lox = sum(sum(sum(f2(1:10, :, :)))) / sum(sum(sum(f1(1:10, :, :))));
hix = sum(sum(sum(f2(11:end, :, :)))) / sum(sum(sum(f1(11:end, :, :))));
assert(abs(lox / hix - 1) < 0.05, 'pattern 2 must be symmetric in x');

function test_rf(extra)
flux = runsim(mkcfg(extra, 'omega', 2 * pi * 1e8, 'outputtype', 'rf'));
assert(~isreal(flux.data) || size(flux.data, 6) == 2, 'RF output must carry the imaginary part');

function test_replay(extra)
cfg = mkcfg(extra, 'detpos', [10 8 0 2], 'nphoton', 2e4);
[flux, detp, vol, seeds] = runsim(cfg);
cfg.seed = seeds.data;
cfg.detphotons = detp.data;
cfg.outputtype = 'jacobian';
jac = runsim(cfg);
assert(isequal(size(jac.data), [20 20 20]) || isequal(size(jac.data), [20 20 20 1]), 'Jacobian size');
assert(all(isfinite(jac.data(:))) && sum(abs(jac.data(:))) > 0, 'Jacobian values');

function test_energy_output(extra)
% the normalized 'energy' output adds up to energyabs/energytot
flux = runsim(mkcfg(extra, 'outputtype', 'energy'));
assert(abs(sum(double(flux.data(:))) - absorbed(flux)) < 1e-4, 'energy sum');

function test_flog_silent(extra)
% flog set to the null device prints nothing, not even the cfg echo
if (~exist('evalc'))
    return
end
cfg = mkcfg(extra, 'nphoton', 1e3);
out = evalc('runsim(cfg);');
assert(isempty(strtrim(out)), ['unexpected output: ' out]);

function test_end_to_end(extra)
% absorbed fractions against 1e7-photon references (mcx and mcx-cl agree within 5e-5)
flux = runsim(mkcfg(extra, 'nphoton', 1e5));
assert(abs(absorbed(flux) - 0.2440) < 0.01, sprintf('homogeneous cube: %.4f', absorbed(flux)));
vol = uint8(ones(20, 20, 20));
vol(:, :, 9:end) = 2;
flux = runsim(mkcfg(extra, 'nphoton', 1e5, 'vol', vol, 'prop', twomedia(), 'isreflect', 0));
assert(abs(absorbed(flux) - 0.3972) < 0.01, sprintf('two layers: %.4f', absorbed(flux)));

function test_pervoxel_unitinmm(extra)
% the mua/mus (1/mm) of the per-voxel media formats are scaled by cfg.unitinmm
% exactly once, whether cfg.unitinmm is set before or after cfg.vol; each
% format describes the same homogeneous medium as the label volume
dim = [20 20 20];
mua = 0.01;
mus = 1;
prop = [0 0 1 1; mua mus 0.9 1.37];
formats = {'mua_float', single(mua * ones([1 dim])), prop
           'muamus_float', single(repmat([mua; mus], [1 dim])), prop
           'asgn_float', single(repmat([mua; mus; 0.9; 1.37], [1 dim])), prop
           % labelplus: [value, property index, label]; index 0 replaces the mua of label 1, 1 its mus
           'labelplus_mua', single(repmat([mua; 0; 1], [1 dim])), [0 0 1 1; 0.05 mus 0.9 1.37]
           'labelplus_mus', single(repmat([mus; 1; 1], [1 dim])), [0 0 1 1; mua 5 0.9 1.37]};
ref = absorbed(runsim(mkcfg(extra, 'nphoton', 5e4, 'unitinmm', 2)));
for i = 1:size(formats, 1)
    % cfg.unitinmm after cfg.vol
    after = absorbed(runsim(mkcfg(extra, 'nphoton', 5e4, 'vol', formats{i, 2}, 'prop', formats{i, 3}, 'unitinmm', 2)));
    % cfg.unitinmm before cfg.vol
    cfg = struct('unitinmm', 2);
    base = mkcfg(extra, 'nphoton', 5e4, 'vol', formats{i, 2}, 'prop', formats{i, 3});
    names = fieldnames(base);
    for j = 1:numel(names)
        cfg.(names{j}) = base.(names{j});
    end
    before = absorbed(runsim(cfg));
    assert(abs(after - ref) < 0.01 && abs(before - ref) < 0.01, ...
           sprintf('%s: label %.4f, unitinmm last %.4f, first %.4f', formats{i, 1}, ref, after, before));
end

function test_input_types(extra)
% any numeric or logical class, MATLAB string scalars and column vectors are accepted
ref = absorbed(runsim(mkcfg(extra, 'nphoton', 5e4)));
cases = {'gpuid', int32(1); 'nphoton', int32(5e4); 'srcpos', [10; 10; 0]; 'srcdir', [0; 0; 1]; ...
         'vol', true(20, 20, 20); 'prop', single([0 0 1 1; 0.01 1 0.9 1.37]); 'isreflect', true};
for i = 1:size(cases, 1)
    cfg = mkcfg(extra, 'nphoton', 5e4, cases{i, 1}, cases{i, 2});
    if (isfield(extra, cases{i, 1}))  % keep a gpuid given to the test
        cfg.(cases{i, 1}) = extra.(cases{i, 1});
    end
    a = absorbed(runsim(cfg));
    assert(abs(a - ref) < 0.01, sprintf('%s as %s: %.4f vs %.4f', cases{i, 1}, class(cases{i, 2}), a, ref));
end
assert_error(@() runsim('notacommand'), 'an unknown command');
% replay with detphotons given as double
cfg = mkcfg(extra, 'detpos', [10 8 0 2], 'nphoton', 2e4);
[flux, detp, vol, seeds] = runsim(cfg);
cfg.seed = seeds.data;
cfg.detphotons = double(detp.data);
cfg.outputtype = 'jacobian';
jac = runsim(cfg);
assert(all(isfinite(jac.data(:))) && sum(abs(jac.data(:))) > 0, 'replay with double detphotons');

function test_launch_angles(extra)
% srcdir(4) = nan (isotropic) or -inf (Lambertian) works for every source type, the pencil
% beam included (mcx ignored it for pencil beams, mcx-cl did not)
pencil = absorbed(runsim(mkcfg(extra, 'nphoton', 1e5)));
lambert = absorbed(runsim(mkcfg(extra, 'nphoton', 1e5, 'srcdir', [0 0 1 -inf])));
disk = absorbed(runsim(mkcfg(extra, 'nphoton', 1e5, 'srcdir', [0 0 1 -inf], 'srctype', 'disk', 'srcparam1', [0.01 0 0 0])));
assert(lambert < pencil - 0.01 && abs(lambert - disk) < 0.01, sprintf('Lambertian %.4f, disk %.4f, pencil %.4f', lambert, disk, pencil));
iso = absorbed(runsim(mkcfg(extra, 'nphoton', 1e5, 'srcpos', [10 10 10], 'srcdir', [0 0 1 nan])));
ref = absorbed(runsim(mkcfg(extra, 'nphoton', 1e5, 'srcpos', [10 10 10], 'srctype', 'isotropic')));
assert(abs(iso - ref) < 0.01, sprintf('isotropic %.4f vs %.4f', iso, ref));

function test_polarized(extra)
% the source Stokes vector defaults to unpolarized (it was uninitialized, failing at random)
cfg = mkcfg(extra, 'polprop', [0.01 0.5 0.1 1.59 1.33], 'lambda', 633);
for i = 1:3
    flux = runsim(cfg);
    assert(isequal(size(flux.prop), [4 2]) && all(isfinite(flux.data(:))), 'polarized output');
end
