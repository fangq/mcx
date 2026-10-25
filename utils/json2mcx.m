function cfg = json2mcx(filename)
%
% Format:
%    cfg=json2mcx(filename)
%
% Convert a JSON file for MCX binary to an MCXLAB configuration structure
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Input:
%    filename: the JSON input file
%
% Output:
%    cfg: a struct defining the parameters associated with a simulation.
%         Please run 'help mcxlab' or 'help mmclab' to see details.
%
% Dependency:
%    this function depends on the savejson/loadjson functions from the
%    Iso2Mesh toolbox (http://iso2mesh.sf.net) or JSONlab toolbox
%    (http://iso2mesh.sf.net/jsonlab)
%
% This function is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

if (ischar(filename))
    json = loadjson(filename);
elseif (isstruct(filename))
    json = filename;
else
    error('first input is not supported');
end

%% define the optodes: sources and detectors

cfg = struct();
if (isfield(json, 'Optode'))
    if (isfield(json.Optode, 'Source'))
        cfg = copycfg(cfg, 'srcpos', json.Optode.Source, 'Pos');
        cfg = copycfg(cfg, 'srcdir', json.Optode.Source, 'Dir');
        if (isfield(cfg, 'srcdir'))  % normalize the direction of each source
            cfg.srcdir(:, 1:3) = cfg.srcdir(:, 1:3) ./ repmat(sqrt(sum(cfg.srcdir(:, 1:3).^2, 2)), 1, 3);
        end
        cfg = copycfg(cfg, 'srcparam1', json.Optode.Source, 'Param1');
        cfg = copycfg(cfg, 'srcparam2', json.Optode.Source, 'Param2');
        cfg = copycfg(cfg, 'srctype', json.Optode.Source, 'Type');
        cfg = copycfg(cfg, 'srcnum', json.Optode.Source, 'SrcNum');
        cfg = copycfg(cfg, 'srcid', json.Optode.Source, 'ID');
        cfg = copycfg(cfg, 'srciquv', json.Optode.Source, 'IQUV');
        cfg = copycfg(cfg, 'lambda', json.Optode.Source, 'WaveLength');
        cfg = copycfg(cfg, 'angleinvcdf', json.Optode.Source, 'AngleInverseCDF');
        if (isfield(json.Optode.Source, 'Frequency'))
            cfg.omega = 2 * pi * json.Optode.Source.Frequency;  % Frequency is in Hz, cfg.omega in rad/s
        end
        if (isfield(json.Optode.Source, 'Pattern') && isnumeric(json.Optode.Source.Pattern))
            cfg.srcpattern = json.Optode.Source.Pattern;  % a JData array, decoded by loadjson
        elseif (isfield(json.Optode.Source, 'Pattern') && isstruct(json.Optode.Source.Pattern))
            nz = jsonopt('Nz', 1, json.Optode.Source.Pattern);
            if (ischar(json.Optode.Source.Pattern.Data))
                fid = fopen(json.Optode.Source.Pattern.Data, 'rb');
                json.Optode.Source.Pattern.Data = fread(fid, inf, 'single=>single');
            end
            cfg.srcpattern = reshape(json.Optode.Source.Pattern.Data, ...
                                     [json.Optode.Source.Pattern.Nx, json.Optode.Source.Pattern.Ny, nz]);
        end
    end
    if (isfield(json.Optode, 'Detector') && ~isempty(json.Optode.Detector))
        det = json.Optode.Detector;
        if (iscell(det))
            det = cell2mat(det);
        end
        if (isfield(det, 'R'))
            detr = [det.R]';
        else
            detr = ones(numel(det), 1);
        end
        cfg.detpos = [reshape([det.Pos], 3, [])', detr];
    end
end

%% define the domain and optical properties

cfg = copycfg(cfg, 'issrcfrom0', json.Domain, 'OriginType');
cfg = copycfg(cfg, 'unitinmm', json.Domain, 'LengthUnit');
cfg = copycfg(cfg, 'invcdf', json.Domain, 'InverseCDF');
if (~isfield(cfg, 'angleinvcdf'))  % older mcx2json versions saved it in Domain
    cfg = copycfg(cfg, 'angleinvcdf', json.Domain, 'AngleInverseCDF');
end
if (isfield(json.Domain, 'MieScatter'))
    mie = json.Domain.MieScatter;
    if (iscell(mie))
        mie = cell2mat(mie);
    end
    cfg.polprop = [[mie.mua]', [mie.radius]', [mie.rho]', [mie.nsph]', [mie.nmed]'];
end

if (iscell(json.Domain.Media))
    json.Domain.Media = cell2mat(json.Domain.Media);
end
cfg.prop = squeeze(cell2mat(struct2cell(json.Domain.Media)))';

if (isfield(json, 'Shapes') && isnumeric(json.Shapes))
    % mcx2json saves cfg.vol as a JData array in Shapes, with the leading
    % (per-voxel record) dimension of a 4-D volume moved to the end
    cfg.vol = json.Shapes;
    mediaformat = '';
    if (isfield(json.Domain, 'MediaFormat'))
        mediaformat = lower(json.Domain.MediaFormat);
    end
    if (ndims(cfg.vol) == 4)
        cfg.vol = permute(cfg.vol, [4 1 2 3]);
    elseif (isfield(json.Domain, 'Dim') && numel(cfg.vol) == prod(json.Domain.Dim))
        cfg.vol = reshape(cfg.vol, json.Domain.Dim(:)');  % restore singleton dimensions, such as [1 Nx Ny]
    end
    switch (mediaformat)
        case 'svmc'
            volsize = size(cfg.vol);
            cfg.vol = reshape(typecast(uint64(cfg.vol(:)), 'uint8'), [8, volsize]);
        case 'labelplus'
            cfg.vol = unpacklabelplus(cfg.vol, jsonopt('LengthUnit', 1, json.Domain));
        case 'mua_float'
            cfg.vol = reshape(single(cfg.vol), [1, size(cfg.vol)]);
    end
elseif (isfield(json, 'Shapes'))
    cfg.shapes = savejson('Shapes', json.Shapes);
end

if (isfield(json, 'Domain') && isfield(json.Domain, 'VolumeFile') && ~isempty(json.Domain.VolumeFile))
    [fpath, fname, fext] = fileparts(json.Domain.VolumeFile);
    switch (fext)
        case '.json'
            if (isfield(json.Domain, 'Dim'))
                cfg.vol = uint8(zeros(json.Domain.Dim));
            end
            cfg.shapes = savejson('', loadjson(json.Domain.VolumeFile));
        case '.bin'
            bytelen = 1;
            mediaclass = 'uint8';
            if (isfield(json.Domain, 'MediaFormat'))
                idx = find(ismember({'byte', 'short', 'integer', 'muamus_float', ...
                                     'mua_float', 'muamus_half', 'asgn_byte', 'muamus_short', 'svmc', 'asgn_float'}, ...
                                    lower(json.Domain.MediaFormat)));
                if (idx)
                    typebyte = [1, 2, 4, 8, 4, 4, 4, 4, 8, 16];
                    typenames = {'uint8', 'uint16', 'uint32', 'single', 'single', 'uint16', 'uint8', 'uint16', 'uint8', 'single'};
                    bytelen = typebyte(idx);
                    mediaclass = typenames{idx};
                else
                    error('incorrect Domain.MediaFormat setting');
                end
            end
            cfg.vol = loadmc2(json.Domain.VolumeFile, [bytelen, json.Domain.Dim], 'uchar=>uchar');
            cfg.vol = typecast(cfg.vol(:), mediaclass);
            cfg.vol = reshape(cfg.vol, [length(cfg.vol) / prod(json.Domain.Dim), json.Domain.Dim]);
            if (size(cfg.vol, 1) == 1)
                if (exist('idx', 'var') && idx ~= 5)
                    cfg.vol = squeeze(cfg.vol);
                end
            end
        case '.nii'
            cfg.vol = mcxloadnii(json.Domain.VolumeFile);
    end
elseif (~isfield(cfg, 'shapes') && ~isfield(cfg, 'vol'))
    cfg.vol = uint8(zeros(60, 60, 60));
end

%% define the simulation session flags

cfg = copycfg(cfg, 'session', json.Session, 'ID');
cfg = copycfg(cfg, 'isreflect', json.Session, 'DoMismatch');
cfg = copycfg(cfg, 'issave2pt', json.Session, 'DoSaveVolume');
cfg = copycfg(cfg, 'issavedet', json.Session, 'DoPartialPath');
cfg = copycfg(cfg, 'issaveexit', json.Session, 'DoSaveExit');
cfg = copycfg(cfg, 'issaveseed', json.Session, 'DoSaveSeed');
cfg = copycfg(cfg, 'isnormalized', json.Session, 'DoNormalize');
cfg = copycfg(cfg, 'issaveref', json.Session, 'DoSaveRef');
cfg = copycfg(cfg, 'ismomentum', json.Session, 'DoDCS');
cfg = copycfg(cfg, 'isspecular', json.Session, 'DoSpecular');
cfg = copycfg(cfg, 'maxdetphoton', json.Session, 'MaxDetPhoton');
cfg = copycfg(cfg, 'minenergy', json.Session, 'MinEnergy');
cfg = copycfg(cfg, 'bc', json.Session, 'BCFlags');
cfg = copycfg(cfg, 'savedetflag', json.Session, 'SaveDataMask');
cfg = copycfg(cfg, 'outputformat', json.Session, 'OutputFormat');
cfg = copycfg(cfg, 'outputtype', json.Session, 'OutputType');
if (isfield(cfg, 'outputtype') && length(cfg.outputtype) == 1)
    otypemap = struct('x', 'flux', 'f', 'fluence', 'e', 'energy', 'j', 'jacobian', 'p', 'nscat', 'm', 'wm', ...
                      'r', 'rf', 'l', 'length', 's', 'rfmus', 't', 'wltof', 'b', 'wptof', ...
                      'a', 'adjoint', 'd', 'adjoint_dcoeff', 'u', 'adjoint_mus', 'v', 'adjoint_musp', ...
                      'w', 'adjoint_mua_d', 'q', 'adjoint_mua_musp');
    if (~isfield(otypemap, cfg.outputtype))
        error('output type %s is not supported', cfg.outputtype);
    end
    cfg.outputtype = otypemap.(cfg.outputtype);
end
cfg = copycfg(cfg, 'debuglevel', json.Session, 'DebugFlag');
if (~isfield(cfg, 'debuglevel'))  % older mcx2json versions used "Debug"
    cfg = copycfg(cfg, 'debuglevel', json.Session, 'Debug');
end
cfg = copycfg(cfg, 'autopilot', json.Session, 'DoAutoThread');
cfg = copycfg(cfg, 'seed', json.Session, 'RNGSeed');
if (isfield(cfg, 'seed') && ischar(cfg.seed) && ~isempty(regexp(cfg.seed, '\.mch$', 'match')))
    [data, header, cfg.seed] = loadmch(cfg.seed);
end
cfg = copycfg(cfg, 'nphoton', json.Session, 'Photons');
cfg = copycfg(cfg, 'rootpath', json.Session, 'RootPath');

%% define the forward simulation settings

if (isfield(json, 'Forward'))
    cfg.tstart = json.Forward.T0;
    cfg = copycfg(cfg, 'tstart', json.Forward, 'T0');
    cfg = copycfg(cfg, 'tend', json.Forward, 'T1');
    cfg = copycfg(cfg, 'tstep', json.Forward, 'Dt');
end

function vol = unpacklabelplus(packed, unitinmm)
% unpack the 32-bit labelplus format, {[half: value][2-bit property index][14-bit label]},
% written by mcx2json into a 3 x Nx x Ny x Nz float32 [value; property index; label] volume
volsize = size(packed);
packed = uint32(packed(:))';
half = bitshift(packed, -16);
expo = double(bitand(bitshift(half, -10), 31));
val = double(bitand(half, 1023)) / 1024;
val(expo > 0) = (1 + val(expo > 0)) .* 2.^(expo(expo > 0) - 15);
val(expo == 0) = val(expo == 0) * 2^-14;
val(bitshift(half, -15) > 0) = -val(bitshift(half, -15) > 0);
idx = double(bitand(bitshift(packed, -14), 3));
isscaled = (idx >= 1 & idx <= 2);  % mcx2json scaled these values by LengthUnit
val(isscaled) = val(isscaled) / unitinmm;
vol = reshape(single([val; idx; double(bitand(packed, 16383))]), [3, volsize]);

function outdata = copycfg(cfg, name, outroot, outfield, defaultval)
if (nargin >= 5 && ~isfield(outroot, outfield))
    outroot.(outfield) = defaultval;
end
if (isfield(outroot, outfield))
    cfg.(name) = outroot.(outfield);
end
outdata = cfg;
