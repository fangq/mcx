function hs = mcxpreview(cfg, varargin)
%
% Format:
%    hs=mcxpreview(cfg);
%
% Preview the simulation configuration for both MCXLAB and MMCLAB
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Input:
%    cfg: a struct, or struct array. Each element of cfg defines
%         the parameters associated with a simulation. Please run
%         'help mcxlab' or 'help mmclab' to see the details.
%         mcxpreview supports the cfg input for both mcxlab and mmclab.
%
% Output:
%    hs: a struct to the list of handles of the plotted domain elements.
%
% Dependency:
%    this function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)
%
% This function is part of Monte Carlo eXtreme (MCX) URL: https://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

if (nargin == 0)
    error('input field cfg must be defined');
end

if (~isstruct(cfg))
    error('cfg must be a struct or struct array');
end

if (~exist('latticegrid', 'file'))
    error('you must install the iso2mesh toolbox from https://github.com/fangq/iso2mesh first');
end

len = length(cfg);
if (nargout > 0)
    hs = cell(len, 1);
end

% a fixed color for each label: a golden-ratio sequence rather than seeded random
% numbers, so the random number generator and its state are left untouched (rng()
% is not in older MATLAB and Octave, and rand('state') switches to a legacy one)
surfcolors = mod((1:1024)' * [0.618033988749895, 0.414213562373095, 0.732050807568877], 1);
isholdplot = ishold;

for i = 1:len
    if (~isfield(cfg(i), 'vol') && ~isfield(cfg(i), 'node') && ~isfield(cfg(i), 'shapes'))
        error('cfg.vol or cfg.node or cfg.shapes is missing');
    end

    if (i > 1)
        figure;
    end
    hold on;

    voxelsize = 1;
    if (isfield(cfg(i), 'unitinmm'))
        voxelsize = cfg(i).unitinmm;
    end

    offset = 1;
    if (isfield(cfg(i), 'issrcfrom0') && cfg(i).issrcfrom0 == 1 || isfield(cfg(i), 'node'))
        offset = 0;
    end

    hseg = [];
    if (isfield(cfg(i), 'vol') && ~isfield(cfg(i), 'node'))
        % render mcxlab voxelated domain
        dim = size(cfg(i).vol);
        if (ndims(cfg(i).vol) == 4) % for spatially varying medium
            dim = dim(2:end);
            cfg(i).vol = ones(dim);
        end
        [bbxno, bbxfc] = latticegrid(0:dim(1):dim(1), 0:dim(2):dim(2), 0:dim(3):dim(3));
        hbbx = plotmesh((bbxno + offset) * voxelsize, bbxfc, 'facecolor', 'none');

        val = unique(cfg(i).vol(:));
        val(val == 0) = [];
        padvol = zeros(dim + 2);
        padvol(2:end - 1, 2:end - 1, 2:end - 1) = cfg(i).vol;

        if (length(val) > 1)
            hseg = zeros(length(val), 1);
            [isoct, octver] = isoctavemesh;
            for id = 1:length(val)
                if (~isoct || sscanf(octver, '%d') > 5)
                    [no, fc] = binsurface(int8(padvol == val(id)), 0.5);
                else
                    [fc, no] = isosurface(int8(padvol == val(id)), 0.5);
                    no = no(:, [2 1 3]) - 0.5;
                end
                hseg(id) = plotmesh((no - 1) * voxelsize, fc, 'facealpha', 0.3, 'linestyle', 'none', 'facecolor', surfcolors(val(id) + 1, :), varargin{:});
            end
        end
    elseif (isfield(cfg(i), 'node') && isfield(cfg(i), 'elem'))
        % render mmclab mesh domain

        elemtype = ones(size(cfg(i).elem, 1), 1);
        if (isfield(cfg(i), 'elemprop'))
            elemtype = cfg(i).elemprop;
        else
            if (size(cfg(i).elem, 2) > 4)
                elemtype = cfg(i).elem(:, 5);
            end
        end
        etypes = unique(elemtype);
        no = cfg(i).node * voxelsize;
        mindim = min(max(no) - min(no));
        hseg = zeros(length(etypes), 1);
        for id = 1:size(etypes, 1)
            hseg(id) = plotmesh(no, [], cfg(i).elem(elemtype == etypes(id), :), 'facealpha', 0.3, 'linestyle', 'none', 'facecolor', surfcolors(etypes(id) + 1, :), varargin{:});
        end
        if (isfield(cfg(i), 'noderoi'))
            [xsp, ysp, zsp] = sphere;
            idx = find(cfg(i).noderoi);
            for id = 1:length(idx)
                hseg(end + id) = surf(xsp * cfg(i).noderoi(idx(id)) + cfg(i).node(idx(id), 1), ...
                                      ysp * cfg(i).noderoi(idx(id)) + cfg(i).node(idx(id), 2), ...
                                      zsp * cfg(i).noderoi(idx(id)) + cfg(i).node(idx(id), 3), 'linestyle', 'none', 'facecolor', 'm', 'facealpha', 0.5);
            end
        end
    end

    if (isfield(cfg(i), 'shapes'))
        if (isfield(cfg(i), 'vol'))
            hseg = mcxplotshapes(cfg(i).shapes, size(cfg(i).vol), offset, hseg, voxelsize, varargin{:});
        else
            hseg = mcxplotshapes(cfg(i).shapes, [60, 60, 60], offset, hseg, voxelsize, varargin{:});
        end
    end

    % rendering source position and direction
    if (~isfield(cfg(i), 'srcpos') || ~isfield(cfg(i), 'srcdir'))
        error('cfg.srcpos or cfg.srcdir is missing');
    end

    % one row of cfg.srcpos per source; a single srcdir/srcparam1/srcparam2 row is shared
    srcpos = cfg(i).srcpos(:, 1:3) * voxelsize;
    nsrc = size(srcpos, 1);
    srcdir = cfg(i).srcdir(min(1:nsrc, end), :);
    hsrc = plotmesh(srcpos, 'r*');
    if (exist('mindim', 'var'))
        srcvec = srcdir(:, 1:3) * mindim * 0.05;
    else
        srcvec = srcdir(:, 1:3) * 10 * voxelsize;
    end
    headsize = 1e2;
    if (isoctavemesh)
        headsize = 0.5;
    end
    hdir = quiver3(srcpos(:, 1), srcpos(:, 2), srcpos(:, 3), srcvec(:, 1), srcvec(:, 2), srcvec(:, 3), 'linewidth', 3, 'color', 'r', 'MaxHeadSize', headsize, 'AutoScaleFactor', 1, varargin{:});

    % rendering area-source aperature, one per source
    hsrcarea = [];
    if (isfield(cfg(i), 'srctype'))
        for is = 1:nsrc
            sp = srcpos(is, :);
            sd = srcdir(is, 1:3);
            if (isfield(cfg(i), 'srcparam1'))
                p1 = cfg(i).srcparam1(min(is, end), :);
            end
            if (isfield(cfg(i), 'srcparam2'))
                p2 = cfg(i).srcparam2(min(is, end), :);
            end
            if (strcmp(cfg(i).srctype, 'disk') || strcmp(cfg(i).srctype, 'gaussian') || strcmp(cfg(i).srctype, 'zgaussian') || strcmp(cfg(i).srctype, 'ring'))
                if (~isfield(cfg(i), 'srcparam1'))
                    error('cfg.srcparam1 is missing');
                end
                [ncyl, fcyl] = meshacylinder(sp, sp + sd * 1e-5, p1(1) * voxelsize, 0, 0);
                hsrcarea = [hsrcarea; plotmesh(ncyl, fcyl{end - 1}, 'facecolor', 'r', 'linestyle', 'none')];
                if (p1(2) > 0)
                    [ncyl, fcyl] = meshacylinder(sp, sp + sd * 1e-5, p1(2) * voxelsize, 0, 0);
                    hsrcarea = [hsrcarea; plotmesh(ncyl, fcyl{end - 1}, 'facecolor', 'k', 'linestyle', 'none')];
                end
            elseif (strcmp(cfg(i).srctype, 'planar') || strcmp(cfg(i).srctype, 'pattern') || strcmp(cfg(i).srctype, 'fourier') || ...
                    strcmp(cfg(i).srctype, 'fourierx') || strcmp(cfg(i).srctype, 'fourierx2d') || strcmp(cfg(i).srctype, 'pencilarray'))
                if (~isfield(cfg(i), 'srcparam1') || ~isfield(cfg(i), 'srcparam2'))
                    error('cfg.srcparam2 or cfg.srcparam2 is missing');
                end
                if (strcmp(cfg(i).srctype, 'fourierx') || strcmp(cfg(i).srctype, 'fourierx2d'))
                    vec2 = cross(sd, p1(1:3) * voxelsize);
                else
                    vec2 = p2(1:3) * voxelsize;
                end
                nrec = [0 0 0; p1(1:3) * voxelsize; p1(1:3) * voxelsize + vec2; vec2];
                hsrcarea = [hsrcarea; plotmesh(nrec + repmat(sp, [4, 1]), {[1 2 3 4 1]})];
            elseif (strcmp(cfg(i).srctype, 'pattern3d'))
                dim = p1(1:3);
                [bbxno, bbxfc] = latticegrid(0:dim(1):dim(1), 0:dim(2):dim(2), 0:dim(3):dim(3));
                hbbx = plotmesh(((bbxno + repmat(cfg(i).srcpos(is, 1:3), size(bbxno, 1), 1)) + offset) * voxelsize, bbxfc, 'facecolor', 'y', 'facealpha', 0.3);
            elseif (strcmp(cfg(i).srctype, 'slit') || strcmp(cfg(i).srctype, 'line'))
                if (~isfield(cfg(i), 'srcparam1'))
                    error('cfg.srcparam1 is missing');
                end
                hsrcarea = [hsrcarea; plotmesh([sp; sp + p1(1:3) * voxelsize], [1 2], 'linewidth', 3, 'color', 'r')];
            end
        end
    end

    % rendering detectors
    hdet = [];
    hdetdir = [];
    if (isfield(cfg(i), 'detpos'))
        hdet = zeros(size(cfg(i).detpos, 1), 1);
        detpos = cfg(i).detpos(:, 1:3) * voxelsize;
        if (size(cfg(i).detpos, 2) == 4)
            detpos(:, 4) = cfg(i).detpos(:, 4) * voxelsize;
        else
            detpos(:, 4) = 1;
        end
        for id = 1:size(detpos, 1)
            [sx, sy, sz] = sphere;
            hdet(id) = surf(sx * detpos(id, 4) + (detpos(id, 1)), ...
                            sy * detpos(id, 4) + (detpos(id, 2)), ...
                            sz * detpos(id, 4) + (detpos(id, 3)), ...
                            'facealpha', 0.3, 'facecolor', 'g', 'linestyle', 'none');
        end
        % detector directions, which the adjoint modes take from cfg.detdir
        if (isfield(cfg(i), 'detdir') && ~isempty(cfg(i).detdir))
            if (exist('mindim', 'var'))
                detvec = cfg(i).detdir(:, 1:3) * mindim * 0.05;
            else
                detvec = cfg(i).detdir(:, 1:3) * 10 * voxelsize;
            end
            hdetdir = quiver3(detpos(:, 1), detpos(:, 2), detpos(:, 3), detvec(:, 1), detvec(:, 2), detvec(:, 3), 'linewidth', 3, 'color', 'g', 'MaxHeadSize', headsize, 'AutoScaleFactor', 1, varargin{:});
        end
    end

    % combining all handles
    if (nargout > 0)
        hs{i} = struct('bbx', hbbx, 'seg', hseg, 'src', hsrc, 'srcarrow', hdir, 'srcarea', hsrcarea, 'det', hdet, 'detarrow', hdetdir);
    end
end

if (~isholdplot)
    hold off;
end
