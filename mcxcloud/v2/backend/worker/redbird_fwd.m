% MCX Cloud v2 redbird: minimal FEM diffusion forward solve (MATLAB/Octave reference).
%
% NOT the deployed worker path -- redbird_fwd.py is (see Dockerfile.redbird). This is the
% MATLAB-parity reference kept for cross-checking the Python driver's numbers; the two
% were verified to agree to ~0.1% on a 6504-node slab (corr 0.99999998, detphi within
% 0.1%), and both agree with the analytical semi-infinite CW solution.
%
% Requires Octave >= 4.2 (or MATLAB) for containers.Map, which redbird's FEM path uses
% unconditionally, plus zmat if the input's mesh arrays are zlib-packed. Notably this
% CANNOT run in fangqq/mcxstudio:v2025.10 (Octave 4.0.0, no containers.Map, incomplete
% zmat) -- which is exactly why the deployed path is Python.
%
% Reads the SAME mcxcloud JSON input that mcx/mmc consume (input.json in the CWD),
% builds a redbird cfg from it, runs a single-wavelength CW/RF forward solve, and
% writes the nodal fluence to output.jnii (JNIfTI) + detector readings to
% output_detp.jdat.
%
% Deliberately minimal: single wavelength, fixed optical properties, forward only.
% Reconstruction, multi-spectral, and microwave (Helmholtz) modes are out of scope
% and are not reachable from the mcxcloud schema.
%
% Field mapping (mcxcloud/mmc JSON -> redbird cfg), see rbmeshprep.m/rbfemlhs.m:
%   Shapes.MeshNode        -> cfg.node          (Nn x 3)
%   Shapes.MeshElem        -> cfg.elem + cfg.seg  redbird keeps the element LABEL in a
%                             separate cfg.seg and wants a 4-column cfg.elem, whereas
%                             mmc carries the label as MeshElem's 5th column (mmc's
%                             cfg.elemprop). So the label column is split out here.
%   Domain.Media           -> cfg.prop          1:1, no conversion: redbird's prop is
%                             also [mua, mus, g, n] with RAW mus, reduced internally
%                             (rbfemlhs: musp = prop(:,2).*(1-prop(:,3))), and cfg.seg
%                             indexes it 0-based (prop(seg+1,:)) exactly like mmc.
%   Optode.Source.Pos/Dir  -> cfg.srcpos / cfg.srcdir
%   Optode.Detector[].Pos  -> cfg.detpos
%   Forward.Omega          -> cfg.omega         (rad/s, 0 = CW; the backend derives this
%                             from the canonical Optode.Source.Frequency in Hz)
%
% NOTE: no local/nested functions on purpose. Octave only registers a script-local
% function once execution REACHES its definition (so it would have to come first),
% while MATLAB requires local functions at the END of a script -- the two conventions
% are mutually exclusive, so this stays function-free to run unmodified in both.

% ---- load the mcxcloud input -------------------------------------------------
cfgin = loadjson('input.json');

% the mesh lives under Shapes (or Mesh), matching detectEngine() in the backend
if (isfield(cfgin, 'Shapes'))
    M = cfgin.Shapes;
elseif (isfield(cfgin, 'Mesh'))
    M = cfgin.Mesh;
else
    error('redbird: input has no Shapes/Mesh section with a tetrahedral mesh');
end
if (~isfield(M, 'MeshNode') || ~isfield(M, 'MeshElem'))
    error('redbird: input mesh requires both MeshNode and MeshElem');
end

clear cfg;

% ---- mesh --------------------------------------------------------------------
node = M.MeshNode;
elem = M.MeshElem;
cfg.node = node(:, 1:3);
if (size(elem, 2) > 4)
    % split mmc's embedded label column out into redbird's separate cfg.seg
    cfg.seg  = elem(:, 5);
    cfg.elem = elem(:, 1:4);
else
    cfg.elem = elem(:, 1:4);
    cfg.seg  = ones(size(elem, 1), 1);
end

% ---- optical properties ------------------------------------------------------
% Domain.Media is either a list of {mua,mus,g,n} objects or of [mua,mus,g,n] rows;
% loadjson yields a struct array, a cell array, or a plain numeric matrix.
Media = cfgin.Domain.Media;
if (isnumeric(Media))
    prop = Media(:, 1:4);
else
    prop = zeros(numel(Media), 4);
    for i = 1:numel(Media)
        if (iscell(Media))
            m = Media{i};
        else
            m = Media(i);
        end
        if (isstruct(m))
            prop(i, :) = [m.mua, m.mus, m.g, m.n];
        else
            row = reshape(m, 1, []);
            prop(i, :) = row(1:4);
        end
    end
end
cfg.prop = prop;

% ---- sources -----------------------------------------------------------------
% loadjson gives [x,y,z] as 1x3 and [[..],[..]] as Nx3; reshape guards the (unlikely)
% column-vector form so a single source is never transposed into 3 bogus sources.
S = cfgin.Optode.Source;
srcpos = S.Pos;
if (size(srcpos, 2) < 3)
    srcpos = reshape(srcpos, 1, []);
end
cfg.srcpos = srcpos(:, 1:3);
srcdir = S.Dir;
if (size(srcdir, 2) < 3)
    srcdir = reshape(srcdir, 1, []);
end
cfg.srcdir = srcdir(:, 1:3);

% ---- detectors ---------------------------------------------------------------
detpos = [];
if (isfield(cfgin.Optode, 'Detector'))
    D = cfgin.Optode.Detector;
    for i = 1:numel(D)
        if (iscell(D))
            d = D{i};
        else
            d = D(i);
        end
        p = d.Pos;
        if (size(p, 2) < 3)
            p = reshape(p, 1, []);
        end
        detpos = [detpos; p(:, 1:3)];
    end
end
if (isempty(detpos))
    % rbfemrhs needs at least one detector column to build a RHS; a mesh-centroid probe
    % keeps a detector-less input runnable (the nodal fluence field is what matters)
    detpos = mean(cfg.node, 1);
    fprintf(1, '[redbird] no detectors given; probing the mesh centroid\n');
end
cfg.detpos = detpos;
% rbgetoptodes dereferences cfg.detdir UNGUARDED whenever detpos is set, and mcxcloud
% mesh detectors only carry {Pos, R} (no direction) -- synthesize inward normals. Slice
% to 3 columns: rbgetoptodes computes detpos + detdir.*ltr against a 3-column detpos.
dd = rbgetdetdir(cfg);
cfg.detdir = dd(:, 1:3);

% ---- RF modulation (0 = CW) --------------------------------------------------
cfg.omega = 0;
if (isfield(cfgin, 'Forward') && isfield(cfgin.Forward, 'Omega'))
    cfg.omega = cfgin.Forward.Omega;
end

% ---- forward solve -----------------------------------------------------------
nsrc = size(cfg.srcpos, 1);
fprintf(1, '[redbird] %d nodes, %d elems, %d src, %d det, omega=%g rad/s\n', ...
        size(cfg.node, 1), size(cfg.elem, 1), nsrc, size(cfg.detpos, 1), cfg.omega);

tic;
cfg = rbmeshprep(cfg);
fprintf(1, '[redbird] mesh prep ... %.3f s\n', toc);

tic;
[detphi, phi] = rbrunforward(cfg);
fprintf(1, '[redbird] forward solve ... %.3f s\n', toc);

% ---- save the nodal fluence as JNIfTI ---------------------------------------
% rbfemrhs builds one RHS column per source AND one per detector (detectors double as
% adjoint sources), so phi is Nn x (Nsrc+Ndet). Only the forward source columns are
% the fluence field the user asked for.
phi = full(phi);
phi = phi(:, 1:nsrc);
if (~isreal(phi))
    % RF (omega>0) gives a complex field; store the amplitude, which is what the
    % preview renders. The complex detector readings are preserved in detphi below.
    fprintf(1, '[redbird] complex (RF) field: saving amplitude\n');
    phi = abs(phi);
end

jnii = jnifticreate(single(phi), 'Name', 'redbird nodal fluence');
% Annotate ONLY the data array, not the whole struct: NIFTIHeader.Dim must stay a plain
% JSON array for the frontend's mesh-output frame detection (preview.js drawmeshOutput).
% Uncompressed _ArrayData_ on purpose -- it needs no zmat/zlib mex in the worker image,
% and the frontend decodes that form natively (util.js decodeJDataArray).
jnii.NIFTIData = jdataencode(single(phi), 'AnnotateArray', 1);
savejson('', jnii, 'FileName', 'output.jnii');

% ---- save detector readings --------------------------------------------------
% Ndet x Nsrc measurement matrix (complex for RF). Offered to the user as a download
% only, so a plain annotated array is enough.
savejson('', struct('DetPhi', jdataencode(detphi, 'AnnotateArray', 1)), ...
         'FileName', 'output_detp.jdat');

fprintf(1, '[redbird] wrote output.jnii (%d nodes x %d src) + output_detp.jdat\n', ...
        size(phi, 1), size(phi, 2));
