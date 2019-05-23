function cfg=json2mcx(filename)
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
% This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

if(ischar(filename))
    json=loadjson(filename);
elseif(isstruct(filename))
    json=filename;
else
    error('first input is not supported');
end

%% define the optodes: sources and detectors

cfg=struct();
if(isfield(json,'Optode'))
  if(isfield(json.Optode,'Source'))
    cfg=copycfg(cfg,'srcpos',json.Optode.Source,'Pos');
    cfg=copycfg(cfg,'srcdir',json.Optode.Source,'Dir');
    if(isfield(cfg,'srcdir'))
        cfg.srcdir(1:3)=cfg.srcdir(1:3)/norm(cfg.srcdir(1:3));
    end
    cfg=copycfg(cfg,'srcparam1',json.Optode.Source,'Param1');
    cfg=copycfg(cfg,'srcparam2',json.Optode.Source,'Param2');
    cfg=copycfg(cfg,'srctype',json.Optode.Source,'Type');
    cfg=copycfg(cfg,'srcnum',json.Optode.Source,'SrcNum');
    if(isfield(json.Optode.Source,'Pattern'))
        nz=jsonopt('Nz',1,Optode.Source.Pattern);
	cfg.srcpattern=reshape(Optode.Source.Pattern.Data,...
	  [Optode.Source.Pattern.Nx,Optode.Source.Pattern.Ny,nz]);
    end
  end
  if(isfield(json.Optode,'Detector'))
    cfg.detpos=cell2mat(struct2cell(cell2mat(json.Optode.Detector)')');
  end
end

%% define the domain and optical properties

cfg=copycfg(cfg,'issrcfrom0',json.Domain,'OriginType');
cfg=copycfg(cfg,'unitinmm',json.Domain,'LengthUnit');

cfg.prop=squeeze(cell2mat(struct2cell(cell2mat(json.Domain.Media))))';

if(isfield(json,'Shapes'))
    cfg.shapes=savejson('',json.Shapes);
end

if(isfield(json,'Domain'))
    [fpath, fname, fext]=fileparts(json.Domain.VolumeFile);
    switch(fext)
        case '.json'
            if(isfield(json.Domain,'Dim'))
               cfg.vol=uint8(zeros(json.Domain.Dim));
            end
            cfg.shapes=savejson('',loadjson(json.Domain.VolumeFile));           
        case '.bin'
            bytelen=1;
            mediaclass='uint8';
            if(isfield(json.Domain,'MediaFormat'))
                idx=find(ismember({'byte','short','integer','muamus_float',...
                    'mua_float','muamus_half','asgn_byte','muamus_short'},...
                    lower(json.Domain.MediaFormat)));
                if(idx)
                    typebyte=[1,2,4,8,4,4,4,4];
                    typenames={'uint8','uint16','uint32','single','single','uint16','uint8','uint16'};
                    bytelen=typebyte(idx);
                    mediaclass=typenames{idx};
                else
                    error('incorrect Domain.MediaFormat setting')
                end
            end
            cfg.vol=loadmc2(json.Domain.VolumeFile,[bytelen, json.Domain.Dim],'uchar=>uchar');
            cfg.vol=typecast(cfg.vol(:),mediaclass);
            cfg.vol=reshape(cfg.vol,[length(cfg.vol)/prod(json.Domain.Dim), json.Domain.Dim]);
            if(size(cfg.vol,1)==1)
                if(exist(idx,'var') && idx~=5)
                    cfg.vol=squeeze(cfg.vol);
                end
            end
        case '.nii'
            cfg.vol=mcxloadnii(json.Domain.VolumeFile);
    end
end

%% define the simulation session flags

cfg=copycfg(cfg,'session',json.Session,'ID');
cfg=copycfg(cfg,'isreflect',json.Session,'DoMismatch');
cfg=copycfg(cfg,'issave2pt',json.Session,'DoSaveVolume');
cfg=copycfg(cfg,'issavedet',json.Session,'DoPartialPath');
cfg=copycfg(cfg,'issaveexit',json.Session,'DoSaveExit');
cfg=copycfg(cfg,'issaveseed',json.Session,'DoSaveSeed');
cfg=copycfg(cfg,'isnormalize',json.Session,'DoNormalize');
cfg=copycfg(cfg,'outputformat',json.Session,'OutputFormat');
cfg=copycfg(cfg,'outputtype',json.Session,'OutputType');
cfg=copycfg(cfg,'debuglevel',json.Session,'Debug');
cfg=copycfg(cfg,'autopilot',json.Session,'DoAutoThread');
cfg=copycfg(cfg,'autopilot',json.Session,'DoAutoThread');
cfg=copycfg(cfg,'seed',json.Session,'RNGSeed');
if(isfield(cfg,'seed') && ischar(cfg.seed) && ~isempty(regexp(cfg.seed,'\.mch$','match')))
    [data, header, cfg.seed]=loadmch(cfg.seed);
end
cfg=copycfg(cfg,'nphoton',json.Session,'Photons');
cfg=copycfg(cfg,'rootpath',json.Session,'RootPath');

%% define the forward simulation settings

if(isfield(json,'Forward'))
    cfg.tstart=json.Forward.T0;
    cfg=copycfg(cfg,'tstart',json.Forward,'T0');
    cfg=copycfg(cfg,'tend',json.Forward,'T1');
    cfg=copycfg(cfg,'tstep',json.Forward,'Dt');
end

function outdata=copycfg(cfg,name,outroot,outfield,defaultval)
if(nargin>=5 && ~isfield(outroot,outfield))
    outroot.(outfield)=defaultval;
end
if(isfield(outroot,outfield))
    cfg.(name)=outroot.(outfield);
end
outdata=cfg;
