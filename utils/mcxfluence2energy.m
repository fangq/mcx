function output=mcxfluence2energy(flux,vol,prop,tstep)
%
% output=mcxfluence2energy(flux,cfg)
%    or
% output=mcxfluence2energy(flux,vol,prop,tstep)
%
% Calculate energy deposition (1/mm^3) use fluence rate (1/mm^2/s) or vice versa
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     flux: the 1st output from mcxlab. flux can be either a struct or an array (flux.data)
%     cfg:  the input used in mcxlab
%        or, you can use
%     vol:  if cfg is not given, user must provide the volume, i.e. cfg.vol, and
%     prop: the property list, i.e. cfg.prop, as well as
%     tstep: the time-step of the 
% output:
%     output: if cfg.output is 'flux', the output is the energy deposition (1/mm^3)
%             if cfg.output is 'energy', the output is the fluence rate (1/mm^2)
%
% this file is part of Monte Carlo eXtreme (MCX)
%
% License: GPLv3, see http://mcx.space/ for details
%

data=flux;

if(nargin>=1 && isstruct(flux) && isfield(flux,'data'))
    data=flux.data;
end

if(nargin==2 && isstruct(vol) && isfield(vol,'vol') && isfield(vol,'prop'))
    cfg=vol;
    vol=cfg.vol;
    prop=cfg.prop;
    tstep=cfg.tstep;
    if(isfield(cfg,'outputtype') && strcmp(cfg.outputtype,'fluence'))
        data=data./tstep;
    end
else
    error('must provide cfg, or vol/prop as inputs');
end

mua=prop(:,1);
mua=repmat(mua(vol+1),1,1,1,size(flux.data,4));
if(exist('cfg','var') && isfield(cfg,'outputtype') && strcmp(cfg.outputtype,'energy'))
    data(mua==0)=0;
    mua(mua==0)=1;
    output=data./(tstep*mua);
else % otherwise, assume input is fluence rate
    output=data*(tstep).*mua;
end