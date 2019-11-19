function detw=mcxdetweight(detp,prop,unitinmm)
%
% detw=mcxdetweight(detp,prop,unitinmm)
%
% Recalculate the detected photon weight using partial path data and 
% optical properties (for perturbation Monte Carlo or detector readings)
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     detp: the 2nd output from mcxlab. detp must a struct 
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%     unitinmm: voxel edge-length in mm, should use cfg.unitinmm used to generate detp; 
%           if ignored, assume to be 1 (mm)
%
% output:
%     detw: re-caculated detected photon weight based on the partial path data and optical property table
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%

if(nargin<2)
    if(isfield(detp,'prop'))
        prop=detp.prop;
    else
        error('must provide input "prop"');
    end
end

medianum=size(prop,1);
if(medianum<=1)
    error('empty property list');
end

if(nargin<3)
    if(isfield(detp,'unitinmm'))
        unitinmm=detp.unitinmm;
    else
        unitinmm=1;
    end
end

if(isstruct(detp))
    if(~isfield(detp,'w0'))
        detw=ones(size(detp.ppath,1),1);
    else
        detw=detp.w0;
    end
    for i=1:medianum-1
        detw=detw.*exp(-prop(i+1,1)*detp.ppath(:,i)*unitinmm);
    end
else
    error('the first input must be a struct with a subfield named "ppath"');
end
