function detw=mcxdetweight(detp,prop,w0)
%
% detw=mcxdetweight(detp,prop)
%    or
% detw=mcxdetweight(detp,prop,w0)
%
% Recalculate the detected photon weight using partial path data and 
% optical properties (for perturbation Monte Carlo or detector readings)
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     detp: the 2nd output from mcxlab. detp can be either a struct or an array (detp.data)
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%     w0: (optional), the initial weight of photon. 
%           if detp is a struct, this input is ignored, detp.w0 is used instead
%           if detp is an array and w0 is ignored, the last row of detp is used as w0
%
% output:
%     detw: re-caculated detected photon weight based on the partial path data and optical property table
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%

medianum=size(prop,1);
if(medianum<=1)
    error('empty property list');
end

if(isstruct(detp))
    if(~isfield(detp,'w0'))
        detw=ones(size(detp.ppath,1),1);
    else
        detw=detp.w0;
    end
    for i=1:medianum-1
        detw=detw.*exp(-prop(i+1,1)*detp.ppath(:,i));
    end
else
    detp=detp';
    if(nargin<3)
        w0=detp(:,end);
    end
    detw=w0(:);
    if(size(detp,2)>=2*medianum+1)
        for i=1:medianum-1
            detw=detw.*exp(-prop(i+1,1)*detp(:,i+medianum));
        end
    else
        for i=1:medianum-1
            detw=detw.*exp(-prop(i+1,1)*detp(:,i+2));
        end
    end
end
