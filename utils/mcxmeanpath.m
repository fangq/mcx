function avgpath=mcxmeanpath(detp,prop)
%
% avgpath=mcxmeanpath(detp,prop)
%
% Calculate the average pathlengths for each tissue type for a given source-detector pair
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     detp: the 2nd output from mcxlab. detp can be either a struct or an array (detp.data)
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%
% output:
%     avepath: the average pathlength for each tissue type 
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%
 if(isfield(detp,'unitinmm'))
    unitinmm=detp.unitinmm;
else
    unitinmm=1;
end

detw=mcxdetweight(detp,prop);
avgpath=sum(detp.ppath.*unitinmm.*repmat(detw(:),1,size(detp.ppath,2))) / sum(detw(:));
