function avgnscat=mcxmeanscat(detp,prop)
%
% avgnscat=mcxmeanscat(detp,prop)
%
% Calculate the average scattering event counts for each tissue type for a given source-detector pair
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     detp: the 2nd output from mcxlab. detp can be either a struct or an array (detp.data)
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%
% output:
%     avgnscat: the average scattering event count for each tissue type 
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%

detw=mcxdetweight(detp,prop);
avgnscat=sum(double(detp.nscat).*repmat(detw(:),1,size(detp.nscat,2))) / sum(detw(:));
