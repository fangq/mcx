function tpsf = mcxdettpsf(detp,detnum,prop,time)
%
% tpsf=mcxdettpsf(detp,detnum,prop,time)
%
% Calculate the temporal point spread function curve of a specified detector
% given the partial path data, optical properties, and distribution of time bins
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%	  Ruoyang Yao (yaor <at> rpi.edu) 
%
% input:
%     detp: the 2nd output from mcxlab. detp must be a struct with detid and ppath subfields
%     detnum: specified detector number
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%     time: distribution of time bins, a 1*3 vector [tstart tend tstep]
%           could be defined different from in the cfg of mcxlab's input
% output:
%     tpsf: caculated temporal point spread function curve of the specified detector
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%

% select the photon data of the specified detector
detp.ppath=detp.ppath(detp.detid==detnum,:);
detp.detid=detp.detid(detp.detid==detnum);

% calculate the detected photon weight and arrival time
replayweight=mcxdetweight(detp,prop);
replaytime=mcxdettime(detp,prop);

% define temporal point spread function vector
nTG = round((time(2)-time(1))/time(3));     % maximum time gate number
tpsf = zeros(nTG,1);

% calculate the time bin, make sure not to exceed the boundary
ntg = ceil((replaytime-time(1))/time(3));
ntg(ntg<1)=1;
ntg(ntg>nTG)=nTG;

% add each photon weight to corresponding time bin
for i=1:length(replayweight)
    tpsf(ntg(i)) = tpsf(ntg(i))+replayweight(i);
end

end
