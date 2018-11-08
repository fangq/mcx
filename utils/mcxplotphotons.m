function varargout=mcxplotphotons(traj,varargin)
%
%    mcxplotphotons(traj)
%       or
%    mcxplotphotons(traj, 'color','r','marker','o')
%    [sorted, linehandle]=mcxplotphotons(traj)
%
%    Plot photon trajectories from MCXLAB's output
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        traj: the 5th output of mcxlab, storing the photon trajectory info
%           traj.id: the photon index being recorded
%           traj.pos: the 3D position of the photon; for each photon, the
%                  positions are stored in serial order
%           traj.data: the combined output, in the form of 
%                [id,pos,weight,reserved]'
%
%    output:
%        sorted: a structure to store the sorted trajectory info
%        sorted.id: the sorted vector of photon id, staring from 0
%        sorted.pos: the sorted position vector of each photon, only
%                 recording the scattering sites.
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if(~isstruct(traj) && size(traj,2)==6)
    traj=struct('id',typecast(single(traj(:,1)),'uint32'),'pos',traj(:,2:4),'weight',traj(:,5));
end

[newid, idx]=sort(traj.id);
newpos=traj.pos(idx,:);
hg=plot3(newpos(:,1),newpos(:,2),newpos(:,3),'.-',varargin{:});

output={struct('id',newid, 'pos',newpos), hg};
[varargout{1:nargout}]=output{1:nargout};
