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

if(isstruct(traj) && ~isfield(traj, 'id') && isfield(traj, 'data'))
    traj=struct('id',typecast(single(traj.data(1,:)'),'uint32'),'pos',traj.data(2:4,:)','weight',traj.data(5,:)', 'data', traj.data);
end
if(~isstruct(traj) && size(traj,2)==6)
    traj=struct('id',typecast(single(traj(:,1)),'uint32'),'pos',traj(:,2:4),'weight',traj(:,5));
end

[newid, idx]=sort(traj.id);
lineend=(diff(newid)>0);
newidx=cumsum([0;lineend]+1);
newpos=nan(length(idx)+length(lineend),4);
newpos(newidx,1:3)=traj.pos(idx,:);
newpos(newidx,4)=traj.data(5,:)';

if(~(length(varargin)==1 && strcmp(varargin{1},'noplot')))
    edgealpha = 1 - (1-(exist('OCTAVE_VERSION','builtin')~=0))*0.75;  % octave 6 returns empty if edgealpha<1
    hg=patch(newpos(:,1),newpos(:,2),newpos(:,3),newpos(:,4), 'edgecolor', 'interp', 'edgealpha', edgealpha, varargin{:});
    view(3);
    axis equal;
else
    hg=[];
end

output={struct('id',newid, 'pos',traj.pos(idx,:)), hg};
[varargout{1:nargout}]=output{1:nargout};
