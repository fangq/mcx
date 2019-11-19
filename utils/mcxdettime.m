function dett=mcxdettime(detp,prop,unitinmm)
%
% dett=mcxdettime(detp,prop,unitinmm)
%
% Recalculate the detected photon time using partial path data and 
% optical properties (for perturbation Monte Carlo or detector readings)
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%	  Ruoyang Yao (yaor <at> rpi.edu) 
%
% input:
%     detp: the 2nd output from mcxlab. detp must be a struct
%     prop: optical property list, as defined in the cfg.prop field of mcxlab's input
%     unitinmm: voxel edge-length in mm, should use cfg.unitinmm used to generate detp; 
%           if ignored, assume to be 1 (mm)
%
% output:
%     dett: re-caculated detected photon time based on the partial path data and optical property table
%
% this file is copied from Mesh-based Monte Carlo (MMC)
%
% License: GPLv3, see http://mcx.space/ for details
%

R_C0 = 3.335640951981520e-12;	% inverse of light speed in vacuum

if(nargin<3)
    if(isfield(detp,'unitinmm'))
        unitinmm=detp.unitinmm;
    else
        unitinmm=1;
    end
end

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

if(isstruct(detp))
    dett=zeros(size(detp.ppath,1),1);
    for i=1:medianum-1
        dett=dett+prop(i+1,4)*detp.ppath(:,i)*R_C0*unitinmm;
    end
else
    error('the first input must be a struct with a subfield named "ppath"');
end
