function [tau,g1]=mcxdcsg1(detps,tau, disp_model, DV, lambda, format, varargin)
%
%   [tau,g1]=mcxdcsg1(detps,tau, disp_model, DV, lambda, format)
%
%   Compute simulated electric-field auto-correlation function using
%   simulated photon pathlengths and scattering momentum transfer
%
%   author: Stefan Carp (carp <at> nmr.mgh.harvard.edu)
%
%   input:
%       detps:      the file name of the output .mch file or the 2nd output from mcxlab
%       tau:        correlation times at which to compute g1 
%                   (default: 1e-7 to 1e-1 seconds, log equidistant)
%       disp_model: displacement model ('brownian', 'random_flow', <custom>)
%                   (default: brownian, see further explanation below)
%       disp_var:   value of displacement variable using mm as unit of
%                   length and s as unit of time
%                   (default: 1e-7 mm^2/s, see further explanation below)
%       lambda:     wavelenght of light used in nm
%                   (default: 785)
%       format:     the format used to save the .mch file 
%                   (default: 'float')
%
%   output:
%
%       tau:        correlation times at which g1 was computed provided for
%                   convenience (copied from input if set, otherwise 
%                   outputs default)
%       g1:         field auto-correlation curves, one for each detector
%
%   The displacement model indicates the formula used to compute the root
%   mean square displacement of scattering particles during a given delay
%   
%   brownian:       RMS= 6 * DV * tau; 
%                   DV(displacement variable)=Db (brownian diffusion coeff)
%   random_flow:    RMS= DV^2 * tau^2; 
%                   DV = V (first moment of velocity distribution)
%   <custom>:       any string other than 'brownian' or 'random_flow' will
%                   be evaluate as is using Matlab evalf, make sure it uses
%                   'DV' as the flow related independent variable, tau is
%                   indexed as tau(J). Any additional parameters can be 
%                   sent via "varargin"
%
%   This file is part of Mesh-Based Monte Carlo
%   License: GPLv3, see http://mcx.space for details
%

if nargin<6, format='float'; end
if nargin<5, lambda=785; end
if nargin<4, DV=1e-7; end
if nargin<3, disp_model='brownian'; end
if nargin<2, tau=logspace(-7,-1,200); end

if(ischar(detps))
   [mch_data,mch_header]=loadmch(detps,format);
   [fpath,fname,fext]=fileparts(detps);

   cfg=loadjson([fpath filesep fname '.json']);
   prop=cell2mat(cfg.Domain.Media)
   mua=cell2mat({prop.mua});
   mua=mua(2:end);
   n=cell2mat({prop.n});
   n=n(2:end);
   medianum=length(mua)-1;
   detps=struct('detid',mch_data(1,:)');
   detps.ppath=mch_data(3:2+medianum,:)';
   if(size(mch_data,1)>=2*medianum+2)
      detps.mom=mch_data(medianum+3:2*medianum+2,:)';
   end
else
   mua=detps.prop(2:end,1)';
   n=detps.prop(2:end,4)';
end

if(~isfield(detps,'mom'))
   error('No momentum transfer data are found, please rerun your simulation and set cfg.ismomentum=1.');
end

if strcmp(disp_model,'brownian'),
    disp_str='rmsdisp=6*DV.*tau(J);';
elseif strcmp(disp_model,'random_flow'),
    disp_str='rmsdisp=DV.^2.*tau(J).^2;';
else 
    disp_str=['rmsdisp=' disp_model ';'];
end

k0=2*pi*n/(lambda*1e-6);

detlist=sort(unique(detps.detid));
g1=zeros(max(detlist),length(tau));

for detid=1:length(detlist)
    I=detlist(detid);
    idx= find(detps.detid==I);     
    fprintf('Processing detector %.0f: %.0f photons\n',I,length(idx));

    for J=1:length(tau),
        eval(disp_str);
        g1(I,J)=sum(exp(-(k0.^2.*rmsdisp/3)*detps.mom'-mua*detps.ppath'));
    end
    g1_norm=sum(exp(-mua*detps.ppath'));
    g1(I,:)=g1(I,:)./g1_norm;
end
    


