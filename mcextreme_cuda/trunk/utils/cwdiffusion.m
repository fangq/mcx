function Phi = cwdiffusion(mua, musp, v, Reff, srcpos,detpos)
% Phi = cwdiffusion(mua, musp, v, Reff, srcpos,detpos)
%
%  semi-infinite medium analytical solution to diffusion model
% 
%  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
%
% see Boas2002

D = 1/(3*(mua+musp));
zb = (1+Reff)/(1-Reff)*2*D

z0 = 1/(musp+mua);
r=getdistance([srcpos(:,1:2) srcpos(:,3)+z0],detpos)
r2=getdistance([srcpos(:,1:2) srcpos(:,3)-z0-2*zb],detpos)

s=4*D*v;
b=sqrt(3*mua*musp);

% unit of phi:  1/(mm^2)
Phi =v./(s*pi).*(exp(-b*r)./r - exp(-b*r2)./r2);
