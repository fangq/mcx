function Phi = tddiffusion(mua, musp, v, Reff, srcpos,detpos, t)
% Phi = tddiffusion(mua, musp, v, Reff, srcpos,detpos, t)
%
%  semi-infinite medium analytical solution to diffusion model
% 
%  Qianqian Fang <fangq at nmr.mgh.harvard.edu>
%
% see Boas2002

D = 1/(3*(mua+musp));
zb = (1+Reff)/(1-Reff)*2*D;

z0 = 1/(musp+mua);
r=getdistance([srcpos(:,1:2) srcpos(:,3)+z0],detpos);
r2=getdistance([srcpos(:,1:2) srcpos(:,3)-z0-2*zb],detpos);

s=4*D*v*t;

% unit of phi:  1/(mm^2*s)

Phi =v./((s*pi).^(3/2)).*exp(-mua*v*t).*(exp(-(r^2)./s) - exp(-(r2^2)./s));
